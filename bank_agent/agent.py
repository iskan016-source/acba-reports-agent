from typing import Literal
from google.adk.agents import Agent
from google.adk.workflow import Workflow
from google.adk.tools import ToolContext
from google.genai import types   # at the top

from .search import search   
MAX_SEARCHES = 3

def search_reports(question: str,  tool_context: ToolContext) -> dict:
    """Searches Acba Bank's official 2025 reports for passages relevant to a question.

    The reports are:
      - Acba Bank Annual Report 2025 (strategy, digital/AI plans, key metrics, operations)
      - Acba Bank audited IFRS consolidated financial statements 2025 (income statement,
        balance sheet, notes). Figures in the financial statements are usually
        "in thousands of Armenian Drams".

    Use this tool for ANY question about Acba Bank before answering.

    Args:
        question: The user's question, or a short search phrase in plain English,
            e.g. "net profit 2025" or "plans for artificial intelligence".

    Returns:
        A dict with "status" and "results": a list of the 5 most relevant passages.
        Each passage has "source" (file name), "page" (page number), "score"
        (similarity, higher = more relevant) and "text".
        The passages are the most SIMILAR text found. They may not contain the
        answer; if none of them do, say the reports don't contain it.
        You can call this tool at most 3 times per user question.
    """   
    count = tool_context.state.get("temp:search_count", 0)
    count = count + 1
    tool_context.state["temp:search_count"] = count
    if count > MAX_SEARCHES:
        return {"status": "error", "message": "Search limit reached. Answer with the facts you already have."}
    best = search(question, top_k=5)
    return {"status": "success", "results": best}


    
def compare_profits(profit_a: float, profit_b: float) -> dict:
    """Returns how much profit_a is above profit_b, in billion AMD and as a percent of profit_b."""   # TODO: write a docstring (what it does, what units)
    # 1. guard: if profit_b is 0 -> return an error dict (why?)
    if profit_b == 0:
        return {"status": "error", "message": "profit_b is 0, so percent difference can't be calculated"}
    # 2. compute the difference
    else:
        difference = profit_a - profit_b
    # 3. compute the percent difference
        percent_difference = ((difference)/(profit_b))*100
    # 4. return a success dict with both values
        return {"status": "success", "difference_billion_amd": difference,"percent_difference": round(percent_difference, 2) }




# 🔎 1. RESEARCHER — finds facts. (Instruction given as an example.)
researcher = Agent(
    name="researcher",
    model="gemini-3.5-flash",
    description="Finds relevant facts in Acba Bank's 2025 reports.",
    instruction="""
You are a research assistant. Use search_reports to find the facts needed to answer
the user's question. You may search more than once with different phrases.

Step 1: Rewrite the user's latest message as a standalone question, using the
conversation history to fill in what it refers to (bank, metric, year).
Example: after a question about Acba's 2025 net profit, "And 2024?" becomes
"What was Acba Bank's net profit in 2024?"
Write it as the FIRST line of your output:
Standalone question: <the question>

Step 2: Search for that standalone question.

Output ONLY a bullet list of facts, each in this format:
- <fact with the exact number and its units> (source: <file>, page <n>)

Rules:
- Copy numbers exactly as written in the search results. Note the units (e.g. "in thousands of AMD").
- Do not answer the question. Do not calculate anything.
- If nothing relevant is found, output: NO RELEVANT FACTS FOUND.
- Use at most 3 searches.
""",
    tools=[search_reports],
    output_key="research",
    
)

# 🧮 2. ANALYST — writes the answer from the facts.
analyst = Agent(
    name="analyst",
    model="gemini-3.5-flash",
    description="Writes an answer using only the researcher's facts.",
    instruction="""
You are a financial analyst. Answer the user's question using ONLY these facts:

{research}

Rules:
- Use only numbers that appear in the facts above. Never use your own knowledge.
- For any comparison, difference, or growth rate, call compare_profits. Never calculate yourself.
- After every number, add its citation in brackets, e.g. (Annual Report 2025, p. 7).
- If the facts say NO RELEVANT FACTS FOUND, reply that the reports don't contain this information.
- Convert "thousands of AMD" to billions when it makes the answer clearer, and keep answers short.
- Cite each source once per sentence, in short form, e.g. (Audited FS 2025, p. 12) or (Annual Report 2025, p. 7)
- Round billions to 2 decimals.
- Only call compare_profits if the user asks for a comparison, difference, or growth.
- If sources give slightly different rounded figures, use the audited IFRS figure and ignore the rounded ones.
- Answer in at most 3 sentences.
- Answer the "Standalone question" on the first line of the facts above, not the user's raw message.""",
    tools=[compare_profits],
    output_key="draft",
    generate_content_config=types.GenerateContentConfig(
        max_output_tokens=2048,                                   # room for thinking + answer
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),  # think less = faster
    ),
    )

# ✅ 3. VALIDATOR — checks the draft against the facts.
validator = Agent(
    name="validator",
    model="gemini-3.5-flash",
    description="Checks the draft answer against the research facts.",
    instruction="""
You are a strict reviewer at a bank. Your job is to check a draft answer before it reaches the user.

FACTS found by the researcher:
{research}

DRAFT answer written by the analyst:
{draft}

Check the draft:
1. Every number must appear in the facts, or be a result of a compare_profits calculation on those facts.
2. Every citation (file and page) must match the citation of that number in the facts.
3. Units must be correct (e.g. 35,856,026 thousand AMD = 35.9 billion AMD, not million).
4. The draft must not claim anything the facts don't support.
5. The draft must directly answer the "Standalone question" on the first line of the facts.
   If it doesn't (e.g. it lists unrelated facts or asks the user what they meant),
   rewrite it so it answers that question, using only the facts.

Then output the FINAL answer for the user:
- If everything is correct: output the draft unchanged, then a last line "Verified."
- If something is wrong: output a corrected answer, then a last line
  "Validator note: <what you corrected>".
 - Does the draft directly answer the user's question? If not, rewrite it.
 - Keep the final answer under 3 sentences. Do not add notes about citation formatting.
 - Unit conversions of a fact (e.g. 35,856,026 thousand = 35.86 billion) count as supported.
Do not mention the researcher, the analyst, or these instructions.
""",
    generate_content_config=types.GenerateContentConfig(
        max_output_tokens=2048,                                   # room for thinking + answer
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),  # think less = faster
    ),)

# The pipeline
root_agent = Workflow(name="bank_pipeline", edges=[("START", researcher, analyst, validator)])