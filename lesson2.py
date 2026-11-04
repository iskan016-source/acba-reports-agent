from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from typing import Literal


client = genai.Client()

# Our "database". These are FAKE placeholder numbers for learning.
# Later we'll pull real numbers from the bank's reports.
FAKE_PROFITS = {
    ("acba", 2025): 42.5,   # billion AMD (made up)
    ("ameriabank", 2025): 55.0,
}

# TODO 1: write a function get_net_profit(bank: str, year: int) -> dict
#   - look up (bank.lower(), year) in FAKE_PROFITS
#   - if found: return {"status": "success", "net_profit_billion_amd": value}
#   - if not:   return {"status": "error", "message": "No data for that bank/year"}
#   - give it a docstring explaining what it does (Gemini READS the docstring!)

def get_net_profit(bank: Literal["acba", "ameriabank"], year: int) -> dict:
    """Returns a bank's net profit for a given year, in billion AMD."""
    bank = bank.lower()
    key = (bank, year)
    if key in FAKE_PROFITS:
        return {"status": "success", "net_profit_billion_amd": FAKE_PROFITS[key]}
    else:
        return {"status": "error", "message": "No data for that bank/year"}
    
    





# response = client.models.generate_content(
#     model="gemini-3.5-flash",
#     contents="What was Acba Bank's net profit in 2025?",
#     config=types.GenerateContentConfig(
#         temperature=0,
#         tools=[get_net_profit],
#         # We turn OFF auto-calling so you can SEE Gemini's request:
#         automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
#     ),
# )


# TODO 2: print what Gemini asked for.
#   Hint: response.function_calls is a list. Each item has .name and .args
# for call in response.function_calls:
#     print(call.name, call.args)

question = "What was Acba Bank's net profit in 2019?"

config = types.GenerateContentConfig(
    temperature=0,
    tools=[get_net_profit],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

response = client.models.generate_content(
    model="gemini-3.5-flash", contents=question, config=config
)


# 1. take the first tool request Gemini made
call = response.function_calls[0]

# 2. actually RUN our Python function with the arguments Gemini chose
# TODO A: one line. Hint: call.args is a dict like {'bank': 'acba', 'year': 2025}.
#         Python trick: f(**d) passes a dict's items as named arguments.
result = get_net_profit(**call.args)
print("Tool result:", result)

# 3. rebuild the full conversation (3 messages)
contents = [
    # message 1: the user's question
    types.Content(role="user", parts=[types.Part(text=question)]),
    # message 2: Gemini's tool request (exactly as Gemini sent it)
    response.candidates[0].content,
    # message 3: our tool's result
    types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=call.name, response=result)],
    ),
]

# 4. ask Gemini again, now with the result included
# TODO B: call generate_content again (same model and config, but contents=contents)
final = client.models.generate_content(
    model="gemini-3.5-flash",
    contents= contents,
    config=config,
)



# 5. print the final answer
# TODO C
print(final.text)
