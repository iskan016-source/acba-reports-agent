"""Lesson 2: the tool-calling loop done by hand.

Automatic function calling is disabled so each step is visible: the model
requests a tool call, we run the function ourselves, send the result back,
and the model writes the final answer. ADK automates this same loop.
"""

from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

client = genai.Client()

# Placeholder figures for the exercise, not real data.
FAKE_PROFITS = {
    ("acba", 2025): 42.5,   # billion AMD (made up)
    ("ameriabank", 2025): 55.0,
}


def get_net_profit(bank: Literal["acba", "ameriabank"], year: int) -> dict:
    """Returns a bank's net profit for a given year, in billion AMD."""
    key = (bank.lower(), year)
    if key in FAKE_PROFITS:
        return {"status": "success", "net_profit_billion_amd": FAKE_PROFITS[key]}
    return {"status": "error", "message": "No data for that bank/year"}


# A year with no data, to show how the model handles a tool error.
question = "What was Acba Bank's net profit in 2019?"

config = types.GenerateContentConfig(
    temperature=0,
    tools=[get_net_profit],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

response = client.models.generate_content(
    model="gemini-3.5-flash", contents=question, config=config
)

# Step 1: the model replies with a tool request instead of text.
call = response.function_calls[0]

# Step 2: run the requested function with the arguments the model chose.
result = get_net_profit(**call.args)
print("Tool result:", result)

# Step 3: rebuild the conversation: question, the model's tool request, our tool result.
contents = [
    types.Content(role="user", parts=[types.Part(text=question)]),
    response.candidates[0].content,
    types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=call.name, response=result)],
    ),
]

# Step 4: call the model again; with the tool result available it writes the answer.
final = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=contents,
    config=config,
)

print(final.text)
