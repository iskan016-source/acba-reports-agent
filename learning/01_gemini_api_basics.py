"""Lesson 1: send a single prompt to Gemini and print the reply.

Without any retrieval, the model answers from its own training data, which is
why the main project grounds every answer in the bank's reports instead.
"""

from dotenv import load_dotenv

# Load credentials from .env before creating the client, which reads them from the environment.
load_dotenv()

from google import genai  # noqa: E402

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="What was Acba Bank’s net profit in 2025?",
)

print(response.text)
