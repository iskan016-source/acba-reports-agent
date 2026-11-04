# Step 1: load the settings from .env into the environment
from dotenv import load_dotenv
load_dotenv()

# Step 2: import the Gemini library
from google import genai

# TODO 1: create a client (one line). It reads your .env settings automatically.
client = genai.Client()

# TODO 2: send the question to the model "gemini-3.5-flash"
#   Hint: the method is under client.models and is called generate_content.
#   It takes two arguments: model=... and contents=...
response = client.models.generate_content(model = "gemini-3.5-flash", contents="What was Acba Bank’s net profit in 2025?")

# TODO 3: print only the answer text (hint: response has a .text attribute)
print(response.text)