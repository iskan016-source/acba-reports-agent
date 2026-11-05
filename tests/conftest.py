# Fake settings so tests run offline: no real Google login, no real secrets.
# These are set BEFORE the app is imported, so load_dotenv won't override them.
import os
os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "0"
os.environ["GOOGLE_API_KEY"] = "dummy-key-for-tests"
os.environ["BANK_API_KEY"] = "test-key"