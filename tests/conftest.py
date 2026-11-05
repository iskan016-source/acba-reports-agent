"""Test configuration: fake credentials so the suite runs offline.

These are set before the app is imported; load_dotenv does not override
variables that already exist, so these test values take precedence over a
local .env.
"""

import os

os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "0"
os.environ["GOOGLE_API_KEY"] = "dummy-key-for-tests"
os.environ["BANK_API_KEY"] = "test-key"
