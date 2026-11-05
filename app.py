"""FastAPI service for the Acba Reports Agent.

Endpoints:
  GET  /        public chat page (static/index.html)
  POST /chat    public, rate-limited question endpoint used by the chat page
  POST /ask     API-key protected endpoint for programmatic access and evaluation
  GET  /health  liveness check
"""

from dotenv import load_dotenv

# Load .env before importing the agent: the agent and the Gemini client read
# their settings from environment variables at import time.
load_dotenv()

import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict, deque  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from bank_agent.agent import root_agent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bank_api")

# Secrets come from the environment so they never end up in source control or the image.
API_KEY = os.environ["BANK_API_KEY"]
APP_NAME = "bank_api"

# Sessions are kept in memory: conversation history is lost on restart, which is
# acceptable for a demo and avoids running a database.
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
app = FastAPI(title="Acba Reports Agent API")


class AskRequest(BaseModel):
    """A question, optionally continuing an existing conversation."""

    # Bounded length keeps prompts (and costs) small; violations return 422.
    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = None


class AskResponse(BaseModel):
    """The validated answer plus diagnostics shown in the UI."""

    answer: str
    session_id: str
    tool_calls: list[str]
    latency_ms: int


@app.get("/health")
def health():
    """Liveness check for the hosting platform."""
    return {"status": "ok"}


async def run_agent(question: str, session_id: str | None) -> AskResponse:
    """Run the researcher -> analyst -> validator pipeline for one question.

    Creates a new session when session_id is None; raises 404 for an unknown session.
    """
    start = time.perf_counter()

    user_id = "api-user"
    if session_id is None:
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)
        session_id = session.id
    else:
        session = await runner.session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found. Omit session_id to start a new one.")

    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer = ""
    tool_calls = []
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        for call in event.get_function_calls():
            tool_calls.append(call.name)
        # The validator runs last, so the final text event is the validated answer.
        if event.content and event.content.parts and event.content.parts[0].text:
            answer = event.content.parts[0].text

    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info("ask session=%s tools=%s latency_ms=%d", session_id, tool_calls, latency_ms)
    return AskResponse(answer=answer, session_id=session_id, tool_calls=tool_calls, latency_ms=latency_ms)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    """Answer a question; requires the X-API-Key header."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return await run_agent(req.question, req.session_id)


# Per-visitor sliding-window rate limit for the public /chat endpoint, which has
# no API key. In-memory state is per instance, which is enough to cap abuse of a demo.
RATE_LIMIT = 10          # questions allowed...
WINDOW_SECONDS = 3600    # ...per hour, per visitor
recent = defaultdict(deque)  # ip -> timestamps of that visitor's recent questions


def client_ip(request: Request) -> str:
    """Return the visitor's IP. Behind Cloud Run's proxy it is the first X-Forwarded-For entry."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str) -> None:
    """Record a question from `ip`, or raise 429 if it exceeded RATE_LIMIT in the window."""
    now = time.time()
    times = recent[ip]

    # Timestamps are appended in order, so expired ones are always at the front.
    while times and now - times[0] > WINDOW_SECONDS:
        times.popleft()

    if len(times) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many questions. Please try again later.")

    times.append(now)


@app.post("/chat", response_model=AskResponse)
async def chat(req: AskRequest, request: Request):
    """Public question endpoint for the chat page: no key, rate-limited per IP."""
    check_rate_limit(client_ip(request))
    return await run_agent(req.question, req.session_id)


@app.get("/")
def home():
    """Serve the chat page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")
