from dotenv import load_dotenv
load_dotenv()                                   # load .env FIRST

import os
import time
import logging
from collections import defaultdict, deque      # NEW (for the rate limiter)
from pathlib import Path                        # NEW (to find static/index.html)

from fastapi import FastAPI, Header, HTTPException, Request          # CHANGED: + Request
from fastapi.responses import FileResponse                           # NEW
from pydantic import BaseModel, Field                                # CHANGED: + Field
from google.adk.runners import InMemoryRunner
from google.genai import types

from bank_agent.agent import root_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bank_api")

API_KEY = os.environ["BANK_API_KEY"]
APP_NAME = "bank_api"
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
app = FastAPI(title="Acba Reports Agent API")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)   # CHANGED: 1-500 characters, else 422
    session_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls: list[str]
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


# NEW: the pipeline code from /ask moved into ONE function, used by /ask AND /chat
async def run_agent(question: str, session_id: str | None) -> AskResponse:
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
        if event.content and event.content.parts and event.content.parts[0].text:
            answer = event.content.parts[0].text

    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info("ask session=%s tools=%s latency_ms=%d", session_id, tool_calls, latency_ms)
    return AskResponse(answer=answer, session_id=session_id, tool_calls=tool_calls, latency_ms=latency_ms)


# CHANGED: /ask is now just "check the key, then run the agent"
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return await run_agent(req.question, req.session_id)


# ── NEW: rate limiter for the public chat page ──
RATE_LIMIT = 10              # questions allowed...
WINDOW_SECONDS = 3600        # ...per hour, per visitor
recent = defaultdict(deque)  # ip -> times of that visitor's recent questions


def client_ip(request: Request) -> str:
    """The visitor's IP. On Cloud Run the real IP is in the X-Forwarded-For header."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str) -> None:
    now = time.time()
    times = recent[ip]                    # this visitor's recent question times (a deque)

    # 1. forget questions older than the window (they're always at the FRONT, since we add new ones at the back)
    while times and now - times[0] > WINDOW_SECONDS:
        times.popleft()

    # 2. still too many in the last hour? refuse
    if len(times) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many questions. Please try again later.")

    # 3. allowed: remember this question
    times.append(now)


# NEW: public chat endpoint (no key, rate-limited)
@app.post("/chat", response_model=AskResponse)
async def chat(req: AskRequest, request: Request):
    check_rate_limit(client_ip(request))
    return await run_agent(req.question, req.session_id)


# NEW: serve the chat page
@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")