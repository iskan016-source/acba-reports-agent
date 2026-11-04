# ── 1. Load .env FIRST (before importing the agent, which reads settings) ──
from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from google.adk.runners import InMemoryRunner
from google.genai import types

from bank_agent.agent import root_agent   # your 3-agent pipeline

# ── 2. Logging: print one line per request with time + level ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bank_api")

API_KEY = os.environ["BANK_API_KEY"]
APP_NAME = "bank_api"

# The Runner runs your agent. "InMemory" = sessions are kept in RAM (lost on restart).
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
app = FastAPI(title="Acba Reports Agent API")


# ── 3. The SHAPE of requests and responses ──
class AskRequest(BaseModel):
    question: str
    session_id: str | None = None   # None = start a new conversation

class AskResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls: list[str]
    latency_ms: int


# ── 4. Health check: other systems ping this to see if we're alive ──
@app.get("/health")
def health():
    return {"status": "ok"}


# ── 5. The main endpoint ──
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    # FastAPI reads the "X-API-Key" header into the variable x_api_key.

    # TODO 1: if x_api_key is not equal to API_KEY:
    #           raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail = "Invalid or missing API key")


    start = time.perf_counter()     # stopwatch start

    # Session: reuse the caller's session, or create a new one
    user_id = "api-user"
    if req.session_id is None:
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)
        session_id = session.id
    else:
        session = await runner.session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=req.session_id
        )

        if session is  None:
            raise HTTPException(status_code=404, detail ="Session not found. Omit session_id to start a new one.")
        session_id = req.session_id

    message = types.Content(role="user", parts=[types.Part(text=req.question)])

    answer = ""
    tool_calls = []
    # Run the pipeline. It produces a stream of EVENTS, the same ones you saw in the Events tab.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        # TODO 2: record tool calls.
        #   event.get_function_calls() returns a list (often empty).
        #   For each call in it, append call.name to tool_calls.
        for call in event.get_function_calls():
            tool_calls.append(call.name)

        # Keep the latest text: the validator's text comes last, so it ends up as the answer
        if event.content and event.content.parts and event.content.parts[0].text:
            answer = event.content.parts[0].text

    # TODO 3: compute latency in milliseconds.
    #   time.perf_counter() - start  gives SECONDS (a float). Convert to an int of ms.
    latency_ms = int((time.perf_counter() - start) * 1000)

    # Monitoring: one log line per request
    log.info("ask session=%s tools=%s latency_ms=%d", session_id, tool_calls, latency_ms)

    return AskResponse(answer=answer, session_id=session_id,
                       tool_calls=tool_calls, latency_ms=latency_ms)