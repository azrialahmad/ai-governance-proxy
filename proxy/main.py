"""
proxy/main.py
-------------
FastAPI entry-point for the Gatekeeper proxy.

Flow
----
POST /chat  { "prompt": "<user text>" }
    │
    ├─ 1. security.sanitize_prompt()
    │       • Redacts PII  (names, emails, phone numbers → [REDACTED])
    │       • Detects jailbreak attempts
    │
    ├─ 2. [Guard] If is_malicious → 400 Bad Request (prompt rejected)
    │
    ├─ 3. router.route_prompt()
    │       • < 50 words & simple  → Gemini 1.5 Flash
    │       • complex / code       → Llama-3-70b on Groq
    │
    └─ 4. Return response + metadata to caller

Run locally
-----------
    uvicorn proxy.main:app --reload
"""

import time
from typing import Annotated

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env before LangChain models are initialized

from proxy.security import sanitize_prompt
from proxy.router import route_prompt


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Gatekeeper Proxy",
    description=(
        "An AI proxy that sanitizes PII, detects jailbreak attempts, "
        "and intelligently routes prompts to the most suitable model."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PromptRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=8000, examples=["Tell me a joke."])]


class ChatResponse(BaseModel):
    response: str
    model_used: str
    redacted_prompt: str
    word_count: int
    is_complex: bool
    is_malicious: bool
    latency_ms: float


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health_check():
    """Liveness probe."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Malicious prompt rejected"},
        422: {"description": "Validation error"},
    },
    tags=["chat"],
    summary="Send a prompt through the Gatekeeper pipeline",
)
def chat(request: PromptRequest) -> ChatResponse:
    """
    Full pipeline:

    1. **Sanitize** – strip PII and check for jailbreak intent.
    2. **Guard** – reject the request if a jailbreak is detected.
    3. **Route** – forward the clean prompt to the right model.
    4. **Respond** – return the model's answer plus pipeline metadata.
    """
    t_start = time.perf_counter()

    # ── Step 1: Sanitize ────────────────────────────────────────────────────
    security_result = sanitize_prompt(request.prompt)
    redacted_prompt: str = security_result["redacted_prompt"]
    is_malicious: bool = security_result["is_malicious"]

    # ── Step 2: Guard ────────────────────────────────────────────────────────
    if is_malicious:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Your prompt was flagged as a potential jailbreak attempt "
                "and has been rejected."
            ),
        )

    # ── Step 3: Route ────────────────────────────────────────────────────────
    routing_result = route_prompt(redacted_prompt)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Step 4: Respond ──────────────────────────────────────────────────────
    return ChatResponse(
        response=routing_result["response"],
        model_used=routing_result["model_used"],
        redacted_prompt=redacted_prompt,
        word_count=routing_result["word_count"],
        is_complex=routing_result["is_complex"],
        is_malicious=is_malicious,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Direct run  (python proxy/main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("proxy.main:app", host="0.0.0.0", port=8000, reload=True)
