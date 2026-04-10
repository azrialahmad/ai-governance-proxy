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
    │       • < 50 words & simple  → Gemini Flash
    │       • complex / code       → Llama-3-70b on Groq
    │
    ├─ 4. fact_checker.check_factuality()
    │       • Compares response to golden-set source of truth
    │       • Score < 70% → block response (hallucination detected)
    │
    └─ 5. Return response + metadata to caller

Run locally
-----------
    uvicorn proxy.main:app --reload
"""

import time
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env before LangChain models are initialized

from proxy.security import sanitize_prompt
from proxy.router import route_prompt
from evaluators.fact_checker import check_factuality


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Gatekeeper Proxy",
    description=(
        "An AI governance proxy that sanitizes PII, detects jailbreak attempts, "
        "intelligently routes prompts, and judges responses for hallucinations."
    ),
    version="0.2.0",
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


class FactualityInfo(BaseModel):
    score: Optional[int] = None
    passed: Optional[bool] = None
    reasoning: Optional[str] = None
    matched_question: Optional[str] = None
    threshold: int = 70


class ChatResponse(BaseModel):
    response: str
    model_used: str
    redacted_prompt: str
    word_count: int
    is_complex: bool
    is_malicious: bool
    factuality: FactualityInfo
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
    4. **Fact-Check** – judge the response for hallucinations.
    5. **Respond** – return the model's answer plus pipeline metadata.
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

    # ── Step 4: Fact-Check ───────────────────────────────────────────────────
    fact_result = check_factuality(
        question=request.prompt,
        ai_response=routing_result["response"],
    )

    factuality = FactualityInfo(
        score=fact_result["score"],
        passed=fact_result["passed"],
        reasoning=fact_result["reasoning"],
        matched_question=fact_result["matched_question"],
        threshold=fact_result["threshold"],
    )

    # If a fact-check was performed and the response failed → block it
    if fact_result["passed"] is False:
        latency_ms = (time.perf_counter() - t_start) * 1000
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    "The AI is hallucinating; try again. "
                    f"(Factuality Score: {fact_result['score']}%)"
                ),
                "raw_ai_response": routing_result["response"],
                "redacted_prompt": redacted_prompt,
                "factuality": {
                    "score": fact_result["score"],
                    "threshold": fact_result["threshold"],
                    "reasoning": fact_result["reasoning"],
                    "matched_question": fact_result["matched_question"],
                },
                "latency_ms": round(latency_ms, 2),
            },
        )

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Step 5: Respond ──────────────────────────────────────────────────────
    return ChatResponse(
        response=routing_result["response"],
        model_used=routing_result["model_used"],
        redacted_prompt=redacted_prompt,
        word_count=routing_result["word_count"],
        is_complex=routing_result["is_complex"],
        is_malicious=is_malicious,
        factuality=factuality,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Direct run  (python proxy/main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("proxy.main:app", host="0.0.0.0", port=8000, reload=True)
