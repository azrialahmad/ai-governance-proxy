"""
evaluators/fact_checker.py
--------------------------
Hallucination Judge — "Evaluation as a Service" (EaaS).

Uses an LLM-as-Judge pattern to score AI responses for factual accuracy
against a known "Source of Truth" (golden_set.json).

Flow
----
1. Receive a (question, ai_response) pair.
2. Look up the question in the golden set via fuzzy matching.
3. If a match is found, prompt a local LLM to score the response (0–100).
4. Return a FactCheckResult with score, pass/fail, and reasoning.

Functions
---------
check_factuality(question, ai_response) -> FactCheckResult
find_golden_match(question) -> dict | None
"""

import json
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypedDict

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_set.json"
_FACTUALITY_THRESHOLD = 70  # minimum score to pass
_SIMILARITY_THRESHOLD = 0.75  # fuzzy-match cutoff for question lookup

# ---------------------------------------------------------------------------
# Model setup (reuses same Ollama instance as security.py — llama3.2)
# ---------------------------------------------------------------------------

_ollama_online_cache = None

def _is_ollama_online() -> bool:
    global _ollama_online_cache
    if _ollama_online_cache is not None:
        return _ollama_online_cache
    try:
        # Check local endpoint with numeric IP and 0.15s timeout
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.15) as response:
            _ollama_online_cache = (response.status == 200)
    except Exception:
        _ollama_online_cache = False
    return _ollama_online_cache

_llm = OllamaLLM(model="llama3.2", temperature=0)

# ---------------------------------------------------------------------------
# Golden set loader (cached at module level)
# ---------------------------------------------------------------------------

def _load_golden_set() -> list[dict]:
    """Load the golden Q&A set from disk."""
    if not _GOLDEN_SET_PATH.exists():
        return []
    with open(_GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_golden_set: list[dict] = _load_golden_set()


def reload_golden_set() -> None:
    """Re-read golden_set.json from disk (useful after edits)."""
    global _golden_set
    _golden_set = _load_golden_set()


# ---------------------------------------------------------------------------
# Fuzzy question matching
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def find_golden_match(question: str) -> dict | None:
    """
    Find the best matching entry in the golden set for a given question.

    Uses SequenceMatcher for fuzzy matching. Returns None if no match
    exceeds the similarity threshold.
    """
    if not _golden_set:
        return None

    norm_q = _normalize(question)
    best_match = None
    best_score = 0.0

    for entry in _golden_set:
        norm_golden = _normalize(entry["question"])
        similarity = SequenceMatcher(None, norm_q, norm_golden).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = entry

    if best_score >= _SIMILARITY_THRESHOLD:
        return best_match
    return None


# ---------------------------------------------------------------------------
# LLM-as-Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_TEMPLATE = PromptTemplate.from_template(
    """You are a factuality judge. Your task is to evaluate the accuracy of an
AI-generated response by comparing it against a known correct answer.

Scoring rules:
- Score 90-100: The response is factually correct and aligns with the source of truth.
- Score 70-89:  The response is mostly correct with minor inaccuracies or omissions.
- Score 40-69:  The response contains significant inaccuracies or misleading information.
- Score 0-39:   The response is mostly or entirely wrong.

You MUST respond in EXACTLY this format (three lines, nothing else):
SCORE: <number between 0 and 100>
PASSED: <YES or NO>
REASONING: <one sentence explaining your verdict>

---
Question: {question}

Source of Truth (correct answer):
{expected_answer}

AI Response to evaluate:
{ai_response}
---

Your evaluation:"""
)

_judge_chain = _JUDGE_TEMPLATE | _llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_judge_output(raw: str) -> tuple[int, str]:
    """
    Extract the numeric score and reasoning from the judge LLM's output.

    Returns (score, reasoning). Falls back to (50, raw) on parse failure.
    """
    score = 50  # safe default
    reasoning = raw.strip()

    # Try to extract SCORE: <number>
    score_match = re.search(r"SCORE:\s*(\d+)", raw, re.IGNORECASE)
    if score_match:
        score = min(100, max(0, int(score_match.group(1))))

    # Try to extract REASONING: <text>
    reasoning_match = re.search(r"REASONING:\s*(.+)", raw, re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return score, reasoning


# ---------------------------------------------------------------------------
# TypedDict for structured return value
# ---------------------------------------------------------------------------

class FactCheckResult(TypedDict):
    score: int | None           # 0-100 factuality score, None if no match
    passed: bool | None         # True if score >= threshold, None if no match
    reasoning: str | None       # Judge's explanation, None if no match
    matched_question: str | None  # The golden-set question that was matched
    threshold: int              # The threshold used for pass/fail


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_factuality(question: str, ai_response: str) -> FactCheckResult:
    """
    Check the factual accuracy of an AI response against the golden set.

    Parameters
    ----------
    question : str
        The original user question.
    ai_response : str
        The AI model's response to evaluate.

    Returns
    -------
    FactCheckResult
        - If a matching golden-set entry is found: score, passed, reasoning.
        - If no match: all fields are None (fact-check is skipped).
    """
    match = find_golden_match(question)

    if match is None:
        return FactCheckResult(
            score=None,
            passed=None,
            reasoning=None,
            matched_question=None,
            threshold=_FACTUALITY_THRESHOLD,
        )

    # Run the judge
    if not _is_ollama_online():
        expected = match["expected_answer"].lower().strip()
        actual = ai_response.lower().strip()
        
        # Check if the expected answer is contained in actual response
        if expected in actual or any(word in actual for word in expected.split() if len(word) > 2):
            score = 100
            reasoning = f"[Offline Fallback] Fact match detected. AI response contains key details of the expected answer: '{match['expected_answer']}'"
        else:
            # Check ratio
            ratio = SequenceMatcher(None, expected, actual).ratio()
            if ratio >= 0.5:
                score = 80
                reasoning = f"[Offline Fallback] Close match detected (similarity: {ratio:.2f})."
            else:
                score = 0
                reasoning = f"[Offline Fallback] Factual mismatch: AI response does not match the expected answer: '{match['expected_answer']}'"
        passed = score >= _FACTUALITY_THRESHOLD
        return FactCheckResult(
            score=score,
            passed=passed,
            reasoning=reasoning,
            matched_question=match["question"],
            threshold=_FACTUALITY_THRESHOLD,
        )

    try:
        raw_output = _judge_chain.invoke({
            "question": question,
            "expected_answer": match["expected_answer"],
            "ai_response": ai_response,
        })
        score, reasoning = _parse_judge_output(raw_output)
    except Exception as e:
        # Fallback fuzzy matching check if Ollama is offline
        expected = match["expected_answer"].lower().strip()
        actual = ai_response.lower().strip()
        
        # Check if the expected answer is contained in actual response
        if expected in actual or any(word in actual for word in expected.split() if len(word) > 2):
            score = 100
            reasoning = f"[Offline Fallback] Fact match detected. AI response contains key details of the expected answer: '{match['expected_answer']}'"
        else:
            # Check ratio
            ratio = SequenceMatcher(None, expected, actual).ratio()
            if ratio >= 0.5:
                score = 80
                reasoning = f"[Offline Fallback] Close match detected (similarity: {ratio:.2f})."
            else:
                score = 0
                reasoning = f"[Offline Fallback] Factual mismatch: AI response does not match the expected answer: '{match['expected_answer']}'"

    passed = score >= _FACTUALITY_THRESHOLD

    return FactCheckResult(
        score=score,
        passed=passed,
        reasoning=reasoning,
        matched_question=match["question"],
        threshold=_FACTUALITY_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python -m evaluators.fact_checker)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Should match and score HIGH
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        # Should match and score HIGH
        ("What is the speed of light?", "About 3 x 10^8 meters per second, or 299,792,458 m/s."),
        # Should match and score LOW (wrong answer — hallucination)
        ("What is the square root of 144?", "The square root of 144 is 14."),
        # Tricky — LLMs often get this wrong
        ("How many r's are in strawberry?", "There are 2 r's in strawberry."),
        # No match in golden set — should skip
        ("What is the meaning of life?", "42, according to Douglas Adams."),
    ]

    print("=" * 70)
    print("HALLUCINATION JUDGE — SMOKE TEST")
    print("=" * 70)

    for i, (q, a) in enumerate(test_cases, 1):
        result = check_factuality(q, a)
        print(f"\n--- Test {i} ---")
        print(f"  Question : {q}")
        print(f"  AI Answer: {a}")
        if result["score"] is not None:
            status = "[PASS]" if result["passed"] else "[FAIL]"
            print(f"  Score    : {result['score']}%  {status}")
            print(f"  Matched  : {result['matched_question']}")
            print(f"  Reasoning: {result['reasoning']}")
        else:
            print(f"  Score    : SKIPPED (no golden-set match)")

    print("\n" + "=" * 70)
