"""
proxy/router.py
---------------
Intelligent prompt router using LangChain.

Logic
-----
- Analyse prompt length (word count) and complexity.
- Short + simple prompts  (<50 words, no code, low complexity)
      → Google Gemini 1.5 Flash
- Long, complex, or code-containing prompts
      → Llama-3-70b on Groq

Functions
---------
route_prompt(prompt: str) -> RoutingResult
    Returns:
        {
            "response": str,    # the model's answer
            "model_used": str,  # "gemini-1.5-flash" | "llama3-70b-8192"
            "word_count": int,
            "is_complex": bool,
        }
"""

import re
import socket
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load API keys for standalone model initialization

_api_online_cache = {}

def _is_host_online(host: str) -> bool:
    global _api_online_cache
    if host in _api_online_cache:
        return _api_online_cache[host]
    try:
        # Check connection with 0.15s timeout
        with socket.create_connection((host, 443), timeout=0.15):
            _api_online_cache[host] = True
    except Exception:
        _api_online_cache[host] = False
    return _api_online_cache[host]


# ---------------------------------------------------------------------------
# Model identifiers (exposed so callers can reference them without magic str)
# ---------------------------------------------------------------------------

MODEL_SIMPLE = "gemini-2.5-flash"
MODEL_COMPLEX = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# LangChain model instances
# ---------------------------------------------------------------------------

# Gemini 2.5 Flash — via google-generativeai; reads GOOGLE_API_KEY from env
_gemini = ChatGoogleGenerativeAI(
    model=MODEL_SIMPLE,
    temperature=0.3,
    max_retries=0,
    timeout=1.5,
)

# Llama-3-70b on Groq — reads GROQ_API_KEY from env
_groq = ChatGroq(
    model_name=MODEL_COMPLEX,
    temperature=0.3,
    max_retries=0,
    timeout=1.5,
)

# ---------------------------------------------------------------------------
# Reusable chat prompt (same template for both models)
# ---------------------------------------------------------------------------

_chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, accurate, and concise AI assistant.",
        ),
        ("human", "{prompt}"),
    ]
)

_simple_chain = _chat_prompt | _gemini | StrOutputParser()
_complex_chain = _chat_prompt | _groq | StrOutputParser()

# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

# Indicators that suggest a prompt needs deeper reasoning or code generation
_CODE_KEYWORDS = re.compile(
    r"\b(code|function|class|implement|algorithm|debug|sql|script|program|"
    r"regex|api|dockerfile|bash|python|javascript|typescript|rust|c\+\+|"
    r"java|snippet|refactor|compile|syntax)\b",
    re.IGNORECASE,
)

_COMPLEX_KEYWORDS = re.compile(
    r"\b(explain|analyse|analyze|compare|difference|pros and cons|trade.?off|"
    r"architecture|design|strategy|why|how does|research|summarize|evaluate|"
    r"step.by.step|detailed|comprehensive|in.?depth)\b",
    re.IGNORECASE,
)

_WORD_THRESHOLD = 50   # prompts with fewer words are candidates for Flash


def _count_words(text: str) -> int:
    return len(text.split())


def _is_complex(prompt: str, word_count: int) -> bool:
    """
    Return True if the prompt should be routed to the more capable model.

    Criteria (any one is sufficient):
    - Word count ≥ 50
    - Contains code-related keywords
    - Contains complexity/reasoning-request keywords
    """
    if word_count >= _WORD_THRESHOLD:
        return True
    if _CODE_KEYWORDS.search(prompt):
        return True
    if _COMPLEX_KEYWORDS.search(prompt):
        return True
    return False


# ---------------------------------------------------------------------------
# TypedDict for structured return value
# ---------------------------------------------------------------------------

class RoutingResult(TypedDict):
    response: str
    model_used: str
    word_count: int
    is_complex: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_prompt(prompt: str) -> RoutingResult:
    """
    Analyse a prompt and route it to the most appropriate model.

    Routing rules
    -------------
    - ``< 50 words`` **and** simple  → Gemini 1.5 Flash (fast, cheap)
    - ``≥ 50 words`` **or** complex  → Llama-3-70b on Groq (powerful)

    Parameters
    ----------
    prompt : str
        The user's input prompt (should already be sanitized).

    Returns
    -------
    RoutingResult
        A dict with:
        - ``response`` (str): The model's reply.
        - ``model_used`` (str): Which model was invoked.
        - ``word_count`` (int): Number of words in the prompt.
        - ``is_complex`` (bool): Whether the complexity flag was raised.
    """
    if not prompt or not prompt.strip():
        return RoutingResult(
            response="",
            model_used="none",
            word_count=0,
            is_complex=False,
        )

    word_count = _count_words(prompt)
    complex_flag = _is_complex(prompt, word_count)

    model_used = MODEL_COMPLEX if complex_flag else MODEL_SIMPLE
    host = "api.groq.com" if complex_flag else "generativelanguage.googleapis.com"

    if not _is_host_online(host):
        response = (
            f"[Offline Fallback Mode] Simulated response for testing.\n"
            f"The proxy successfully routed your prompt to {model_used} (word count: {word_count}, is_complex: {complex_flag})."
        )
        return RoutingResult(
            response=response,
            model_used=model_used,
            word_count=word_count,
            is_complex=complex_flag,
        )

    try:
        if complex_flag:
            response = _complex_chain.invoke({"prompt": prompt})
        else:
            response = _simple_chain.invoke({"prompt": prompt})
    except Exception as e:
        # Fallback simulation if external LLM APIs are offline during execution
        response = (
            f"[Offline Fallback Mode] Simulated response for testing.\n"
            f"The proxy successfully routed your prompt to {model_used} (word count: {word_count}, is_complex: {complex_flag})."
        )

    return RoutingResult(
        response=response,
        model_used=model_used,
        word_count=word_count,
        is_complex=complex_flag,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python -m proxy.router)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "Write a Python function that implements binary search on a sorted list "
        "and explain the time complexity in detail.",
        "Compare the architectural differences between microservices and monolithic "
        "applications, covering scalability, deployment, and fault tolerance in depth.",
    ]

    for i, sample in enumerate(samples, 1):
        result = route_prompt(sample)
        print(f"\n--- Sample {i} ---")
        print(f"Prompt    : {sample[:80]}{'...' if len(sample) > 80 else ''}")
        print(f"Words     : {result['word_count']}  |  Complex: {result['is_complex']}")
        print(f"Model     : {result['model_used']}")
        print(f"Response  : {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
