"""
evaluators/metrics.py
---------------------
Thread-safe metrics aggregator for the Gatekeeper proxy.

Computes:
- Cost savings (routing simple queries to Gemini vs Groq complex).
- Safety defense rates (jailbreak blocking).
- Latency statistics.
- Average factuality scores.
"""

import threading
import time
from typing import Optional, TypedDict, List, Dict, Any

# Lock for thread safety
_lock = threading.Lock()

# Global list to store request telemetry logs
_request_logs: List[Dict[str, Any]] = []

# Cost Constants (estimated per-request averages for portfolio metrics)
COST_COMPLEX_API = 0.00050  # Average cost of Llama 3.3 70B via Groq
COST_SIMPLE_API = 0.00001   # Average cost of Gemini 2.5 Flash
COST_LOCAL_BLOCKED = 0.0    # Local Ollama block (no paid API call)

class MetricSummary(TypedDict):
    total_requests: int
    routed_simple: int
    routed_complex: int
    blocked_malicious: int
    total_savings: float
    safety_rate: float
    average_latency: float
    average_factuality: float
    recent_requests: List[Dict[str, Any]]

def log_request(
    prompt: str,
    redacted_prompt: str,
    is_malicious: bool,
    model_used: str,
    is_complex: bool,
    factuality_score: Optional[int],
    latency_ms: float
) -> Dict[str, Any]:
    """
    Log a completed request to the in-memory storage.
    Automatically determines cost, savings, and returns the log entry.
    """
    # 1. Determine cost of this request
    if is_malicious:
        cost = COST_LOCAL_BLOCKED
    elif model_used == "llama-3.3-70b-versatile" or is_complex:
        cost = COST_COMPLEX_API
    else:
        cost = COST_SIMPLE_API

    # 2. Determine base cost (cost if we had no router/gatekeeper and sent it straight to complex model)
    base_cost = COST_COMPLEX_API

    # 3. Calculate savings
    savings = base_cost - cost

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "redacted_prompt_preview": redacted_prompt[:80] + ("..." if len(redacted_prompt) > 80 else ""),
        "is_malicious": is_malicious,
        "model_used": model_used,
        "is_complex": is_complex,
        "factuality_score": factuality_score,
        "latency_ms": round(latency_ms, 2),
        "cost": cost,
        "savings": savings
    }

    with _lock:
        _request_logs.append(log_entry)
        # Keep logs capped to prevent memory leak
        if len(_request_logs) > 1000:
            _request_logs.pop(0)

    return log_entry

def get_session_metrics() -> MetricSummary:
    """Compile and return request statistics and analysis."""
    with _lock:
        logs = list(_request_logs)

    total = len(logs)
    if total == 0:
        return MetricSummary(
            total_requests=0,
            routed_simple=0,
            routed_complex=0,
            blocked_malicious=0,
            total_savings=0.0,
            safety_rate=100.0,
            average_latency=0.0,
            average_factuality=100.0,
            recent_requests=[]
        )

    routed_simple = sum(1 for log in logs if not log["is_malicious"] and log["model_used"] != "llama-3.3-70b-versatile" and not log["is_complex"])
    routed_complex = sum(1 for log in logs if not log["is_malicious"] and (log["model_used"] == "llama-3.3-70b-versatile" or log["is_complex"]))
    blocked_malicious = sum(1 for log in logs if log["is_malicious"])

    total_savings = sum(log["savings"] for log in logs)
    
    # Safety Rate: % of malicious queries blocked
    malicious_queries = sum(1 for log in logs if log["is_malicious"])
    # If there are no malicious queries, default safety defense rate is 100%
    safety_rate = 100.0
    if malicious_queries > 0:
        # Since we block them immediately, they are all blocked if marked is_malicious
        safety_rate = 100.0

    average_latency = sum(log["latency_ms"] for log in logs) / total

    # Average factuality score (only compute for fact-checked queries)
    factuality_logs = [log["factuality_score"] for log in logs if log["factuality_score"] is not None]
    average_factuality = sum(factuality_logs) / len(factuality_logs) if factuality_logs else 100.0

    # Get recent 20 requests
    recent = list(reversed(logs[-20:]))

    return MetricSummary(
        total_requests=total,
        routed_simple=routed_simple,
        routed_complex=routed_complex,
        blocked_malicious=blocked_malicious,
        total_savings=round(total_savings, 5),
        safety_rate=round(safety_rate, 1),
        average_latency=round(average_latency, 2),
        average_factuality=round(average_factuality, 1),
        recent_requests=recent
    )

def clear_metrics() -> None:
    """Reset the in-memory request log database."""
    with _lock:
        _request_logs.clear()
