"""
evaluators/red_team.py
-----------------------
Adversarial Shield Red-Team runner.

Reads jailbreak prompts from data/jailbreaks.txt and fires them at the
Gatekeeper FastAPI application (using TestClient for offline/in-process execution).
Generates a safety defense report and saves it to data/red_team_report.json.

Run:
    python -m evaluators.red_team
"""

import json
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Import the FastAPI application
from proxy.main import app

# Config
_JAILBREAKS_PATH = Path(__file__).resolve().parent.parent / "data" / "jailbreaks.txt"
_REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "red_team_report.json"

def run_red_team():
    print("=" * 80)
    print("                  GATEKEEPER ADVERSARIAL SHIELD RUNNER                  ")
    print("=" * 80)

    if not _JAILBREAKS_PATH.exists():
        print(f"Error: Jailbreak prompts file not found at {_JAILBREAKS_PATH}")
        return

    # Read jailbreak prompts
    with open(_JAILBREAKS_PATH, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} jailbreak prompts for testing.")
    print("Initializing FastAPI TestClient...")
    
    client = TestClient(app)
    
    results = []
    blocked_count = 0
    bypassed_count = 0
    error_count = 0
    
    print("\nRunning safety evaluations...")
    print(f"{'ID':<4} | {'Prompt (truncated)':<45} | {'Verdict':<10} | {'Latency':<8}")
    print("-" * 80)

    for idx, prompt in enumerate(prompts, 1):
        t_start = time.perf_counter()
        verdict = "ERROR"
        latency = 0.0
        detail = ""
        
        try:
            response = client.post("/chat", json={"prompt": prompt})
            latency = (time.perf_counter() - t_start) * 1000
            
            if response.status_code == 400:
                verdict = "BLOCKED"
                blocked_count += 1
                detail = response.json().get("detail", "")
            elif response.status_code == 200:
                # If it got through to the model, it is a bypass
                verdict = "BYPASSED"
                bypassed_count += 1
                detail = response.json().get("response", "")[:100]
            else:
                verdict = "ERROR"
                error_count += 1
                detail = f"Status {response.status_code}: {response.text}"
                
        except Exception as e:
            latency = (time.perf_counter() - t_start) * 1000
            verdict = "ERROR"
            error_count += 1
            detail = str(e)
            
        results.append({
            "id": idx,
            "prompt": prompt,
            "verdict": verdict,
            "latency_ms": round(latency, 2),
            "detail": detail
        })
        
        # Format terminal display
        truncated_prompt = prompt if len(prompt) <= 42 else prompt[:42] + "..."
        if verdict == "BLOCKED":
            verdict_str = "\033[92mBLOCKED\033[0m"  # Green
        elif verdict == "BYPASSED":
            verdict_str = "\033[91mBYPASSED\033[0m"  # Red
        else:
            verdict_str = "\033[93mERROR\033[0m"  # Yellow
            
        print(f"{idx:<4} | {truncated_prompt:<45} | {verdict_str:<19} | {latency:.1f}ms")

    # Compute overall statistics
    total_runs = len(prompts)
    valid_runs = total_runs - error_count
    safety_rate = (blocked_count / valid_runs * 100) if valid_runs > 0 else 0.0
    bypass_rate = (bypassed_count / valid_runs * 100) if valid_runs > 0 else 0.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tested": total_runs,
        "blocked_caught": blocked_count,
        "passed_bypassed": bypassed_count,
        "errors": error_count,
        "safety_defense_rate_pct": round(safety_rate, 2),
        "bypass_rate_pct": round(bypass_rate, 2)
    }

    report = {
        "summary": summary,
        "results": results
    }

    # Write JSON report
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("                           RED-TEAM SUMMARY REPORT                      ")
    print("=" * 80)
    print(f"Total Prompts Tested : {total_runs}")
    print(f"Blocked (Caught)     : {blocked_count} ({safety_rate:.1f}%)")
    print(f"Passed (Bypassed)    : {bypassed_count} ({bypass_rate:.1f}%)")
    print(f"Errors (API Offline) : {error_count}")
    print("-" * 80)
    print(f"Safety Defense Rate  : {safety_rate:.2f}%")
    print(f"Report saved to      : {_REPORT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    run_red_team()
