"""
PagerSim-OpenEnv — Inference Script
Required by Meta PyTorch OpenEnv Hackathon.

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=your_key_here
    python3 inference.py
"""

from __future__ import annotations
import os
import json
import time
import sys
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Required hackathon environment variables ──────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

# ── Environment server ────────────────────────────────────────────────────────
SERVER_URL = os.environ.get("PAGERSIM_URL", "http://localhost:7860")
MAX_STEPS  = 12

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) responding to a production incident.
You will receive alerts, logs, and service statuses. Diagnose and resolve the incident efficiently.

At each step, respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON.

Required JSON structure:
{
  "action_type": <one of: "investigate_service", "escalate", "restart_service",
    "rollback_deployment", "check_dependencies", "silence_alert",
    "write_postmortem", "declare_resolved">,
  "target_service": <service name string or null>,
  "reasoning": <explanation, minimum 10 characters>,
  "postmortem": <null, or object with root_cause, timeline (list of 3+ strings),
    impact, resolution, prevention>
}

Strategy:
1. Investigate the most critical alerting service first
2. Check dependencies to understand service relationships
3. Look for recent deployments or config changes
4. Apply the correct fix (restart or rollback root cause service)
5. Write a postmortem with correct root cause before declaring resolved
6. Only declare_resolved after applying a fix AND submitting postmortem"""


def format_observation(obs: dict) -> str:
    """Format observation dict into readable text for the LLM."""
    alerts = "\n".join(
        f"  [{a['severity'].upper()}] {a['service_name']}: {a['message']}"
        for a in obs.get("alerts", [])
    )
    statuses = "\n".join(
        f"  {svc}: {state}"
        for svc, state in obs.get("current_status", {}).items()
    )
    logs = "\n".join(
        f"  {l['timestamp']} [{l['service']}] {l['level']}: {l['message']}"
        for l in obs.get("logs", [])[-10:]
    )
    hint = f"\nHINT: {obs['hint']}" if obs.get("hint") else ""

    return f"""=== INCIDENT: {obs.get('incident_id')} | Task: {obs.get('task_id')} | {obs.get('timestamp')} ===
Time: {obs.get('time_elapsed')}s / {obs.get('time_limit')}s

ALERTS:
{alerts}

SERVICE STATUS:
{statuses}

RECENT LOGS:
{logs}

ACTIONS TAKEN: {obs.get('actions_taken', [])}
{hint}"""


def parse_action(text: str) -> dict | None:
    """Parse LLM response into action dict. Returns None on failure."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            )
        result = json.loads(cleaned)
        if isinstance(result, dict) and "action_type" in result:
            return result
        return None
    except Exception:
        return None


def run_task(client: OpenAI, task_id: str) -> dict:
    """Run one complete task episode. Returns result dict."""
    print(f"\n{'='*55}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*55}")

    # Reset environment
    try:
        r = requests.post(f"{SERVER_URL}/reset", json={"task_id": task_id}, timeout=15)
        if r.status_code != 200:
            print(f"  ERROR: Reset failed HTTP {r.status_code}")
            return {"task_id": task_id, "final_score": 0.0, "steps": 0,
                    "time_seconds": 0.0, "success": False, "error": r.text}
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to {SERVER_URL}")
        return {"task_id": task_id, "final_score": 0.0, "steps": 0,
                "time_seconds": 0.0, "success": False, "error": "connection_refused"}

    obs          = r.json()
    messages     = [{"role": "system", "content": SYSTEM_PROMPT}]
    start_time   = time.time()
    steps        = 0
    final_score  = 0.0

    for step in range(MAX_STEPS):
        # Format and send to LLM
        user_content = format_observation(obs)
        messages.append({"role": "user", "content": user_content})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                stream=False,
            )
            assistant_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Step {step+1}: LLM request failed — {exc}")
            break

        messages.append({"role": "assistant", "content": assistant_text})

        # Parse action
        action_dict = parse_action(assistant_text)
        if not action_dict:
            # Retry once with correction
            messages.append({
                "role": "user",
                "content": "Invalid JSON. Reply ONLY with a JSON object matching the required schema."
            })
            try:
                retry = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800,
                    stream=False,
                )
                action_dict = parse_action(retry.choices[0].message.content or "")
            except Exception:
                pass

            if not action_dict:
                print(f"  Step {step+1}: Could not parse action, stopping")
                break

        action_type = action_dict.get("action_type", "?")
        target      = action_dict.get("target_service") or "-"
        print(f"  Step {step+1}: {action_type} → {target}")

        # Submit to environment
        try:
            r = requests.post(f"{SERVER_URL}/step", json=action_dict, timeout=15)
            if r.status_code != 200:
                print(f"  Step failed HTTP {r.status_code}: {r.text[:80]}")
                break
        except requests.exceptions.ConnectionError:
            print(f"  Lost connection to environment server")
            break

        result      = r.json()
        obs         = result["observation"]
        reward      = result["reward"]
        done        = result["done"]
        final_score = reward["cumulative_score"]
        steps       = step + 1

        sign = "+" if reward["score"] >= 0 else ""
        print(f"         reward: {sign}{reward['score']:.2f} | "
              f"cumulative: {final_score:.3f} | "
              f"{reward['feedback'][:55]}")

        if done:
            print(f"  Episode ended: {result.get('info', {}).get('reason', 'done')}")
            break

    elapsed = time.time() - start_time
    success = final_score >= 0.5

    return {
        "task_id":      task_id,
        "final_score":  round(max(0.0, min(1.0, final_score)), 3),
        "steps":        steps,
        "time_seconds": round(elapsed, 1),
        "success":      success,
    }


def main():
    """Run inference against all 3 tasks. Must complete under 20 minutes."""

    print("\n" + "="*55)
    print("  PAGERSIM-OPENENV INFERENCE")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Server: {SERVER_URL}")
    print("="*55)

    # Validate required env vars
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN (or OPENAI_API_KEY) not set")
        sys.exit(1)

    if not API_BASE_URL:
        print("ERROR: API_BASE_URL not set")
        sys.exit(1)

    # Check server health
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"ERROR: Server health check failed: {health.status_code}")
            sys.exit(1)
        print(f"\nServer healthy: {health.json()}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach server at {SERVER_URL}")
        print("Start it with: python3 -m uvicorn api.server:app --port 7860")
        sys.exit(1)

    # Initialize OpenAI client with hackathon-required variables
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    # Run all 3 tasks
    results = []
    total_start = time.time()

    for task_id in ["easy", "medium", "hard"]:
        result = run_task(client, task_id)
        results.append(result)
        time.sleep(1)

        # Safety check: abort if close to 20 min limit
        if time.time() - total_start > 1100:
            print("\nWARNING: Approaching 20min limit, stopping early")
            break

    # Print results table
    print("\n" + "="*55)
    print("  RESULTS")
    print("="*55)
    print(f"  {'Task':<10} {'Score':<8} {'Steps':<8} {'Time':<10} Status")
    print(f"  {'-'*48}")
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task_id']:<10} {r['final_score']:<8.3f} "
              f"{r['steps']:<8} {r['time_seconds']:<10.1f}s {status}")

    avg = sum(r["final_score"] for r in results) / max(len(results), 1)
    print(f"  {'-'*48}")
    print(f"  {'AVERAGE':<10} {avg:<8.3f}")
    print("="*55)

    # Save results
    output = {
        "results":    results,
        "average":    round(avg, 3),
        "model":      MODEL_NAME,
        "api_base":   API_BASE_URL,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to inference_results.json")

    return output


if __name__ == "__main__":
    main()