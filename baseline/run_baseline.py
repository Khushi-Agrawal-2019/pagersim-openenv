from __future__ import annotations
import os
import json
import time
import sys
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:7860"
MODEL = "gpt-4o-mini"
MAX_STEPS = 12
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) responding to a production incident.
You will receive alerts, logs, and service statuses. Your job is to diagnose and resolve
the incident as efficiently as possible.

At each step, respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON.

The JSON must have this exact structure:
{
  "action_type": <one of: "investigate_service", "escalate", "restart_service", "rollback_deployment", "check_dependencies", "silence_alert", "write_postmortem", "declare_resolved">,
  "target_service": <service name string, or null if not applicable>,
  "reasoning": <your explanation, minimum 10 characters>,
  "postmortem": <null, or object with root_cause, timeline (list of 3+ strings), impact, resolution, prevention>
}

Strategy:
1. Start by investigating the most critical alerting services
2. Check dependencies to understand service relationships
3. Look for recent deployments or config changes in the logs
4. Apply the correct fix (restart or rollback the root cause service)
5. Write a postmortem with the correct root cause before declaring resolved
6. Only declare_resolved after applying a fix AND submitting a postmortem"""

def format_observation(obs_dict: dict) -> str:
    """Format observation dict into readable text for the LLM."""
    obs = obs_dict
    alerts_str = "".join([f"  [{alert['severity'].upper()}] {alert['service_name']}: {alert['message']}" for alert in obs['alerts']])
    service_status_str = "".join([f"  {service}: {status}" for service, status in obs['current_status'].items()])
    logs_str = "".join([f"  {log['timestamp']} [{log['service']}] {log['level']}: {log['message']}" for log in obs['logs'][-10:]])
    
    hint_str = f"💡 HINT: {obs['hint']}" if obs.get('hint') else ""

    return f"""=== INCIDENT: {obs['incident_id']} | Task: {obs['task_id']} | Time: {obs['timestamp']} ===
Time elapsed: {obs['time_elapsed']}s / {obs['time_limit']}s limit

ACTIVE ALERTS ({len(obs['alerts'])}):
{alerts_str}

SERVICE STATUS:
{service_status_str}

RECENT LOGS ({len(obs['logs'])} entries):
{logs_str}

ACTIONS TAKEN SO FAR: {obs['actions_taken']}

{hint_str}"""

def parse_action(response_text: str) -> dict | None:
    """Parse LLM response text into action dict. Returns None on failure."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split('\n', 1)[1]
            if text.endswith("```"):
                text = text[:-3].strip()
        return json.loads(text)
    except Exception:
        return None

def run_task(client: OpenAI, task_id: str) -> dict:
    """Run one task episode with the LLM agent. Returns result dict."""
    print(f"\n{'='*50}\nRunning task: {task_id.upper()}\n{'='*50}")

    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    if r.status_code != 200:
        return {"error": f"Reset failed: {r.status_code} - {r.text}"}

    obs = r.json()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    start_time = time.time()
    steps = 0
    final_score = 0.0

    for steps in range(MAX_STEPS):
        user_content = format_observation(obs)
        messages.append({"role": "user", "content": user_content})
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
        )
        assistant_text = response.choices[0].message.content
        
        messages.append({"role": "assistant", "content": assistant_text})
        
        action_dict = parse_action(assistant_text)
        
        if not action_dict:
            messages.append({"role": "user", "content": "Your response was not valid JSON. Reply ONLY with a JSON object matching the required schema."})
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
            )
            assistant_text = response.choices[0].message.content
            action_dict = parse_action(assistant_text)
            if not action_dict:
                print("WARNING: Could not parse action, skipping step")
                break
        
        print(f"  Step {steps+1}: {action_dict.get('action_type')} → {action_dict.get('target_service', '-')}")
        
        r = requests.post(f"{BASE_URL}/step", json=action_dict)
        if r.status_code != 200:
            print(f"ERROR: Step failed: {r.status_code} - {r.text}")
            break
        
        result = r.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        final_score = reward["cumulative_score"]
        
        print(f"         reward: {reward['score']:+.2f} | cumulative: {final_score:.2f} | {reward['feedback'][:60]}")
        
        if done:
            break

    elapsed = time.time() - start_time
    return {
        "task_id": task_id,
        "final_score": round(max(0.0, min(1.0, final_score)), 3),
        "steps": steps + 1,
        "time_seconds": round(elapsed, 1),
        "success": final_score >= 0.5
    }

def main():
    """Run baseline inference against all 3 tasks."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)

    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.exceptions.RequestException:
        print(f"ERROR: Server not running at {BASE_URL}")
        print("Start it with: python3 -m uvicorn api.server:app --port 7860")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    results = []
    for task_id in ["easy", "medium", "hard"]:
        result = run_task(client, task_id)
        results.append(result)
        time.sleep(1)

    print("\n" + "="*60)
    print("PAGERSIM-OPENENV BASELINE RESULTS")
    print("="*60)
    print(f"{'Task':<10} {'Score':<8} {'Steps':<8} {'Time':<10} {'Pass'}")
    print("-"*60)
    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"{r['task_id']:<10} {r['final_score']:<8.3f} {r['steps']:<8} {r['time_seconds']:<10.1f}s {status}")
    print("-"*60)
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"{'AVERAGE':<10} {avg:<8.3f}")
    print("="*60)

    with open("baseline/results.json", "w") as f:
        json.dump({"results": results, "average": round(avg, 3), "model": MODEL}, f, indent=2)
    print(f"\nResults saved to baseline/results.json")

if __name__ == "__main__":
    main()