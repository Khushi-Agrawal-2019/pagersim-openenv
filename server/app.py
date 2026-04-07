"""
PagerSim-OpenEnv — Gradio UI
FastAPI on port 8000. Gradio on port 7860.

Credentials from .env (local) or HF Secrets (deployed):
  HF_TOKEN     — OpenRouter or OpenAI API key
  MODEL_NAME   — e.g. openai/gpt-4o-mini  (OpenRouter) or gpt-4o-mini (OpenAI)
  API_BASE_URL — https://openrouter.ai/api/v1  OR  https://api.openai.com/v1
"""

from __future__ import annotations
import gradio as gr
import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Credentials ───────────────────────────────────────────────────────────────
API_PORT   = 7860
BASE_URL   = f"http://127.0.0.1:{API_PORT}"
HF_TOKEN   = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini")
API_BASE   = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_server() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


def fmt_alerts(alerts: list) -> str:
    if not alerts:
        return "_No active alerts._"
    icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    return "\n\n".join(
        f"{icons.get(a['severity'],'⚪')} **[{a['severity'].upper()}] {a['service_name']}**  \n"
        f"&nbsp;&nbsp;&nbsp;{a['message']}"
        for a in alerts
    )


def fmt_logs(logs: list) -> str:
    if not logs:
        return "_No logs yet._"
    icons = {"ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️", "DEBUG": "🔍"}
    return "\n".join(
        f"`{l['timestamp']}` {icons.get(l['level'],'•')} `[{l['service']}]` {l['message']}"
        for l in logs[-15:]
    )


def fmt_status(status: dict) -> str:
    if not status:
        return "_No services._"
    icons = {"up": "✅", "down": "🔴", "degraded": "🟠", "recovering": "🔄"}
    return "\n".join(
        f"{icons.get(state,'⚪')} **{svc}**: `{state}`"
        for svc, state in status.items()
    )


def score_bar(score: float) -> str:
    c = max(0.0, min(1.0, score))
    bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
    return f"`[{bar}]` **{c:.3f}** ({int(c*100)}%)"




# ── Human Play ────────────────────────────────────────────────────────────────

def human_reset(task_id: str):
    if not check_server():
        err = f"❌ **Backend offline.** Run: `python3 -m uvicorn api.server:app --port {API_PORT}`"
        return err, "_", "_", "_", score_bar(0.0), err, "{}"
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
        if r.status_code != 200:
            err = f"❌ Reset failed HTTP {r.status_code}: {r.text}"
            return err, "", "", "", score_bar(0.0), err, "{}"
        obs = r.json()
        info = (
            f"### 🚨 `{obs['incident_id']}` — {task_id.upper()}\n"
            f"**Time limit:** `{obs['time_limit']}s`  ·  Investigate the most critical alerting service first."
        )
        return (info, fmt_alerts(obs["alerts"]), fmt_status(obs["current_status"]),
                fmt_logs(obs["logs"]), score_bar(0.0),
                "✅ Episode started — take your first action below.", json.dumps(obs))
    except Exception as e:
        err = f"❌ {type(e).__name__}: {e}"
        return err, "", "", "", score_bar(0.0), err, "{}"


def human_step(obs_state, action_type, target_service, reasoning,
               root_cause, timeline, impact, resolution, prevention):
    if not obs_state or obs_state == "{}":
        msg = "⚠️ **No active episode.** Click **Start Episode** first."
        return msg, "_", "_", "_", score_bar(0.0), msg, "{}"
    if not reasoning or len(reasoning.strip()) < 10:
        cur = json.loads(obs_state) if obs_state != "{}" else {}
        return ("_", fmt_alerts(cur.get("alerts", [])), fmt_status(cur.get("current_status", {})),
                fmt_logs(cur.get("logs", [])), score_bar(0.0),
                "⚠️ **Reasoning required** — minimum 10 characters.", obs_state)
    action: dict = {
        "action_type": action_type,
        "target_service": target_service.strip() or None,
        "reasoning": reasoning.strip(),
        "postmortem": None,
    }
    if action_type == "write_postmortem":
        tl = [t.strip() for t in timeline.split("\n") if t.strip()]
        if len(tl) < 2:
            tl = ["T+0s: Incident detected", "T+60s: Root cause found", "T+120s: Fix applied"]
        action["postmortem"] = {
            "root_cause": root_cause.strip() or "unknown",
            "timeline": tl,
            "impact": impact.strip() or "Services degraded",
            "resolution": resolution.strip() or "Fix applied",
            "prevention": prevention.strip() or "Add monitoring",
        }
    try:
        r = requests.post(f"{BASE_URL}/step", json=action, timeout=10)
        if r.status_code != 200:
            err = f"❌ Step failed HTTP {r.status_code}: {r.text}"
            return err, "", "", "", score_bar(0.0), err, obs_state
        res    = r.json()
        obs    = res["observation"]
        reward = res["reward"]
        done   = res["done"]
        n      = len(obs["actions_taken"])
        tgt    = f":{target_service.strip()}" if target_service.strip() else ""
        emoji  = "📈" if reward["score"] >= 0 else "📉"
        fb = (
            f"**Step {n}:** `{action_type}{tgt}`\n\n"
            f"{emoji} Reward: `{reward['score']:+.2f}` | Cumulative: `{reward['cumulative_score']:.3f}`\n\n"
            f"💬 _{reward['feedback']}_"
        )
        if obs.get("hint"):
            fb += f"\n\n💡 **Hint:** {obs['hint']}"
        if done:
            f = reward["cumulative_score"]
            g = "🏆 Excellent!" if f>=0.8 else "✅ Good!" if f>=0.6 else "⚠️ Partial" if f>=0.4 else "❌ Needs work"
            fb += f"\n\n---\n## 🎬 Episode Complete!\n**Final Score: `{f:.3f}`** — {g}"
        info = (
            f"### 🚨 `{obs['incident_id']}` — {obs['task_id'].upper()}\n"
            f"**Time:** `{obs['time_elapsed']}s / {obs['time_limit']}s` · **Step:** `{n}`"
        )
        return (info, fmt_alerts(obs["alerts"]), fmt_status(obs["current_status"]),
                fmt_logs(obs["logs"]), score_bar(reward["cumulative_score"]), fb, json.dumps(obs))
    except Exception as e:
        err = f"❌ {type(e).__name__}: {e}"
        return err, "", "", "", score_bar(0.0), err, obs_state


def run_baseline_quick():
    if not check_server():
        return f"❌ Backend offline. Start FastAPI on port {API_PORT} first."
    try:
        r = requests.post(f"{BASE_URL}/baseline", timeout=60)
        if r.status_code != 200:
            return f"❌ Baseline failed HTTP {r.status_code}"
        data = r.json()
        md = "## 🤖 Rule-Based Baseline Results\n\n| Task | Score | Rating |\n|---|---|---|\n"
        for task, score in data["scores"].items():
            rt = "🏆" if score >= 0.8 else "✅" if score >= 0.6 else "⚠️"
            md += f"| **{task.capitalize()}** | `{score:.3f}` | {rt} |\n"
        md += f"\n**Average: `{data['average']:.3f}`** — Deterministic rule-based agent."
        return md
    except Exception as e:
        return f"❌ {e}"


# ── Agent Run ─────────────────────────────────────────────────────────────────

AGENT_PROMPT = """You are an expert SRE responding to a production incident.
At each step respond ONLY with valid JSON — no markdown, no text outside JSON.

{
  "action_type": <one of: investigate_service | escalate | restart_service |
                  rollback_deployment | check_dependencies | silence_alert |
                  write_postmortem | declare_resolved>,
  "target_service": <service name or null>,
  "reasoning": <explanation, min 10 chars>,
  "postmortem": <null or {root_cause, timeline (3+ items), impact, resolution, prevention}>
}

Strategy: investigate critical services → check dependencies → apply fix → postmortem → declare_resolved."""


def obs_to_text(obs: dict) -> str:
    alerts   = "\n".join(f"  [{a['severity'].upper()}] {a['service_name']}: {a['message']}" for a in obs.get("alerts", []))
    statuses = "\n".join(f"  {svc}: {state}" for svc, state in obs.get("current_status", {}).items())
    logs     = "\n".join(f"  {l['timestamp']} [{l['service']}] {l['level']}: {l['message']}" for l in obs.get("logs", [])[-10:])
    hint     = f"\nHINT: {obs['hint']}" if obs.get("hint") else ""
    return (f"INCIDENT: {obs.get('incident_id')} | {obs.get('task_id','').upper()} | "
            f"{obs.get('time_elapsed')}s/{obs.get('time_limit')}s\n\n"
            f"ALERTS:\n{alerts}\n\nSERVICE STATUS:\n{statuses}\n\n"
            f"RECENT LOGS:\n{logs}\n\nACTIONS SO FAR: {obs.get('actions_taken', [])}{hint}")


def parse_action(text: str) -> dict | None:
    try:
        cleaned = text.strip()
        if "```" in cleaned:
            cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```"))
        result = json.loads(cleaned)
        if isinstance(result, dict) and "action_type" in result:
            return result
    except Exception:
        pass
    return None


def run_agent_episode(task_id: str, progress=gr.Progress()):
    """Generator — yields live updates as the agent works."""

    if not check_server():
        yield (f"❌ **Backend offline.**\n```\npython3 -m uvicorn api.server:app --port {API_PORT}\n```",
               "_", "_", score_bar(0.0), "❌ Backend offline")
        return

    api_key  = HF_TOKEN
    model    = MODEL_NAME
    api_base = API_BASE

    if not api_key:
        yield (
            "❌ **No API credentials found.**\n\n"
            "**Local:** Add to `.env`:\n"
            "```\nHF_TOKEN=sk-or-v1-your-openrouter-key\n"
            "MODEL_NAME=openai/gpt-4o-mini\n"
            "API_BASE_URL=https://openrouter.ai/api/v1\n```\n\n"
            "**HF Spaces:** Space Settings → Variables and Secrets → add `HF_TOKEN`",
            "_", "_", score_bar(0.0), "❌ No credentials",
        )
        return

    # Show which provider we're using
    provider = "OpenRouter" if "openrouter" in api_base else "OpenAI"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)
    except Exception as e:
        yield (f"❌ Failed to init LLM client: {e}", "_", "_", score_bar(0.0), "❌ Init error")
        return

    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
        if r.status_code != 200:
            yield (f"❌ Reset failed HTTP {r.status_code}", "_", "_", score_bar(0.0), "❌ Reset failed")
            return
        obs = r.json()
    except Exception as e:
        yield (f"❌ {e}", "_", "_", score_bar(0.0), "❌ Connection error")
        return

    SCENARIO_NAMES = {"easy": "Database Overload", "medium": "Cascading Auth Failure", "hard": "Rate Limiter Poisoning"}
    scenario_name = SCENARIO_NAMES.get(task_id, task_id)

    messages   = [{"role": "system", "content": AGENT_PROMPT}]
    step_log   = []
    cumulative = 0.0
    max_steps  = 15
    start_ts   = time.time()

    def build_log(extra: str = "") -> str:
        header = (
            f"## 🤖 AI Agent — `{scenario_name}` ({task_id.upper()})\n\n"
            f"**Provider:** {provider} &nbsp;·&nbsp; **Model:** `{model}` &nbsp;·&nbsp; "
            f"**Incident:** `{obs.get('incident_id','')}` &nbsp;·&nbsp; "
            f"**Time limit:** `{obs.get('time_limit','')}s`\n\n"
            f"---\n### 🔔 Incident Alerts\n{fmt_alerts(obs.get('alerts', []))}\n\n---\n"
        )
        return header + "\n".join(step_log) + ("\n" + extra if extra else "")

    yield (
        f"## 🤖 AI Agent Starting — `{scenario_name}`\n\n"
        f"**Incident:** `{obs.get('incident_id')}` · **Time limit:** `{obs.get('time_limit')}s`\n\n"
        f"**Affected:** {', '.join(f'`{s}`' for s in obs.get('current_status', {}).keys())}\n\n"
        f"---\n### 🔔 Active Alerts\n{fmt_alerts(obs.get('alerts', []))}\n\n"
        f"---\n*Agent analyzing the incident...*",
        fmt_status(obs.get("current_status", {})),
        fmt_logs(obs.get("logs", [])),
        score_bar(0.0),
        "🔄 Agent initializing...",
    )
    time.sleep(0.5)

    for step in range(max_steps):
        progress((step + 1) / max_steps, desc=f"Step {step + 1}/{max_steps}")
        messages.append({"role": "user", "content": obs_to_text(obs)})

        yield (build_log(f"\n⏳ **Step {step+1}:** Agent thinking..."),
               fmt_status(obs.get("current_status", {})),
               fmt_logs(obs.get("logs", [])),
               score_bar(cumulative),
               f"🔄 Step {step+1} — thinking...")

        try:
            completion = client.chat.completions.create(
                model=model, messages=messages, temperature=0.1, max_tokens=800, stream=False)
            assistant_text = completion.choices[0].message.content or ""
        except Exception as exc:
            # Show helpful message for 401
            err_str = str(exc)
            if "401" in err_str or "invalid_api_key" in err_str:
                msg = (
                    f"\n❌ **Step {step+1}: Authentication failed (401)**\n\n"
                    f"Your API key was rejected. Check:\n"
                    f"- Is `HF_TOKEN` in your `.env` set to your **OpenRouter** key (`sk-or-v1-...`)?\n"
                    f"- Is `API_BASE_URL` set to `https://openrouter.ai/api/v1`?\n"
                    f"- Is `MODEL_NAME` set to `openai/gpt-4o-mini` (with the `openai/` prefix)?\n\n"
                    f"Current config: base=`{api_base}` model=`{model}`"
                )
            else:
                msg = f"\n❌ **Step {step+1}: LLM error** — `{exc}`"
            step_log.append(msg)
            yield (build_log(), fmt_status(obs.get("current_status", {})),
                   fmt_logs(obs.get("logs", [])), score_bar(cumulative), "❌ LLM error")
            break

        messages.append({"role": "assistant", "content": assistant_text})

        action_dict = parse_action(assistant_text)
        if not action_dict:
            messages.append({"role": "user", "content": "Reply ONLY with a JSON object."})
            try:
                retry = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.1, max_tokens=800)
                action_dict = parse_action(retry.choices[0].message.content or "")
                messages.append({"role": "assistant", "content": retry.choices[0].message.content or ""})
            except Exception:
                pass

        if not action_dict:
            step_log.append(f"\n❌ **Step {step+1}:** Could not parse agent response.")
            yield (build_log(), fmt_status(obs.get("current_status", {})),
                   fmt_logs(obs.get("logs", [])), score_bar(cumulative), "❌ Parse failed")
            break

        action_type = action_dict.get("action_type", "?")
        target      = action_dict.get("target_service") or "—"
        reasoning   = action_dict.get("reasoning", "")

        try:
            r = requests.post(f"{BASE_URL}/step", json=action_dict, timeout=10)
            if r.status_code != 200:
                step_log.append(f"\n❌ **Step {step+1}:** Server error {r.status_code}")
                break
            result     = r.json()
            obs        = result["observation"]
            reward     = result["reward"]
            done       = result["done"]
            cumulative = reward["cumulative_score"]
            step_n     = len(obs["actions_taken"])
        except Exception as e:
            step_log.append(f"\n❌ **Step {step+1}:** {e}")
            break

        sign      = "+" if reward["score"] >= 0 else ""
        emoji     = "📈" if reward["score"] >= 0 else "📉"
        tgt_str   = f" → `{target}`" if target != "—" else ""

        step_log.append(
            f"\n---\n"
            f"### Step {step_n} · `{action_type}`{tgt_str}\n\n"
            f"**🧠 Agent's Reasoning:**\n> _{reasoning}_\n\n"
            f"{emoji} **Reward:** `{sign}{reward['score']:.2f}` &nbsp;·&nbsp; "
            f"**Cumulative:** `{cumulative:.3f}`\n\n"
            f"💬 **Environment feedback:** _{reward['feedback']}_"
            + (f"\n\n💡 **Hint:** {obs['hint']}" if obs.get("hint") else "")
        )

        yield (build_log(), fmt_status(obs.get("current_status", {})),
               fmt_logs(obs.get("logs", [])), score_bar(cumulative), f"✅ Step {step_n} — {action_type}")

        if done:
            break
        time.sleep(0.2)

    elapsed = round(time.time() - start_ts, 1)
    f_score = cumulative
    grade, g_icon = (
        ("Excellent",         "🏆") if f_score >= 0.8 else
        ("Good",              "✅") if f_score >= 0.6 else
        ("Partial Success",   "⚠️") if f_score >= 0.4 else
        ("Needs Improvement", "❌")
    )
    step_log.append(
        f"\n---\n## {g_icon} Episode Complete — {grade}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| **Final Score** | `{f_score:.3f} / 1.000` |\n"
        f"| **Steps** | `{len(step_log)}` |\n"
        f"| **Time** | `{elapsed}s` |\n"
        f"| **Task** | `{task_id.upper()}` |\n"
        f"| **Model** | `{model}` |\n"
    )
    yield (build_log(), fmt_status(obs.get("current_status", {})),
           fmt_logs(obs.get("logs", [])), score_bar(f_score), f"{g_icon} Done — {f_score:.3f}")


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { background: #0d1117 !important; max-width: 100% !important; }
footer { display: none !important; }

/* Tabs */
.tab-nav button {
    font-weight: 700 !important; font-size: 13px !important;
    color: #8b949e !important; background: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    padding: 12px 20px !important;
}
.tab-nav button.selected { color: #f85149 !important; border-bottom-color: #f85149 !important; }

/* Labels */
label span {
    color: #8b949e !important; font-size: 11px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Inputs */
input, textarea, select {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #c9d1d9 !important; border-radius: 6px !important; font-size: 13px !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #f85149 !important;
    box-shadow: 0 0 0 3px rgba(248,81,73,0.1) !important;
}

/* ── Scenario Radio — styled as cards ── */
.scenario-radio .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr 1fr !important;
    gap: 12px !important;
}
.scenario-radio .wrap label {
    display: block !important;
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    position: relative !important;
}
.scenario-radio .wrap label:hover {
    border-color: rgba(248,81,73,0.4) !important;
    background: #1c2128 !important;
}
.scenario-radio .wrap label:has(input:checked) {
    border: 2px solid #f85149 !important;
    background: #1c2128 !important;
    box-shadow: 0 0 12px rgba(248,81,73,0.2) !important;
}
/* Hide the actual radio dot */
.scenario-radio .wrap input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}
/* The text shown inside each radio label */
.scenario-radio .wrap .wrap { display: block !important; }

/* Buttons */
.primary-btn button {
    background: #f85149 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 700 !important; font-size: 14px !important;
    text-transform: uppercase !important;
}
.primary-btn button:hover { background: #da3633 !important; }

.agent-btn button {
    background: linear-gradient(135deg, #f85149, #ff9a00) !important;
    color: #fff !important; border: none !important; border-radius: 6px !important;
    font-weight: 900 !important; font-size: 15px !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    height: 52px !important; box-shadow: 0 4px 15px rgba(248,81,73,0.3) !important;
}
.agent-btn button:hover { transform: translateY(-1px) !important; }

.secondary-btn button {
    background: transparent !important; color: #8b949e !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-weight: 600 !important; font-size: 13px !important;
}
.secondary-btn button:hover { border-color: #f85149 !important; color: #f85149 !important; }

/* Panels */
.panel-md {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 8px !important; padding: 12px 16px !important; min-height: 80px;
}
.panel-md p, .panel-md li { color: #c9d1d9 !important; font-size: 13px !important; }
.panel-md code {
    background: #0d1117 !important; color: #79c0ff !important;
    padding: 2px 6px !important; border-radius: 4px !important; font-size: 12px !important;
}
.panel-md strong { color: #e6edf3 !important; }

/* Agent log */
.agent-log {
    background: #0d1117 !important; border: 1px solid #21262d !important;
    border-radius: 8px !important; padding: 20px 24px !important;
    min-height: 500px !important; font-size: 13px !important; line-height: 1.8 !important;
}
.agent-log h2 { color: #f85149 !important; font-size: 18px !important; }
.agent-log h3 {
    color: #79c0ff !important; font-size: 14px !important;
    border-left: 3px solid #79c0ff; padding-left: 10px; margin: 20px 0 8px !important;
}
.agent-log blockquote {
    border-left: 3px solid #f85149 !important; padding-left: 12px !important;
    color: #8b949e !important; margin: 8px 0 !important;
}
.agent-log table { width: 100% !important; border-collapse: collapse !important; margin: 12px 0 !important; }
.agent-log th {
    background: #161b22 !important; color: #8b949e !important;
    font-size: 11px !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important; padding: 8px 12px !important;
}
.agent-log td { padding: 8px 12px !important; border-bottom: 1px solid #21262d !important; color: #c9d1d9 !important; }

/* Score */
.score-panel {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 8px !important; padding: 12px 16px !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }
"""

EXPLAINER = """
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
<div style="background:linear-gradient(135deg,#161b22,#0d1117);border:1px solid #30363d;
            border-radius:12px;padding:24px 28px;margin-bottom:8px;font-family:'Space Grotesk',sans-serif">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px">
        <span style="font-size:36px">🚨</span>
        <div>
            <div style="font-size:22px;font-weight:700;color:#e6edf3">PagerSim-OpenEnv</div>
            <div style="font-size:11px;color:#8b949e;margin-top:3px;text-transform:uppercase;letter-spacing:0.12em">
                SRE Incident Response · AI Agent Training Environment · Meta PyTorch Hackathon
            </div>
        </div>
        <div style="margin-left:auto;background:rgba(196,136,1,0.15);border:1px solid rgba(255,186,64,0.3);
                    padding:6px 14px;border-radius:20px;font-size:11px;font-weight:700;
                    color:#ffba40;text-transform:uppercase">Meta PyTorch Hackathon</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px">
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px">
            <div style="color:#f85149;font-weight:700;font-size:11px;text-transform:uppercase;margin-bottom:8px">🎯 What Is This?</div>
            <div style="color:#8b949e;font-size:12px;line-height:1.6">A simulation where AI agents learn to handle production outages like a real PagerDuty on-call incident.</div>
        </div>
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px">
            <div style="color:#f85149;font-weight:700;font-size:11px;text-transform:uppercase;margin-bottom:8px">🤖 Agent Run Tab</div>
            <div style="color:#8b949e;font-size:12px;line-height:1.6">Watch an AI agent work live every step, its exact reasoning, and how the environment rewards each action.</div>
        </div>
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px">
            <div style="color:#f85149;font-weight:700;font-size:11px;text-transform:uppercase;margin-bottom:8px">🧑‍💻 Human Play Tab</div>
            <div style="color:#8b949e;font-size:12px;line-height:1.6">Play it yourself. Investigate services, find the root cause, fix it, write a postmortem. Compare your score to the AI.</div>
        </div>
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px">
            <div style="color:#f85149;font-weight:700;font-size:11px;text-transform:uppercase;margin-bottom:8px">📊 Scoring</div>
            <div style="color:#8b949e;font-size:12px;line-height:1.6">Correct investigation +0.15 · Right fix +0.20 · Good postmortem +0.35 · Perfect = 1.0. Dense partial rewards.</div>
        </div>
    </div>
</div>
"""

# ── Scenario labels for Radio (shown as cards via CSS) ─────────────────────────
# Each string is multi-line — Gradio renders it inside the label element
# CSS above hides the radio dot and styles the label as a card

SCENARIO_CHOICES = [
    "easy",
    "medium",
    "hard",
]


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="PagerSim-OpenEnv", css=CSS, theme=gr.themes.Base()) as demo:

    obs_state = gr.State("{}")
    gr.HTML(EXPLAINER)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — AGENT RUN
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🤖 Agent Run — Watch AI Solve Incidents Live"):

            gr.Markdown("""
### How it works
**Click a scenario** below to select it, then click **Deploy Agent**.
The AI reads alerts, investigates services, reasons through the problem,
applies the fix, and writes a postmortem — completely autonomously.
Every step of its thinking is shown in real time.
""")

            # Scenario info cards (visual only — for context)
            gr.HTML("""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:4px;font-family:'Space Grotesk',sans-serif">
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px">
        <div style="color:#3fb950;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">● EASY · 5 min · 2 services</div>
        <div style="color:#e6edf3;font-weight:700;font-size:14px;margin-bottom:5px">Database Overload</div>
        <div style="color:#8b949e;font-size:12px;line-height:1.5">Postgres connection pool exhausted. payment-service down. Clear log trail.</div>
    </div>
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px">
        <div style="color:#f0883e;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">● MEDIUM · 10 min · 3 services</div>
        <div style="color:#e6edf3;font-weight:700;font-size:14px;margin-bottom:5px">Cascading Auth Failure</div>
        <div style="color:#8b949e;font-size:12px;line-height:1.5">Auth service memory leak causes cascade. api-gateway is a victim, not the cause.</div>
    </div>
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px">
        <div style="color:#f85149;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">● HARD · 15 min · 5 services</div>
        <div style="color:#e6edf3;font-weight:700;font-size:14px;margin-bottom:5px">Rate Limiter Poisoning</div>
        <div style="color:#8b949e;font-size:12px;line-height:1.5">Bad config throttles all traffic. payment-service deploy is a red herring trap.</div>
    </div>
</div>
""")

            # Actual functional selector
            agent_task = gr.Radio(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Select Scenario",
                elem_classes=["scenario-radio"],
            )

            with gr.Row(elem_classes=["agent-btn"]):
                agent_run_btn = gr.Button(
                    "🚀  Deploy Agent — Start Autonomous Incident Response",
                    size="lg",
                )

            agent_status = gr.Markdown(
                "_Select a scenario above and click Deploy Agent._"
            )

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 🧠 Agent Activity Log — Live")
                    gr.Markdown(
                        "_Every action appears here in real time: "
                        "what the agent observed, what it decided, "
                        "its exact reasoning, and what reward it received._"
                    )
                    agent_log = gr.Markdown(
                        "**Waiting for agent...**\n\nClick **Deploy Agent** above.",
                        elem_classes=["agent-log"],
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Live Score")
                    agent_score = gr.Markdown(score_bar(0.0), elem_classes=["score-panel"])
                    gr.Markdown("### 🖥️ Service Status")
                    agent_svc   = gr.Markdown("_Not started_", elem_classes=["panel-md"])
                    gr.Markdown("### 📋 Revealed Logs")
                    agent_logs  = gr.Markdown("_No logs yet_", elem_classes=["panel-md"])

            with gr.Accordion("📖 Reward Reference", open=False):
                gr.Markdown("""
| Action | Condition | Score |
|---|---|---|
| `investigate_service` | Correct service | **+0.15** |
| `check_dependencies` | Correct service | **+0.10** |
| `restart_service` / `rollback_deployment` | Correct service | **+0.20** |
| `write_postmortem` | Correct root cause + quality | **+0.35** |
| `declare_resolved` | After fix + postmortem | **+0.25** |
| Wrong restart/rollback | — | **-0.10** |
| Redundant investigation | — | **-0.05** |
| Timeout / premature resolve | — | **-0.10 to -0.15** |
""")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — HUMAN PLAY
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🧑‍💻 Human Play — Try It Yourself"):

            gr.Markdown("""
### You are the on-call SRE
Production is broken. Read the alerts, investigate services, find the root cause, apply the fix.
Compare your score against the AI agent.
""")

            # Info cards (visual)
            gr.HTML("""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:4px;font-family:'Space Grotesk',sans-serif">
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px">
        <div style="color:#3fb950;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">● EASY · 5 min</div>
        <div style="color:#e6edf3;font-weight:700;font-size:14px;margin-bottom:5px">Database Overload</div>
        <div style="color:#8b949e;font-size:12px">Root cause: <code style="background:#0d1117;color:#79c0ff;padding:2px 5px;border-radius:3px">database_connection_pool_exhausted</code></div>
    </div>
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px">
        <div style="color:#f0883e;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">● MEDIUM · 10 min</div>
        <div style="color:#e6edf3;font-weight:700;font-size:14px;margin-bottom:5px">Cascading Auth Failure</div>
        <div style="color:#8b949e;font-size:12px">Root cause: <code style="background:#0d1117;color:#79c0ff;padding:2px 5px;border-radius:3px">auth_service_memory_leak_bad_deployment</code></div>
    </div>
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px">
        <div style="color:#f85149;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">● HARD · 15 min</div>
        <div style="color:#e6edf3;font-weight:700;font-size:14px;margin-bottom:5px">Rate Limiter Poisoning</div>
        <div style="color:#8b949e;font-size:12px">Root cause: <code style="background:#0d1117;color:#79c0ff;padding:2px 5px;border-radius:3px">api_gateway_rate_limiter_config_poisoning</code></div>
    </div>
</div>
""")

            # Functional selector
            task_picker = gr.Radio(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Select Task",
                elem_classes=["scenario-radio"],
            )

            with gr.Row():
                with gr.Column(scale=3, elem_classes=["primary-btn"]):
                    reset_btn = gr.Button("🚀 Start Episode", size="lg")
                with gr.Column(scale=2, elem_classes=["secondary-btn"]):
                    baseline_btn = gr.Button("🤖 Run Rule-Based Baseline Agent")
                    baseline_out = gr.Markdown("_Click to run the deterministic baseline._")

            incident_info = gr.Markdown("_Select a task above and click Start Episode._")
            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔔 Active Alerts")
                    alerts_box = gr.Markdown("_No alerts._", elem_classes=["panel-md"])
                    gr.Markdown("### 🖥️ Service Status")
                    status_box = gr.Markdown("_No services._", elem_classes=["panel-md"])
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Service Logs")
                    logs_box = gr.Markdown("_No logs yet._", elem_classes=["panel-md"])

            gr.Markdown("### 📊 Episode Score")
            score_display = gr.Markdown(score_bar(0.0), elem_classes=["score-panel"])
            gr.Markdown("---")
            gr.Markdown("### ⚡ Take Action")

            with gr.Row():
                with gr.Column(scale=2):
                    action_type = gr.Dropdown(
                        choices=["investigate_service", "check_dependencies",
                                 "restart_service", "rollback_deployment",
                                 "escalate", "silence_alert",
                                 "write_postmortem", "declare_resolved"],
                        value="investigate_service",
                        label="Action Type",
                        interactive=True,
                    )
                with gr.Column(scale=2):
                    target_service = gr.Textbox(
                        label="Target Service",
                        placeholder="e.g. payment-service, postgres-db, auth-service",
                    )

            reasoning = gr.Textbox(
                label="Reasoning (required — min 10 characters)",
                placeholder="Explain why you are taking this action...",
                lines=2,
            )

            with gr.Accordion("📝 Postmortem Fields (expand for write_postmortem)", open=False):
                root_cause = gr.Textbox(label="Root Cause", placeholder="e.g. database_connection_pool_exhausted")
                timeline   = gr.Textbox(label="Timeline (one per line, min 2)", lines=3,
                                         placeholder="T+0s: Alerts fired\nT+60s: Root cause found\nT+120s: Fix applied")
                with gr.Row():
                    impact     = gr.Textbox(label="Impact",     placeholder="What was affected?")
                    resolution = gr.Textbox(label="Resolution", placeholder="How was it fixed?")
                    prevention = gr.Textbox(label="Prevention", placeholder="How to prevent?")

            with gr.Row(elem_classes=["primary-btn"]):
                step_btn = gr.Button("▶️ Submit Action", size="lg")

            gr.Markdown("### 📜 Last Action Result")
            history_box = gr.Markdown("_No actions taken yet._", elem_classes=["panel-md"])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — ARCHITECTURE
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🏗️ System Architecture"):
            gr.Markdown("""
## PagerSim-OpenEnv Architecture

PagerSim-OpenEnv is designed as a high-fidelity simulation of Site Reliability Engineering (SRE) operations. It bridges the gap between traditional LLM benchmarks and real-world operational complexity.

### 🧩 Core Components

1.  **Simulation Engine (Backend)**:
    *   **State Machine**: Manages the lifecycle of an incident across multiple services.
    *   **Service Mesh Emulator**: Simulates dependencies, health checks, and cascading failures.
    *   **Fault Injector**: Dynamically introduces specific root causes (e.g., memory leaks, config poisoning) based on the selected task.

2.  **Observation Layer**:
    *   Synthesizes realistic "noisy" data: Prometheus-style alerts, structured application logs, and service status maps.
    *   Implements "reveal" mechanics: Logs are hidden until an agent takes the `investigate_service` action, simulating the cost of context switching and search in real SRE work.

3.  **Reward System**:
    *   **Dense Shaping**: Provides incremental rewards for investigative steps (checking dependencies, investigating the right service).
    *   **Outcome-Based**: Large rewards for applying the correct fix and writing a technically accurate postmortem.
    *   **Penalty-Driven**: Differentiates between "unnecessary" actions and "harmful" actions (like restarting the wrong service during a peak incident).


### 🎯 Design Philosophy: "Operational Reasoning"
Most OpenEnv environments focus on web navigation or code editing. PagerSim-OpenEnv focuses on **Operational Reasoning**: the ability to distinguish between **symptoms** (api-gateway down) and **root causes** (auth-service memory leak).
""")

    # ── Wire events ───────────────────────────────────────────────────────────

    agent_run_btn.click(
        fn=run_agent_episode,
        inputs=[agent_task],
        outputs=[agent_log, agent_svc, agent_logs, agent_score, agent_status],
    )

    reset_btn.click(
        fn=human_reset,
        inputs=[task_picker],
        outputs=[incident_info, alerts_box, status_box, logs_box,
                 score_display, history_box, obs_state],
    )

    step_btn.click(
        fn=human_step,
        inputs=[obs_state, action_type, target_service, reasoning,
                root_cause, timeline, impact, resolution, prevention],
        outputs=[incident_info, alerts_box, status_box, logs_box,
                 score_display, history_box, obs_state],
    )

    baseline_btn.click(fn=run_baseline_quick, inputs=[], outputs=[baseline_out])


# ── Unified App for HF Spaces ───────────────────────────────────────────────
from api.server import app as fastapi_app

# Mount Gradio at / directly. 
# FastAPI routes in fastapi_app (like /reset) will take precedence.
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

def main():
    import uvicorn
    # Use 7860 as the main port for everything
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()