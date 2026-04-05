"""
PagerSim-OpenEnv — Gradio UI
Pure Gradio frontend. Talks to FastAPI backend via HTTP only.
FastAPI runs on port 8000. Gradio runs on port 7860.
"""

from __future__ import annotations
import gradio as gr
import requests
import json
import os

# FastAPI backend URL — separate process on port 8000
API_PORT = int(os.environ.get("API_PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{API_PORT}"


# ── helpers ───────────────────────────────────────────────────────────────────

def check_server() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def format_alerts(alerts: list) -> str:
    if not alerts:
        return "_No active alerts._"
    icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    lines = []
    for a in alerts:
        icon = icons.get(a["severity"], "⚪")
        lines.append(
            f"{icon} **[{a['severity'].upper()}] {a['service_name']}**  \n"
            f"&nbsp;&nbsp;&nbsp;{a['message']}"
        )
    return "\n\n".join(lines)


def format_logs(logs: list) -> str:
    if not logs:
        return "_No logs yet._"
    icons = {"ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️", "DEBUG": "🔍"}
    lines = []
    for log in logs[-15:]:
        icon = icons.get(log["level"], "•")
        lines.append(
            f"`{log['timestamp']}` {icon} `[{log['service']}]` {log['message']}"
        )
    return "\n".join(lines)


def format_status(status: dict) -> str:
    if not status:
        return "_No services._"
    icons = {"up": "✅", "down": "🔴", "degraded": "🟠", "recovering": "🔄"}
    lines = []
    for service, state in status.items():
        icon = icons.get(state, "⚪")
        lines.append(f"{icon} **{service}**: `{state}`")
    return "\n".join(lines)


def format_score_bar(score: float) -> str:
    clamped = max(0.0, min(1.0, score))
    filled = int(clamped * 20)
    bar = "█" * filled + "░" * (20 - filled)
    pct = int(clamped * 100)
    return f"`[{bar}]` **{clamped:.3f}** ({pct}%)"


# ── API functions ─────────────────────────────────────────────────────────────

def do_reset(task_id: str):
    if not check_server():
        err = (
            f"❌ **Cannot reach FastAPI backend on port {API_PORT}.**\n\n"
            f"Open a terminal and run:\n"
            f"```\nsource venv/bin/activate\n"
            f"python3 -m uvicorn api.server:app --host 0.0.0.0 --port {API_PORT}\n```"
        )
        return err, "_No alerts_", "_No services_", "_No logs_", format_score_bar(0.0), err, "{}"

    try:
        r = requests.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=10
        )
        if r.status_code != 200:
            err = f"❌ Reset failed (HTTP {r.status_code}): {r.text}"
            return err, "", "", "", format_score_bar(0.0), err, "{}"

        obs = r.json()
        info_md = (
            f"### 🚨 Incident: `{obs['incident_id']}`\n"
            f"**Task:** `{task_id.upper()}` &nbsp;|&nbsp; "
            f"**Time limit:** `{obs['time_limit']}s`"
        )
        alerts_md  = format_alerts(obs["alerts"])
        status_md  = format_status(obs["current_status"])
        logs_md    = format_logs(obs["logs"])
        score_md   = format_score_bar(0.0)
        history_md = (
            f"✅ **Episode started — {task_id.upper()} task loaded.**\n\n"
            f"You have `{obs['time_limit']}s` to resolve the incident.\n\n"
            f"**Start by investigating the most critical alerting service.**"
        )
        return info_md, alerts_md, status_md, logs_md, score_md, history_md, json.dumps(obs)

    except requests.exceptions.ConnectionError:
        err = f"❌ Connection refused on port {API_PORT}. Is FastAPI running?"
        return err, "", "", "", format_score_bar(0.0), err, "{}"
    except Exception as e:
        err = f"❌ Error: {type(e).__name__}: {str(e)}"
        return err, "", "", "", format_score_bar(0.0), err, "{}"


def do_step(
    obs_state: str,
    action_type: str,
    target_service: str,
    reasoning: str,
    root_cause: str,
    timeline: str,
    impact: str,
    resolution: str,
    prevention: str,
):
    if not obs_state or obs_state == "{}":
        msg = "⚠️ **No active episode.** Click **Start Episode** first."
        return msg, "_No alerts_", "_No services_", "_No logs_", format_score_bar(0.0), msg, "{}"

    if not reasoning or len(reasoning.strip()) < 10:
        try:
            current_obs = json.loads(obs_state)
        except Exception:
            current_obs = {}
        msg = "⚠️ **Reasoning is required** — must be at least 10 characters."
        return (
            "_",
            format_alerts(current_obs.get("alerts", [])),
            format_status(current_obs.get("current_status", {})),
            format_logs(current_obs.get("logs", [])),
            format_score_bar(0.0),
            msg,
            obs_state,
        )

    action: dict = {
        "action_type": action_type,
        "target_service": target_service.strip() if target_service.strip() else None,
        "reasoning": reasoning.strip(),
        "postmortem": None,
    }

    if action_type == "write_postmortem":
        timeline_list = [t.strip() for t in timeline.split("\n") if t.strip()]
        if len(timeline_list) < 2:
            timeline_list = [
                "T+0s: Incident detected via monitoring alerts",
                "T+60s: Root cause identified through log investigation",
                "T+120s: Fix applied and services recovering",
            ]
        action["postmortem"] = {
            "root_cause": root_cause.strip() if root_cause.strip() else "unknown root cause",
            "timeline": timeline_list,
            "impact": impact.strip() if impact.strip() else "Services degraded affecting users",
            "resolution": resolution.strip() if resolution.strip() else "Fix applied to affected service",
            "prevention": prevention.strip() if prevention.strip() else "Add monitoring and alerts",
        }

    try:
        r = requests.post(f"{BASE_URL}/step", json=action, timeout=10)

        if r.status_code != 200:
            err = f"❌ Step failed (HTTP {r.status_code}): {r.text}"
            return err, "", "", "", format_score_bar(0.0), err, obs_state

        result = r.json()
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]

        alerts_md = format_alerts(obs["alerts"])
        status_md = format_status(obs["current_status"])
        logs_md   = format_logs(obs["logs"])
        score_md  = format_score_bar(reward["cumulative_score"])

        step_num     = len(obs["actions_taken"])
        score_emoji  = "📈" if reward["score"] >= 0 else "📉"
        target_label = f":{target_service.strip()}" if target_service.strip() else ""

        history_md = (
            f"**Step {step_num}:** `{action_type}{target_label}`\n\n"
            f"{score_emoji} Step reward: `{reward['score']:+.2f}` &nbsp;|&nbsp; "
            f"Cumulative: `{reward['cumulative_score']:.3f}`\n\n"
            f"💬 _{reward['feedback']}_"
        )

        if obs.get("hint"):
            history_md += f"\n\n💡 **Hint:** {obs['hint']}"

        if done:
            final = reward["cumulative_score"]
            if final >= 0.8:
                grade = "🏆 Excellent work!"
            elif final >= 0.6:
                grade = "✅ Good job!"
            elif final >= 0.4:
                grade = "⚠️ Partial success"
            else:
                grade = "❌ Needs improvement"

            history_md += (
                f"\n\n---\n"
                f"## 🎬 Episode Complete!\n"
                f"**Final Score: `{final:.3f}`** — {grade}\n\n"
                f"Click **Start Episode** to try again."
            )

        info_md = (
            f"### 🚨 Incident: `{obs['incident_id']}`\n"
            f"**Task:** `{obs['task_id'].upper()}` &nbsp;|&nbsp; "
            f"**Time:** `{obs['time_elapsed']}s / {obs['time_limit']}s` &nbsp;|&nbsp; "
            f"**Step:** `{step_num}`"
        )
        if obs.get("hint"):
            info_md += f"\n\n💡 **Hint:** {obs['hint']}"

        return (
            info_md, alerts_md, status_md, logs_md,
            score_md, history_md, json.dumps(obs)
        )

    except requests.exceptions.ConnectionError:
        err = f"❌ Lost connection to backend on port {API_PORT}."
        return err, "", "", "", format_score_bar(0.0), err, obs_state
    except Exception as e:
        err = f"❌ Error: {type(e).__name__}: {str(e)}"
        return err, "", "", "", format_score_bar(0.0), err, obs_state


def do_baseline():
    if not check_server():
        return f"❌ Cannot reach backend on port {API_PORT}. Start FastAPI first."
    try:
        r = requests.post(f"{BASE_URL}/baseline", timeout=60)
        if r.status_code != 200:
            return f"❌ Baseline failed (HTTP {r.status_code}): {r.text}"

        data   = r.json()
        scores = data["scores"]
        avg    = data["average"]

        md  = "## 🤖 Baseline Agent Results\n\n"
        md += "| Task | Score | Bar | Rating |\n|---|---|---|---|\n"
        for task, score in scores.items():
            filled = int(score * 10)
            bar    = "█" * filled + "░" * (10 - filled)
            rating = "🏆" if score >= 0.8 else "✅" if score >= 0.6 else "⚠️"
            md += f"| **{task.capitalize()}** | `{score:.3f}` | `{bar}` | {rating} |\n"
        md += f"\n**Average Score: `{avg:.3f}`**\n\n"
        md += "_Deterministic rule-based agent following optimal action sequences._"
        return md

    except Exception as e:
        return f"❌ Error: {type(e).__name__}: {str(e)}"


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="PagerSim-OpenEnv",
    theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange"),
) as demo:

    obs_state = gr.State("{}")

    gr.Markdown("""
# 🚨 PagerSim-OpenEnv
**SRE Incident Response Simulation · OpenEnv Environment · Meta PyTorch Hackathon**

An AI agent training environment where agents act as on-call SRE engineers —
reading alerts, investigating services, finding root causes, and resolving incidents.
""")

    with gr.Row():
        with gr.Column(scale=3):
            task_picker = gr.Radio(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="📋 Select Task Difficulty",
                info=(
                    "Easy: DB overload (2 services) | "
                    "Medium: Auth cascade (3 services) | "
                    "Hard: Config poisoning + red herring (5 services)"
                ),
            )
            reset_btn = gr.Button("🚀 Start Episode", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Baseline Agent")
            baseline_btn = gr.Button("Run Baseline on All 3 Tasks", variant="secondary")
            baseline_out = gr.Markdown("_Runs a rule-based agent against all tasks._")

    incident_info = gr.Markdown("_Select a task and click Start Episode to begin._")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔔 Active Alerts")
            alerts_box = gr.Markdown("_No alerts._")
            gr.Markdown("### 🖥️ Service Status")
            status_box = gr.Markdown("_No services._")

        with gr.Column(scale=2):
            gr.Markdown("### 📋 Service Logs")
            logs_box = gr.Markdown("_No logs yet._")

    gr.Markdown("### 📊 Episode Score")
    score_display = gr.Markdown(format_score_bar(0.0))

    gr.Markdown("---")
    gr.Markdown("### ⚡ Take Action")

    with gr.Row():
        with gr.Column(scale=2):
            action_type = gr.Dropdown(
                choices=[
                    "investigate_service",
                    "check_dependencies",
                    "restart_service",
                    "rollback_deployment",
                    "escalate",
                    "silence_alert",
                    "write_postmortem",
                    "declare_resolved",
                ],
                value="investigate_service",
                label="Action Type",
            )
        with gr.Column(scale=2):
            target_service = gr.Textbox(
                label="Target Service",
                placeholder="e.g. payment-service, postgres-db",
                info="Required for: investigate, check_dependencies, restart, rollback",
            )

    reasoning = gr.Textbox(
        label="Reasoning (required — min 10 characters)",
        placeholder="Explain why you are taking this action...",
        lines=2,
    )

    with gr.Accordion("📝 Postmortem Fields (expand for write_postmortem action)", open=False):
        root_cause = gr.Textbox(
            label="Root Cause",
            placeholder="e.g. database_connection_pool_exhausted",
        )
        timeline = gr.Textbox(
            label="Timeline (one entry per line, min 2)",
            lines=3,
            placeholder="T+0s: Alerts fired\nT+60s: Root cause found\nT+120s: Fix applied",
        )
        with gr.Row():
            impact     = gr.Textbox(label="Impact",     placeholder="What was affected?")
            resolution = gr.Textbox(label="Resolution", placeholder="How was it fixed?")
            prevention = gr.Textbox(label="Prevention", placeholder="How to prevent recurrence?")

    step_btn = gr.Button("▶️ Submit Action", variant="primary", size="lg")

    gr.Markdown("### 📜 Last Action Result")
    history_box = gr.Markdown("_No actions taken yet._")

    gr.Markdown("---")

    with gr.Accordion("📖 Scoring Guide & Task Reference", open=False):
        gr.Markdown("""
## Reward Breakdown

| Action | Score |
|---|---|
| Investigate correct service | **+0.15** |
| Check dependencies correctly | **+0.10** |
| Apply correct fix | **+0.20** |
| Correct root cause in postmortem | **+0.20** |
| High quality postmortem | **+0.15** |
| Declare resolved (fix + postmortem) | **+0.25** |
| Wrong restart or rollback | **-0.10** |
| Redundant investigation | **-0.05** |
| Premature resolution / timeout | **-0.10 to -0.15** |

## Task Root Causes (use these exact strings)

| Task | Root Cause String |
|---|---|
| Easy | `database_connection_pool_exhausted` |
| Medium | `auth_service_memory_leak_bad_deployment` |
| Hard | `api_gateway_rate_limiter_config_poisoning` |

## Optimal Strategy
1. `investigate_service` on the most critical service
2. `check_dependencies` to map relationships
3. `investigate_service` on root cause service
4. `restart_service` or `rollback_deployment` on correct service
5. `write_postmortem` with exact root cause string
6. `declare_resolved`
""")

    # Wire up
    reset_btn.click(
        fn=do_reset,
        inputs=[task_picker],
        outputs=[incident_info, alerts_box, status_box,
                 logs_box, score_display, history_box, obs_state],
    )

    step_btn.click(
        fn=do_step,
        inputs=[obs_state, action_type, target_service, reasoning,
                root_cause, timeline, impact, resolution, prevention],
        outputs=[incident_info, alerts_box, status_box,
                 logs_box, score_display, history_box, obs_state],
    )

    baseline_btn.click(
        fn=do_baseline,
        inputs=[],
        outputs=[baseline_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )