---
title: PagerSim-OpenEnv
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: Docker
pinned: false
---

# PagerSim-OpenEnv

**SRE Incident Response Simulation for AI Agent Training**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://huggingface.co/spaces/Kurru/pagersim-openenv)
[![Python](https://img.shields.io/badge/python-3.11-green)](https://python.org)

## Overview

PagerSim-OpenEnv simulates a production SRE (Site Reliability Engineer) on-call workflow. An AI agent receives monitoring alerts and logs, investigates services, identifies root causes, applies fixes, and writes postmortems — exactly like a real engineer responding to a PagerDuty incident.

This environment was built for the Meta PyTorch OpenEnv Hackathon to fill a real gap: there are no existing OpenEnv environments for training agents on operational/DevOps tasks.

## Environment Description

The agent plays the role of an on-call SRE during a production incident. At each step it:
- Reads alerts and logs from monitoring systems
- Takes investigative or corrective actions
- Receives shaped rewards for correct behavior
- Must identify root cause and write a postmortem to fully resolve

## Observation Space
```json
{
  "incident_id": "INC-EASY-001",
  "task_id": "easy",
  "timestamp": "T+45s",
  "alerts": [{"service_name": "payment-service", "severity": "critical", "message": "..."}],
  "logs": [{"timestamp": "T+0s", "service": "payment-service", "level": "ERROR", "message": "..."}],
  "current_status": {"payment-service": "down", "postgres-db": "degraded"},
  "time_elapsed": 45,
  "time_limit": 300,
  "actions_taken": ["investigate_service:payment-service"],
  "hint": null
}
```

## Action Space

| Action | Target Required | Description |
|---|---|---|
| `investigate_service` | Yes | Reveal hidden logs for a service |
| `check_dependencies` | Yes | Reveal service dependency map |
| `restart_service` | Yes | Restart a specific service |
| `rollback_deployment` | Yes | Revert a deployment |
| `escalate` | No | Page another team |
| `silence_alert` | Yes | Mute a noisy alert |
| `write_postmortem` | No | Submit root cause analysis |
| `declare_resolved` | No | End the episode |
```json
{
  "action_type": "investigate_service",
  "target_service": "payment-service",
  "reasoning": "Starting with the critically down service",
  "postmortem": null
}
```

## Reward Function

| Event | Reward |
|---|---|
| Investigate correct service | +0.15 |
| Check dependencies correctly | +0.10 |
| Apply correct fix | +0.20 |
| Correct root cause in postmortem | +0.20 |
| Postmortem quality | +0.15 |
| Declare resolved (with fix + postmortem) | +0.25 |
| Wrong restart/rollback | -0.10 |
| Redundant investigation (3+ times) | -0.05 |
| Timeout | -0.15 |
| Premature resolution | -0.10 |

**Max score: 1.0** (clamped)

## Tasks

### Easy — Database Overload
- **Services:** payment-service, postgres-db
- **Root cause:** Connection pool exhausted
- **Time limit:** 300s / 10 steps
- **Expected difficulty:** GPT-4o-mini solves this reliably

### Medium — Cascading Auth Failure
- **Services:** api-gateway, auth-service, user-service
- **Root cause:** Memory leak in auth-service v2.3.1 deployment
- **Complication:** api-gateway shows errors but is a victim, not the cause
- **Time limit:** 600s / 15 steps

### Hard — Rate Limiter Config Poisoning
- **Services:** frontend, api-gateway, payment-service, notification-service, cdn
- **Root cause:** api-gateway rate_limiter.yaml accidentally set to 10 req/min
- **Red herring:** payment-service had a deployment 30min ago — not the cause
- **Time limit:** 900s / 20 steps

## Baseline Scores

Rule-based agent following optimal action sequences:

| Task | Score | Steps |
|---|---|---|
| easy | 0.85 | 6 |
| medium | 0.75 | 7 |
| hard | 0.65 | 7 |
| **Average** | **0.75** | |

## Setup
```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/pagersim-openenv
cd pagersim-openenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn api.server:app --host 0.0.0.0 --port 7860
```

## Usage Example
```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_id": "easy"}).json()
print(obs["alerts"])

# Take action
result = requests.post(f"{BASE}/step", json={
    "action_type": "investigate_service",
    "target_service": "payment-service",
    "reasoning": "Starting with the critically down service"
}).json()

print(result["reward"]["score"])
print(result["reward"]["feedback"])
```

## Docker
```bash
docker build -t pagersim-openenv .
docker run -p 7860:7860 pagersim-openenv
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks + action schema |
| POST | `/reset` | Start new episode `{"task_id": "easy"}` |
| POST | `/step` | Submit action |
| GET | `/state` | Current episode state |
| POST | `/grader` | Score a full episode |
| POST | `/baseline` | Run rule-based baseline agent |
| GET | `/docs` | Interactive API docs (Swagger) |