# Project: PagerSim-OpenEnv
# Hackathon: Meta PyTorch OpenEnv Hackathon 2026
# Reference: PagerDuty-style SRE incident response simulation

## What This Project Is
A production-grade OpenEnv-compliant AI training environment simulating 
SRE (Site Reliability Engineer) incident response. AI agents learn to 
diagnose and resolve production outages through a step()/reset()/state() API.

## Tech Stack
- Python 3.11
- FastAPI (web server, port 7860)
- Pydantic v2 (all data models)
- Uvicorn (ASGI server)
- Pytest (testing)
- Docker (containerization)
- Hugging Face Spaces (deployment)

## Project Structure
pagersim-openenv/
├── environment/
│   ├── __init__.py
│   ├── env.py          # Main IncidentResponseEnv class
│   ├── models.py       # Pydantic models
│   ├── incidents.py    # Scenario definitions
│   └── grader.py       # Task graders
├── api/
│   ├── __init__.py
│   └── server.py       # FastAPI server
├── baseline/
│   └── run_baseline.py # OpenAI baseline script
├── tests/
│   └── test_env.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md

## Coding Rules (ALWAYS follow these)
- All Python functions must have type hints
- All classes and modules must have docstrings
- No hardcoded secrets — use environment variables only
- No eval() or exec() anywhere in the codebase
- Pydantic v2 syntax only (not v1)
- Functions must be under 50 lines — split if longer
- No global mutable state except single env instance in server.py
- Use dataclasses or Pydantic models, never raw dicts for structured data

## Security Rules (NEVER violate these)
- Validate ALL inputs before processing
- API endpoints must return generic error messages (no stack traces)
- Docker must run as non-root user
- No secrets in any file that gets committed to git
- All string inputs must be stripped of whitespace before use

## Testing Rules
- Every new function needs a test
- Run after every change: python -m pytest tests/ -v
- Minimum 80% code coverage required
- Tests must be deterministic (no random, no time.sleep)

## OpenEnv Spec Requirements (hackathon critical)
- reset(task_id) → returns Observation
- step(action) → returns (Observation, Reward, done, info)  
- state() → returns dict
- All 3 endpoints must be exposed via FastAPI
- Additional endpoints: /tasks, /grader, /baseline
- openenv.yaml must exist with correct schema
- Grader scores must be between 0.0 and 1.0
- Port must be 7860 for Hugging Face Spaces

## The 3 Tasks
1. easy   → "database_overload" — single service down, clear root cause
2. medium → "cascading_auth_failure" — 3 services, misleading signals  
3. hard   → "rate_limiter_config_poisoning" — 5 services, red herring deployment

## Reward Function
+0.15 investigating the correct root cause service
+0.10 per correct action in optimal sequence
+0.20 correct root_cause string in postmortem
+0.25 declare_resolved after correct fix
+0.15 postmortem quality (all 5 fields > 20 chars)
-0.05 redundant investigation (same service 3+ times)
-0.10 restarting/rolling back wrong service
-0.15 timeout penalty
Score always clamped to [0.0, 1.0]