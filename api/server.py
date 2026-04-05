"""FastAPI server for PagerSim-OpenEnv.
Exposes the OpenEnv spec endpoints (reset, step, state) plus
hackathon-required endpoints (tasks, grader, baseline, health).
Runs on port 7860 for Hugging Face Spaces compatibility."""

from __future__ import annotations
import time
import os
from contextlib import asynccontextmanager
from typing import Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from environment.env import IncidentResponseEnv
from environment.models import Action, TaskInfo, EpisodeResult
from environment.incidents import get_scenario, SCENARIOS


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env, server_start_time
    try:
        env = IncidentResponseEnv()
        server_start_time = time.monotonic()
        print("✅ IncidentResponseEnv initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize env: {e}")
        raise
    yield


app = FastAPI(
    title="PagerSim-OpenEnv",
    description="SRE Incident Response simulation environment for AI agent training",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env: IncidentResponseEnv = None
server_start_time: float = 0.0


class ResetRequest(BaseModel):
    task_id: str


class GraderRequest(BaseModel):
    task_id: str
    episode_actions: list[dict[str, Any]]


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "uptime_seconds": time.monotonic() - server_start_time}


@app.get("/")
async def root():
    return {
        "name": "PagerSim-OpenEnv",
        "description": "SRE Incident Response OpenEnv environment",
        "version": "1.0.0",
        "endpoints": [{"path": route.path, "method": route.methods} for route in app.routes],
        "tasks": list(SCENARIOS.keys()),
        "docs": "/docs",
    }


@app.get("/tasks", response_model=list[TaskInfo])
async def tasks():
    task_list = []
    for task_id, scenario in SCENARIOS.items():
        task_info = TaskInfo(
            id=scenario.id,
            name=scenario.name,
            description=scenario.description,
            difficulty=scenario.difficulty,
            time_limit_seconds=scenario.time_limit_seconds,
            max_steps=scenario.max_steps,
            action_schema={
                "action_type": "One of: investigate_service, escalate, restart_service, rollback_deployment, check_dependencies, silence_alert, write_postmortem, declare_resolved",
                "target_service": "Required for: restart_service, rollback_deployment, investigate_service, check_dependencies, silence_alert. The name of the service to act on.",
                "reasoning": "Required. Min 10 chars. Explain why you are taking this action.",
                "postmortem": "Required only for write_postmortem. Object with fields: root_cause, timeline (list), impact, resolution, prevention.",
            },
        )
        task_list.append(task_info)
    return task_list


@app.post("/reset")
async def reset(request: ResetRequest):
    if env is None:
        raise HTTPException(500, detail="Environment not initialized. Server still starting up.")
    try:
        obs = env.reset(request.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(action: Action):
    if env is None:
        raise HTTPException(500, detail="Environment not initialized.")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


@app.get("/state")
async def state():
    if env is None:
        raise HTTPException(400, detail="Environment not initialized.")
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(400, detail=f"State failed: {str(e)}")


@app.post("/grader")
async def grader(request: GraderRequest):
    temp_env = IncidentResponseEnv()
    try:
        temp_env.reset(request.task_id)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    for action_dict in request.episode_actions:
        try:
            action = Action(**action_dict)
            obs, reward, done, info = temp_env.step(action)
            if done:
                break
        except Exception as e:
            continue

    final_state = temp_env.state()

    episode_result = EpisodeResult(
        task_id=request.task_id,
        final_score=max(0.0, min(1.0, final_state["cumulative_score"])),
        steps_taken=final_state["steps_taken"],
        time_seconds=float(final_state["time_elapsed"]),
        actions_summary=final_state["actions_taken"],
        postmortem_submitted=final_state["postmortem_submitted"],
        correct_root_cause=final_state["correct_fix_applied"],
    )

    return episode_result.model_dump()


@app.post("/baseline")
async def baseline():
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        temp_env = IncidentResponseEnv()
        obs = temp_env.reset(task_id)
        scenario = get_scenario(task_id)

        for action_str in scenario.optimal_action_sequence:
            parts = action_str.split(":")
            action_type = parts[0]
            target_service = parts[1] if len(parts) > 1 else None

            if action_type == "write_postmortem":
                from environment.models import PostMortem

                pm = PostMortem(
                    root_cause=scenario.correct_root_cause.replace("_", " "),
                    timeline=[
                        "T+0s: Incident detected via alerts",
                        "T+60s: Root cause identified through log investigation",
                        "T+120s: Fix applied and services recovering",
                    ],
                    impact="Production services degraded affecting all users",
                    resolution="Applied correct fix by following investigation trail in logs",
                    prevention="Add monitoring alerts and runbooks for this failure mode",
                )
                action = Action(
                    action_type="write_postmortem",
                    reasoning="Writing postmortem after identifying root cause from logs",
                    postmortem=pm,
                )
            else:
                action = Action(
                    action_type=action_type,
                    target_service=target_service,
                    reasoning=f"Following optimal investigation sequence: {action_str}",
                )

            obs, reward, done, info = temp_env.step(action)
            if done:
                break

        final_state = temp_env.state()
        scores[task_id] = max(0.0, min(1.0, final_state["cumulative_score"]))

    average_score = sum(scores.values()) / len(scores)

    return {
        "scores": scores,
        "average": average_score,
        "note": "Deterministic rule-based agent following optimal action sequences",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)