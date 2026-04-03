"""Pydantic v2 data models for PagerSim-OpenEnv.
Defines the typed contracts for Observation, Action, Reward, and supporting
models used throughout the incident response simulation environment."""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Alert(BaseModel):
    """Represents a single monitoring alert firing during an incident."""

    service_name: str = Field(..., description="name of the alerting service")
    severity: Literal["critical", "high", "medium", "low"]
    message: str = Field(..., description="alert message text")


class LogEntry(BaseModel):
    """Represents a single log line from a service."""

    timestamp: str = Field(..., description="relative timestamp like T+0s, T+30s")
    service: str = Field(..., description="which service emitted this log")
    level: Literal["ERROR", "WARN", "INFO", "DEBUG"]
    message: str = Field(..., description="log message content")


class Observation(BaseModel):
    """What the agent sees at each step. Returned by reset() and step()."""

    incident_id: str = Field(..., description="unique ID like INC-001")
    task_id: str = Field(..., description="easy, medium, or hard")
    timestamp: str = Field(..., description="current relative time like T+45s")
    alerts: list[Alert] = Field(..., description="currently firing alerts")
    logs: list[LogEntry] = Field(..., description="visible log entries")
    current_status: dict[str, str] = Field(..., description="maps service name to status")
    time_elapsed: int = Field(..., description="seconds since incident started")
    time_limit: int = Field(..., description="max seconds allowed for this task")
    actions_taken: list[str] = Field(..., description="history of actions taken so far")
    hint: Optional[str] = Field(None, description="optional nudge shown after agent is stuck")

    @field_validator("time_elapsed")
    @classmethod
    def time_elapsed_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("time_elapsed must be >= 0")
        return v


class PostMortem(BaseModel):
    """Written by the agent when it thinks it knows the root cause."""

    root_cause: str = Field(..., min_length=10, description="what caused the incident")
    timeline: list[str] = Field(..., description="sequence of events, minimum 2 entries")
    impact: str = Field(..., min_length=10, description="what was affected")
    resolution: str = Field(..., min_length=10, description="how it was fixed")
    prevention: str = Field(..., min_length=10, description="how to prevent recurrence")

    @model_validator(mode="after")
    def validate_timeline(self) -> "PostMortem":
        if len(self.timeline) < 2:
            raise ValueError("timeline must have at least 2 entries")
        return self


class Action(BaseModel):
    """What the agent sends to step(). One action per step."""

    action_type: Literal[
        "investigate_service", "escalate", "restart_service",
        "rollback_deployment", "check_dependencies", "silence_alert",
        "write_postmortem", "declare_resolved"
    ]
    target_service: Optional[str] = Field(None)
    reasoning: str = Field(..., min_length=10, description="agent must explain its action")
    postmortem: Optional[PostMortem] = Field(None)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        if self.action_type in ("restart_service", "rollback_deployment"):
            if self.target_service is None:
                raise ValueError(f"target_service is required for {self.action_type}")
        if self.action_type == "write_postmortem":
            if self.postmortem is None:
                raise ValueError("postmortem is required for write_postmortem")
        return self


class Reward(BaseModel):
    """Returned after every step() call."""

    score: float = Field(...)
    cumulative_score: float = Field(...)
    breakdown: dict[str, float] = Field(..., description="named partial scores")
    feedback: str = Field(..., description="human-readable explanation of the reward")
    is_terminal: bool = Field(False, description="True if episode should end")

    @model_validator(mode="after")
    def clamp_scores(self) -> "Reward":
        self.score = max(-1.0, min(1.0, self.score))
        self.cumulative_score = max(-1.0, min(1.0, self.cumulative_score))
        return self


class TaskInfo(BaseModel):
    """Returned by the /tasks endpoint to describe available tasks."""

    id: str = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    difficulty: Literal["easy", "medium", "hard"]
    time_limit_seconds: int = Field(...)
    max_steps: int = Field(...)
    action_schema: dict[str, str] = Field(..., description="field name to description")


class EpisodeResult(BaseModel):
    """Returned by /grader endpoint after a full episode."""

    task_id: str = Field(...)
    final_score: float = Field(..., ge=0.0, le=1.0)
    steps_taken: int = Field(...)
    time_seconds: float = Field(...)
    actions_summary: list[str] = Field(...)
    postmortem_submitted: bool = Field(...)
    correct_root_cause: bool = Field(...)