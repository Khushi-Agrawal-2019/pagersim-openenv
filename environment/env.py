"""Core IncidentResponseEnv class for PagerSim-OpenEnv.
Implements the OpenEnv spec: reset(), step(), state().
Simulates SRE incident response with shaped rewards and
partial progress signals across 3 difficulty levels."""

from __future__ import annotations
import time
import copy
from typing import Any
from environment.models import (
    Alert, LogEntry, Observation, Action, Reward, PostMortem
)
from environment.incidents import get_scenario, IncidentScenario


class IncidentResponseEnv:
    """OpenEnv-compliant incident response simulation environment.
    An AI agent plays the role of an on-call SRE, receiving alerts and logs,
    taking actions to diagnose and resolve production incidents, and being
    scored on accuracy, efficiency, and quality of root cause analysis."""

    MAX_REDUNDANT_INVESTIGATIONS = 2
    CORRECT_ROOT_CAUSE_THRESHOLD = 0.85

    def __init__(self):
        self.current_scenario: IncidentScenario | None = None
        self.current_service_status: dict[str, str] = {}
        self.visible_logs: list[LogEntry] = []
        self.actions_taken: list[str] = []
        self.investigation_counts: dict[str, int] = {}
        self.episode_start_time: float = 0.0
        self.time_elapsed: int = 0
        self.cumulative_score: float = 0.001
        self.episode_active: bool = False
        self.correct_fix_applied: bool = False
        self.postmortem_submitted: bool = False
        self.incident_id: str = ""

    def reset(self, task_id: str) -> Observation:
        """Start a new episode for the given task. Returns initial observation."""
        scenario = get_scenario(task_id)
        self.current_scenario = copy.deepcopy(scenario)
        self.current_service_status = copy.deepcopy(scenario.initial_service_status)
        self.visible_logs = copy.deepcopy(scenario.initial_logs)
        self.actions_taken = []
        self.investigation_counts = {}
        self.episode_start_time = time.monotonic()
        self.time_elapsed = 0
        self.cumulative_score = 0.001
        self.episode_active = True
        self.correct_fix_applied = False
        self.postmortem_submitted = False
        self.incident_id = f"INC-{task_id.upper()}-001"

        return Observation(
            incident_id=self.incident_id,
            task_id=task_id,
            timestamp="T+0s",
            alerts=copy.deepcopy(scenario.initial_alerts),
            logs=self.visible_logs,
            current_status=self.current_service_status,
            time_elapsed=0,
            time_limit=scenario.time_limit_seconds,
            actions_taken=[],
            hint=None,
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Process one agent action. Returns (observation, reward, done, info)."""
        if not self.episode_active:
            raise RuntimeError("No active episode. Call reset() first.")

        elapsed_float = time.monotonic() - self.episode_start_time
        self.time_elapsed = int(elapsed_float)

        action_str = f"{action.action_type}"
        if action.target_service:
            action_str += f":{action.target_service}"
        self.actions_taken.append(action_str)

        if self.time_elapsed >= self.current_scenario.time_limit_seconds:
            reward = self._make_reward(-0.15, "Time limit exceeded", {"timeout": -0.15})
            self.episode_active = False
            return self._make_observation(), reward, True, {"reason": "timeout"}

        action_reward_score, reward_breakdown, reward_feedback = self._apply_action(action)

        new_cum = max(-0.999, min(0.999, self.cumulative_score + action_reward_score))
        self.cumulative_score = new_cum if new_cum != 0.0 else 0.001

        done = False
        reason = "continue"

        if action.action_type == "declare_resolved":
            done = True
            reason = "agent_declared_resolved"
            self.episode_active = False

        if len(self.actions_taken) >= self.current_scenario.max_steps:
            done = True
            reason = "max_steps_reached"
            self.episode_active = False

        hint = None
        if len(self.actions_taken) >= 5 and not self.correct_fix_applied:
            hint = "Tip: Focus on the service with the most recent deployment or config change."

        reward = self._make_reward(action_reward_score, reward_feedback, reward_breakdown)
        reward.cumulative_score = self.cumulative_score
        reward.is_terminal = done

        return self._make_observation(hint=hint), reward, done, {"reason": reason, "steps": len(self.actions_taken)}

    def state(self) -> dict[str, Any]:
        """Return the full current environment state as a serializable dict."""
        return {
            "episode_active": self.episode_active,
            "incident_id": self.incident_id,
            "task_id": self.current_scenario.id if self.current_scenario else None,
            "time_elapsed": self.time_elapsed,
            "time_limit": self.current_scenario.time_limit_seconds if self.current_scenario else None,
            "cumulative_score": self.cumulative_score,
            "actions_taken": self.actions_taken,
            "current_service_status": self.current_service_status,
            "correct_fix_applied": self.correct_fix_applied,
            "postmortem_submitted": self.postmortem_submitted,
            "steps_taken": len(self.actions_taken),
            "steps_remaining": (self.current_scenario.max_steps - len(self.actions_taken)) if self.current_scenario else 0,
        }

    def _apply_action(self, action: Action) -> tuple[float, dict[str, float], str]:
        """Apply action to environment state. Returns (score_delta, breakdown, feedback)."""
        if action.action_type == "investigate_service":
            target = action.target_service
            if target not in self.current_scenario.hidden_logs:
                return 0.0, {}, f"No additional logs found for {target}"

            self.investigation_counts[target] = self.investigation_counts.get(target, 0) + 1
            count = self.investigation_counts[target]

            if count > self.MAX_REDUNDANT_INVESTIGATIONS:
                return -0.05, {"redundant_investigation": -0.05}, f"Redundant investigation of {target} — no new information"

            self.visible_logs.extend(self.current_scenario.hidden_logs[target])

            correct_services = [
                a.split(":")[1] for a in self.current_scenario.optimal_action_sequence
                if a.startswith("investigate_service:")
            ]

            if target in correct_services:
                return 0.15, {"correct_investigation": 0.15}, f"Good investigation of {target} — new logs revealed {len(self.current_scenario.hidden_logs[target])} entries"
            else:
                return 0.0, {"investigation": 0.0}, f"Investigated {target} — no significant findings"

        elif action.action_type == "check_dependencies":
            target = action.target_service
            if target not in self.current_scenario.dependency_map:
                return 0.0, {}, f"No dependency info available for {target}"

            deps = self.current_scenario.dependency_map[target]

            if target in [a.split(":")[1] for a in self.current_scenario.optimal_action_sequence if "check_dependencies" in a]:
                return 0.10, {"correct_dependency_check": 0.10}, f"Dependencies for {target}: {deps}"
            else:
                return 0.0, {}, f"Dependencies for {target}: {deps}"

        elif action.action_type == "restart_service":
            target = action.target_service
            action_str = f"restart_service:{target}"

            if action_str in self.current_scenario.wrong_actions:
                return -0.10, {"wrong_restart": -0.10}, f"Restarting {target} had no effect — this is not the root cause service"

            if action_str in self.current_scenario.optimal_action_sequence:
                self.current_service_status[target] = "recovering"
                self.correct_fix_applied = True
                return 0.20, {"correct_fix": 0.20}, f"Restarting {target} is working — service is recovering"

            return 0.0, {}, f"Restarted {target} — status unchanged"

        elif action.action_type == "rollback_deployment":
            target = action.target_service
            action_str = f"rollback_deployment:{target}"

            if action_str in self.current_scenario.wrong_actions:
                return -0.10, {"wrong_rollback": -0.10}, f"Rolling back {target} had no effect — this was not the problem deployment"

            if action_str in self.current_scenario.optimal_action_sequence:
                self.current_service_status[target] = "recovering"
                self.correct_fix_applied = True
                return 0.20, {"correct_fix": 0.20}, f"Rolling back {target} is working — services are recovering"

            return 0.0, {}, f"Rolled back {target} — no clear effect"

        elif action.action_type == "write_postmortem":
            self.postmortem_submitted = True
            pm = action.postmortem
            score = 0.0
            breakdown = {}

            submitted = pm.root_cause.lower().strip().replace(" ", "_")
            correct = self.current_scenario.correct_root_cause.lower()

            if submitted == correct or correct in submitted:
                score += 0.20
                breakdown["correct_root_cause"] = 0.20
                feedback_rc = "Root cause correctly identified!"
            else:
                feedback_rc = f"Root cause incorrect. Submitted: '{pm.root_cause}'"

            quality_score = 0.0
            fields = [pm.root_cause, pm.impact, pm.resolution, pm.prevention]
            meaningful = sum(1 for f in fields if len(f) > 30)
            timeline_ok = len(pm.timeline) >= 3

            if meaningful >= 3 and timeline_ok:
                quality_score = 0.15
                breakdown["postmortem_quality"] = 0.15
            elif meaningful >= 2:
                quality_score = 0.07
                breakdown["postmortem_quality"] = 0.07

            score += quality_score

            return score, breakdown, f"{feedback_rc} Postmortem quality score: {quality_score:.2f}"

        elif action.action_type == "declare_resolved":
            if not self.correct_fix_applied:
                return -0.10, {"premature_resolution": -0.10}, "Declared resolved without applying a fix — incident not actually resolved"

            if not self.postmortem_submitted:
                return 0.15, {"resolved_no_postmortem": 0.15}, "Incident resolved! Missing postmortem — submit one for full score"

            return 0.25, {"incident_resolved": 0.25}, "Excellent! Incident resolved with postmortem submitted"

        elif action.action_type == "escalate":
            return 0.0, {}, "Escalation noted — continuing investigation"

        elif action.action_type == "silence_alert":
            if action.target_service and f"silence_alert:{action.target_service}" in self.current_scenario.wrong_actions:
                return -0.05, {"wrong_silence": -0.05}, f"Silencing {action.target_service} alert hides important signal"
            return 0.0, {}, "Alert silenced"

        else:
            return 0.0, {}, f"Unknown action type: {action.action_type}"

    def _make_observation(self, hint=None) -> Observation:
        """Build current Observation from environment state."""
        timestamp = f"T+{self.time_elapsed}s"
        return Observation(
            incident_id=self.incident_id,
            task_id=self.current_scenario.id,
            timestamp=timestamp,
            alerts=copy.deepcopy(self.current_scenario.initial_alerts),
            logs=self.visible_logs,
            current_status=self.current_service_status,
            time_elapsed=self.time_elapsed,
            time_limit=self.current_scenario.time_limit_seconds,
            actions_taken=self.actions_taken,
            hint=hint,
        )

    def _make_reward(self, score, feedback, breakdown) -> Reward:
        """Build a Reward object with clamped scores."""
        # Clamp and ensure never exactly 0.0
        clamped = max(-0.999, min(0.999, score))
        if clamped == 0.0:
            clamped = 0.001
        cum = max(-0.999, min(0.999, self.cumulative_score))
        if cum == 0.0:
            cum = 0.001
        return Reward(
            score=clamped,
            cumulative_score=cum,
            breakdown=breakdown,
            feedback=feedback,
            is_terminal=False,
        )