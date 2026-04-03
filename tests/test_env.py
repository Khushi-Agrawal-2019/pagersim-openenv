from __future__ import annotations
import pytest

from environment.models import (
    Alert, LogEntry, Observation, PostMortem, Action, Reward, TaskInfo, EpisodeResult
)
from environment.incidents import get_scenario, SCENARIOS
from environment.env import IncidentResponseEnv


# ===================== MODELS TESTS =====================

def test_alert_valid():
    a = Alert(service_name="payment-service", severity="critical", message="Service down")
    assert a.service_name == "payment-service"
    assert a.severity == "critical"


def test_alert_invalid_severity():
    with pytest.raises(Exception):
        Alert(service_name="x", severity="unknown", message="test")


def test_action_requires_target_for_restart():
    with pytest.raises(Exception):
        Action(action_type="restart_service", reasoning="restarting without target")


def test_action_requires_target_for_rollback():
    with pytest.raises(Exception):
        Action(action_type="rollback_deployment", reasoning="rolling back without target")


def test_action_requires_postmortem_for_write_postmortem():
    with pytest.raises(Exception):
        Action(action_type="write_postmortem", reasoning="writing postmortem without content")


def test_action_valid_investigate():
    a = Action(
        action_type="investigate_service",
        target_service="payment-service",
        reasoning="Checking the down service first"
    )
    assert a.action_type == "investigate_service"


def test_postmortem_timeline_too_short():
    with pytest.raises(Exception):
        PostMortem(
            root_cause="database connection pool exhausted completely",
            timeline=["only one entry"],
            impact="all users affected by the outage",
            resolution="restarted the database service",
            prevention="add monitoring and alerts for pool usage"
        )


def test_postmortem_valid():
    pm = PostMortem(
        root_cause="database connection pool exhausted completely",
        timeline=["T+0s: alerts fired", "T+60s: root cause found"],
        impact="payment service down for all users",
        resolution="restarted postgres and increased pool size",
        prevention="add pool monitoring and auto-scaling alerts"
    )
    assert pm.root_cause is not None


def test_reward_score_clamped_high():
    r = Reward(score=99.0, cumulative_score=0.0, breakdown={}, feedback="test", is_terminal=False)
    assert r.score == 1.0


def test_reward_score_clamped_low():
    r = Reward(score=-99.0, cumulative_score=0.0, breakdown={}, feedback="test", is_terminal=False)
    assert r.score == -1.0


def test_observation_rejects_negative_time():
    with pytest.raises(Exception):
        Observation(
            incident_id="INC-001",
            task_id="easy",
            timestamp="T+0s",
            alerts=[],
            logs=[],
            current_status={},
            time_elapsed=-1,
            time_limit=300,
            actions_taken=[]
        )


# ===================== SCENARIO TESTS =====================

def test_all_three_scenarios_exist():
    assert "easy" in SCENARIOS
    assert "medium" in SCENARIOS
    assert "hard" in SCENARIOS


def test_get_scenario_easy():
    s = get_scenario("easy")
    assert s.id == "easy"
    assert len(s.initial_alerts) >= 3
    assert len(s.initial_logs) >= 5
    assert s.correct_root_cause == "database_connection_pool_exhausted"


def test_get_scenario_medium():
    s = get_scenario("medium")
    assert s.id == "medium"
    assert s.correct_root_cause == "auth_service_memory_leak_bad_deployment"


def test_get_scenario_hard():
    s = get_scenario("hard")
    assert s.id == "hard"
    assert s.correct_root_cause == "api_gateway_rate_limiter_config_poisoning"
    assert len(s.initial_service_status) == 5


def test_get_scenario_invalid():
    with pytest.raises(ValueError):
        get_scenario("nonexistent")


def test_scenarios_have_hidden_logs():
    for task_id in ["easy", "medium", "hard"]:
        s = get_scenario(task_id)
        assert len(s.hidden_logs) > 0, f"{task_id} missing hidden logs"


def test_scenarios_have_dependency_map():
    for task_id in ["easy", "medium", "hard"]:
        s = get_scenario(task_id)
        assert len(s.dependency_map) > 0, f"{task_id} missing dependency map"


def test_hard_scenario_has_red_herring():
    s = get_scenario("hard")
    wrong = s.wrong_actions
    assert any("payment-service" in w for w in wrong)


# ===================== ENV TESTS =====================

@pytest.fixture
def env():
    return IncidentResponseEnv()


def test_reset_easy(env):
    obs = env.reset("easy")
    assert obs.incident_id == "INC-EASY-001"
    assert obs.task_id == "easy"
    assert obs.time_elapsed == 0
    assert len(obs.alerts) == 3


def test_reset_medium(env):
    obs = env.reset("medium")
    assert obs.task_id == "medium"
    assert len(obs.alerts) == 5


def test_reset_hard(env):
    obs = env.reset("hard")
    assert obs.task_id == "hard"
    assert len(obs.alerts) == 7


def test_reset_invalid_task(env):
    with pytest.raises(ValueError):
        env.reset("impossible")


def test_step_without_reset_raises(env):
    action = Action(
        action_type="investigate_service",
        target_service="payment-service",
        reasoning="testing without reset first"
    )
    with pytest.raises(RuntimeError):
        env.step(action)


def test_step_correct_investigation_gives_positive_reward(env):
    env.reset("easy")
    action = Action(
        action_type="investigate_service",
        target_service="payment-service",
        reasoning="Investigating the down payment service"
    )
    _, reward, done, _ = env.step(action)
    assert reward.score > 0
    assert not done


def test_step_reveals_hidden_logs(env):
    obs1 = env.reset("easy")
    initial_log_count = len(obs1.logs)

    action = Action(
        action_type="investigate_service",
        target_service="payment-service",
        reasoning="Investigating payment service logs"
    )

    obs2, _, _, _ = env.step(action)
    assert len(obs2.logs) > initial_log_count


def test_step_wrong_restart_gives_negative_reward(env):
    env.reset("easy")
    action = Action(
        action_type="restart_service",
        target_service="api-gateway",
        reasoning="Trying to restart api-gateway to fix the issue"
    )
    _, reward, _, _ = env.step(action)
    assert reward.score < 0


def test_state_reflects_steps(env):
    env.reset("easy")
    env.step(Action(
        action_type="investigate_service",
        target_service="payment-service",
        reasoning="Starting investigation of payment service"
    ))

    state = env.state()
    assert state["steps_taken"] == 1
    assert state["episode_active"] is True


def test_premature_resolve_penalized(env):
    env.reset("easy")
    action = Action(
        action_type="declare_resolved",
        reasoning="declaring resolved without doing anything at all"
    )
    _, reward, done, _ = env.step(action)
    assert reward.score < 0
    assert done is True


def test_correct_fix_marks_episode(env):
    env.reset("easy")

    env.step(Action(
        action_type="investigate_service",
        target_service="payment-service",
        reasoning="investigating payment service logs first"
    ))

    env.step(Action(
        action_type="restart_service",
        target_service="postgres-db",
        reasoning="restarting postgres to fix connection pool"
    ))

    assert env.correct_fix_applied is True


def test_reset_clears_previous_state(env):
    env.reset("easy")

    env.step(Action(
        action_type="investigate_service",
        target_service="payment-service",
        reasoning="investigating payment service"
    ))

    env.reset("medium")

    state = env.state()
    assert state["steps_taken"] == 0
    assert state["task_id"] == "medium"
    assert state["cumulative_score"] == 0.0


def test_full_easy_episode_scores_well(env):
    env.reset("easy")

    actions = [
        Action(action_type="investigate_service", target_service="payment-service",
               reasoning="Investigate the critically down payment service first"),

        Action(action_type="investigate_service", target_service="postgres-db",
               reasoning="Check the database that payment service depends on"),

        Action(action_type="check_dependencies", target_service="payment-service",
               reasoning="Check dependencies"),

        Action(action_type="restart_service", target_service="postgres-db",
               reasoning="Restart postgres to fix issue"),

        Action(action_type="write_postmortem",
               reasoning="Document root cause",
               postmortem=PostMortem(
                   root_cause="database_connection_pool_exhausted",
                   timeline=[
                       "T+0s: alerts fired",
                       "T+30s: logs investigated",
                       "T+120s: fix applied"
                   ],
                   impact="Payment service down",
                   resolution="Restarted postgres",
                   prevention="Add monitoring"
               )),

        Action(action_type="declare_resolved",
               reasoning="Incident resolved")
    ]

    for action in actions:
        _, reward, done, _ = env.step(action)
        if done:
            break

    assert env.cumulative_score >= 0.5