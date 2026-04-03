"""Incident scenario definitions for PagerSim-OpenEnv.
Each scenario simulates a real-world SRE production incident with
alerts, logs, service states, and grading criteria."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from environment.models import Alert, LogEntry


@dataclass
class IncidentScenario:
    """A complete definition of one incident task."""
    id: str
    name: str
    description: str
    difficulty: str
    time_limit_seconds: int
    max_steps: int
    initial_alerts: list[Alert]
    initial_logs: list[LogEntry]
    initial_service_status: dict[str, str]
    correct_root_cause: str
    optimal_action_sequence: list[str]
    wrong_actions: list[str]
    services_metadata: dict[str, dict[str, Any]]
    hidden_logs: dict[str, list[LogEntry]]
    dependency_map: dict[str, list[str]]


easy_scenario = IncidentScenario(
    id="easy",
    name="Database Overload",
    description="The payment service is completely down. Customers cannot complete purchases. On-call SRE must identify and resolve the root cause quickly.",
    difficulty="easy",
    time_limit_seconds=300,
    max_steps=10,
    initial_alerts=[
        Alert(service_name="payment-service", severity="critical", message="payment-service is DOWN - health check failing for 3 minutes"),
        Alert(service_name="postgres-db", severity="high", message="postgres-db connection pool at 98% capacity (490/500 connections)"),
        Alert(service_name="payment-service", severity="high", message="payment-service error rate 100% - all requests failing"),
    ],
    initial_logs=[
        LogEntry(timestamp="T+0s", service="payment-service", level="ERROR", message="Failed to acquire database connection: connection pool exhausted"),
        LogEntry(timestamp="T+15s", service="payment-service", level="ERROR", message="Health check failed: cannot connect to postgres-db"),
        LogEntry(timestamp="T+30s", service="payment-service", level="ERROR", message="Retrying database connection... attempt 47 of 50"),
        LogEntry(timestamp="T+45s", service="postgres-db", level="WARN", message="Connection pool utilization critical: 490/500 connections active"),
        LogEntry(timestamp="T+60s", service="postgres-db", level="ERROR", message="Refusing new connections: pool limit reached"),
    ],
    initial_service_status={
        "payment-service": "down",
        "postgres-db": "degraded",
        "api-gateway": "up",
    },
    correct_root_cause="database_connection_pool_exhausted",
    optimal_action_sequence=[
        "investigate_service:payment-service",
        "investigate_service:postgres-db",
        "check_dependencies:payment-service",
        "restart_service:postgres-db",
        "write_postmortem",
        "declare_resolved",
    ],
    wrong_actions=[
        "restart_service:api-gateway",
        "rollback_deployment:payment-service",
        "silence_alert:payment-service",
    ],
    services_metadata={
        "payment-service": {"version": "v2.1.4", "last_deploy": "3 days ago", "owner": "payments-team"},
        "postgres-db": {"version": "14.2", "last_deploy": "2 weeks ago", "owner": "infra-team"},
        "api-gateway": {"version": "v1.8.0", "last_deploy": "1 week ago", "owner": "platform-team"},
    },
    hidden_logs={
        "payment-service": [
            LogEntry(timestamp="T+75s", service="payment-service", level="ERROR", message="pool exhausted after traffic spike"),
            LogEntry(timestamp="T+90s", service="payment-service", level="ERROR", message="pool exhausted after traffic spike"),
            LogEntry(timestamp="T+105s", service="payment-service", level="ERROR", message="pool exhausted after traffic spike"),
            LogEntry(timestamp="T+120s", service="payment-service", level="ERROR", message="pool exhausted after traffic spike"),
        ],
        "postgres-db": [
            LogEntry(timestamp="T+75s", service="postgres-db", level="WARN", message="max_connections=500 reached"),
            LogEntry(timestamp="T+90s", service="postgres-db", level="WARN", message="max_connections=500 reached"),
            LogEntry(timestamp="T+105s", service="postgres-db", level="WARN", message="max_connections=500 reached"),
            LogEntry(timestamp="T+120s", service="postgres-db", level="WARN", message="max_connections=500 reached"),
        ],
    },
    dependency_map={
        "payment-service": ["postgres-db", "api-gateway"],
        "postgres-db": [],
        "api-gateway": ["payment-service"],
    },
)

medium_scenario = IncidentScenario(
    id="medium",
    name="Cascading Auth Failure",
    description="Multiple services are degraded. The API gateway is throwing timeouts. Auth is failing for all users. A bad deployment 2 hours ago may be involved — but which service? The SRE must trace the cascade back to the true root cause.",
    difficulty="medium",
    time_limit_seconds=600,
    max_steps=15,
    initial_alerts=[
        Alert(service_name="api-gateway", severity="high", message="api-gateway latency P99 > 30s - severe degradation"),
        Alert(service_name="auth-service", severity="critical", message="auth-service DOWN - all authentication requests failing"),
        Alert(service_name="user-service", severity="high", message="user-service degraded - 60% of requests failing"),
        Alert(service_name="auth-service", severity="critical", message="auth-service memory usage at 98% - OOMKill imminent"),
        Alert(service_name="api-gateway", severity="high", message="api-gateway upstream auth failures: 2847 errors in last 5 min"),
    ],
    initial_logs=[
        LogEntry(timestamp="T+0s", service="api-gateway", level="ERROR", message="Upstream timeout: auth-service not responding after 30s"),
        LogEntry(timestamp="T+10s", service="api-gateway", level="ERROR", message="Circuit breaker OPEN for auth-service - all requests failing fast"),
        LogEntry(timestamp="T+20s", service="auth-service", level="ERROR", message="OutOfMemoryError: Java heap space - service degrading"),
        LogEntry(timestamp="T+30s", service="user-service", level="ERROR", message="Cannot validate user session: auth-service unavailable"),
        LogEntry(timestamp="T+40s", service="api-gateway", level="WARN", message="Returning 503 to clients: no healthy auth upstream available"),
        LogEntry(timestamp="T+50s", service="auth-service", level="ERROR", message="GC overhead limit exceeded - memory leak suspected"),
    ],
    initial_service_status={
        "api-gateway": "degraded",
        "auth-service": "down",
        "user-service": "degraded",
    },
    correct_root_cause="auth_service_memory_leak_bad_deployment",
    optimal_action_sequence=[
        "investigate_service:api-gateway",
        "investigate_service:auth-service",
        "check_dependencies:api-gateway",
        "investigate_service:user-service",
        "rollback_deployment:auth-service",
        "write_postmortem",
        "declare_resolved",
    ],
    wrong_actions=[
        "restart_service:api-gateway",
        "rollback_deployment:api-gateway",
        "rollback_deployment:user-service",
        "restart_service:user-service",
    ],
    services_metadata={
        "api-gateway": {"version": "v1.8.0", "last_deploy": "5 days ago", "owner": "platform-team"},
        "auth-service": {"version": "v2.3.1", "last_deploy": "2 hours ago", "owner": "auth-team", "deploy_note": "Added token refresh caching - performance improvement"},
        "user-service": {"version": "v3.1.0", "last_deploy": "1 week ago", "owner": "user-team"},
    },
    hidden_logs={
        "auth-service": [
            LogEntry(timestamp="T+60s", service="auth-service", level="ERROR", message="memory leak in token cache introduced in v2.3.1"),
            LogEntry(timestamp="T+70s", service="auth-service", level="ERROR", message="memory leak in token cache introduced in v2.3.1"),
            LogEntry(timestamp="T+80s", service="auth-service", level="ERROR", message="memory leak in token cache introduced in v2.3.1"),
            LogEntry(timestamp="T+90s", service="auth-service", level="ERROR", message="memory leak in token cache introduced in v2.3.1"),
            LogEntry(timestamp="T+100s", service="auth-service", level="ERROR", message="memory leak in token cache introduced in v2.3.1"),
        ],
        "api-gateway": [
            LogEntry(timestamp="T+60s", service="api-gateway", level="INFO", message="Healthy before auth went down"),
            LogEntry(timestamp="T+70s", service="api-gateway", level="INFO", message="Healthy before auth went down"),
            LogEntry(timestamp="T+80s", service="api-gateway", level="INFO", message="Healthy before auth went down"),
        ],
        "user-service": [
            LogEntry(timestamp="T+60s", service="user-service", level="ERROR", message="Started failing AFTER auth went down"),
            LogEntry(timestamp="T+70s", service="user-service", level="ERROR", message="Started failing AFTER auth went down"),
            LogEntry(timestamp="T+80s", service="user-service", level="ERROR", message="Started failing AFTER auth went down"),
        ],
    },
    dependency_map={
        "api-gateway": ["auth-service", "user-service"],
        "auth-service": [],
        "user-service": ["auth-service"],
    },
)

hard_scenario = IncidentScenario(
    id="hard",
    name="Rate Limiter Config Poisoning",
    description="All five services are degraded. Customers are getting throttled across every product surface. A payment service deployment happened 30 minutes ago - the obvious suspect. But the real cause is hidden deeper. The SRE must resist the red herring and find the true root cause.",
    difficulty="hard",
    time_limit_seconds=900,
    max_steps=20,
    initial_alerts=[
        Alert(service_name="frontend", severity="high", message="frontend - user requests being throttled: HTTP 429 rate limit errors"),
        Alert(service_name="api-gateway", severity="critical", message="api-gateway rate limiter blocking 78% of requests - abnormal"),
        Alert(service_name="payment-service", severity="high", message="payment-service error rate 45% - elevated failures post-deploy"),
        Alert(service_name="notification-service", severity="medium", message="notification-service - email delivery failures, rate limit hit"),
        Alert(service_name="cdn", severity="medium", message="cdn - origin request throttling detected, serving stale cache"),
        Alert(service_name="api-gateway", severity="critical", message="api-gateway config last modified 45 minutes ago - rate_limiter.yaml changed"),
        Alert(service_name="payment-service", severity="high", message="payment-service deployed v3.2.1 - 30 minutes ago by payments-team"),
    ],
    initial_logs=[
        LogEntry(timestamp="T+0s", service="payment-service", level="ERROR", message="Deployment v3.2.1 complete - elevated error rate detected post-deploy"),
        LogEntry(timestamp="T+5s", service="api-gateway", level="ERROR", message="Rate limit exceeded for client 10.0.0.0/8: 429 Too Many Requests"),
        LogEntry(timestamp="T+10s", service="frontend", level="WARN", message="API calls being throttled - backing off and retrying"),
        LogEntry(timestamp="T+15s", service="payment-service", level="ERROR", message="Payment processing failed: upstream API rate limit hit"),
        LogEntry(timestamp="T+20s", service="notification-service", level="WARN", message="Email delivery throttled: API gateway returning 429"),
        LogEntry(timestamp="T+25s", service="cdn", level="WARN", message="Origin throttled: falling back to cached content from 2 hours ago"),
        LogEntry(timestamp="T+30s", service="api-gateway", level="ERROR", message="rate_limiter.yaml loaded: global_rate_limit set to 10 req/min (was 10000)"),
    ],
    initial_service_status={
        "frontend": "degraded",
        "api-gateway": "degraded",
        "payment-service": "degraded",
        "notification-service": "degraded",
        "cdn": "degraded",
    },
    correct_root_cause="api_gateway_rate_limiter_config_poisoning",
    optimal_action_sequence=[
        "investigate_service:api-gateway",
        "check_dependencies:api-gateway",
        "investigate_service:payment-service",
        "investigate_service:frontend",
        "rollback_deployment:api-gateway",
        "write_postmortem",
        "declare_resolved",
    ],
    wrong_actions=[
        "rollback_deployment:payment-service",
        "restart_service:payment-service",
        "rollback_deployment:frontend",
        "restart_service:api-gateway",
    ],
    services_metadata={
        "frontend": {"version": "v5.1.0", "last_deploy": "2 days ago", "owner": "frontend-team"},
        "api-gateway": {"version": "v1.8.0", "last_deploy": "45 minutes ago - CONFIG CHANGE ONLY", "owner": "platform-team", "deploy_note": "rate_limiter.yaml updated - global limit accidentally set to 10 req/min"},
        "payment-service": {"version": "v3.2.1", "last_deploy": "30 minutes ago", "owner": "payments-team", "deploy_note": "Added retry logic - RED HERRING deploy, unrelated to throttling"},
        "notification-service": {"version": "v2.0.4", "last_deploy": "1 week ago", "owner": "comms-team"},
        "cdn": {"version": "v4.2.0", "last_deploy": "3 days ago", "owner": "infra-team"},
    },
    hidden_logs={
        "api-gateway": [
            LogEntry(timestamp="T+45s", service="api-gateway", level="ERROR", message="global_rate_limit dropped from 10000 to 10 req/min"),
            LogEntry(timestamp="T+50s", service="api-gateway", level="ERROR", message="global_rate_limit dropped from 10000 to 10 req/min"),
            LogEntry(timestamp="T+55s", service="api-gateway", level="ERROR", message="global_rate_limit dropped from 10000 to 10 req/min"),
            LogEntry(timestamp="T+60s", service="api-gateway", level="ERROR", message="global_rate_limit dropped from 10000 to 10 req/min"),
            LogEntry(timestamp="T+65s", service="api-gateway", level="ERROR", message="global_rate_limit dropped from 10000 to 10 req/min"),
        ],
        "payment-service": [
            LogEntry(timestamp="T+45s", service="payment-service", level="ERROR", message="upstream rate limit"),
            LogEntry(timestamp="T+50s", service="payment-service", level="ERROR", message="upstream rate limit"),
            LogEntry(timestamp="T+55s", service="payment-service", level="ERROR", message="upstream rate limit"),
            LogEntry(timestamp="T+60s", service="payment-service", level="ERROR", message="upstream rate limit"),
            LogEntry(timestamp="T+65s", service="payment-service", level="ERROR", message="upstream rate limit"),
        ],
        "frontend": [
            LogEntry(timestamp="T+45s", service="frontend", level="ERROR", message="HTTP 429 from api-gateway"),
            LogEntry(timestamp="T+50s", service="frontend", level="ERROR", message="HTTP 429 from api-gateway"),
            LogEntry(timestamp="T+55s", service="frontend", level="ERROR", message="HTTP 429 from api-gateway"),
        ],
        "notification-service": [
            LogEntry(timestamp="T+45s", service="notification-service", level="ERROR", message="throttled by api-gateway"),
            LogEntry(timestamp="T+50s", service="notification-service", level="ERROR", message="throttled by api-gateway"),
            LogEntry(timestamp="T+55s", service="notification-service", level="ERROR", message="throttled by api-gateway"),
        ],
        "cdn": [
            LogEntry(timestamp="T+45s", service="cdn", level="ERROR", message="throttled by api-gateway"),
            LogEntry(timestamp="T+50s", service="cdn", level="ERROR", message="throttled by api-gateway"),
            LogEntry(timestamp="T+55s", service="cdn", level="ERROR", message="throttled by api-gateway"),
        ],
    },
    dependency_map={
        "frontend": ["api-gateway"],
        "api-gateway": [],
        "payment-service": ["api-gateway"],
        "notification-service": ["api-gateway"],
        "cdn": ["api-gateway", "frontend"],
    },
)

SCENARIOS: dict[str, IncidentScenario] = {
    "easy": easy_scenario,
    "medium": medium_scenario,
    "hard": hard_scenario,
}


def get_scenario(task_id: str) -> IncidentScenario:
    """Get a scenario by task_id. Raises ValueError if not found."""
    if task_id not in SCENARIOS:
        raise ValueError(f"Unknown task_id: {task_id!r}. Must be one of: {list(SCENARIOS.keys())}")
    return SCENARIOS[task_id]