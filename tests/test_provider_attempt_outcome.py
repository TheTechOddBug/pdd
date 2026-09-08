"""Provider-free story regression coverage for PDD issue #2422.

The primary cases call the public ``run_agentic_task`` API and replace only
provider discovery plus the CLI process/result boundary. No provider account,
credential, network service, or paid model is used.
"""

from __future__ import annotations

import io
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import pdd.agentic_common as ac


pytestmark = pytest.mark.timeout(60)


def _zero_work_rejection() -> dict[str, Any]:
    """Sanitized shape observed from Claude CLI 2.1.263 at the real boundary."""
    return {
        "duration_api_ms": 0,
        "stop_reason": "stop_sequence",
        "session_id": "11111111-2222-4333-8444-555555555555",
        "total_cost_usd": 0,
        "usage": {
            "output_tokens_details": {"thinking_tokens": 0},
            "input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
            "server_tool_use": {
                "web_search_requests": 0,
                "web_fetch_requests": 0,
            },
            "service_tier": "standard",
            "cache_creation": {
                "ephemeral_1h_input_tokens": 0,
                "ephemeral_5m_input_tokens": 0,
            },
            "inference_geo": "",
            "iterations": [],
            "speed": "standard",
        },
        "modelUsage": {},
        "permission_denials": [],
        "terminal_reason": "api_error",
        "fast_mode_state": "off",
        "fast_mode_disabled_reason": "sdk_opt_in_required",
        "subagent_stats": {
            "spawned": 0,
            "requested": {"background": 0, "foreground": 0, "unset": 0},
            "started_in_background": 0,
            "max_depth": 0,
            "spawned_by_subagents": 0,
            "completed": 0,
            "failed": 0,
            "killed": {"parent": 0, "user": 0, "system": 0},
            "refused": {
                "depth_limit": 0,
                "concurrency_limit": 0,
                "budget": 0,
            },
            "by_type": {},
        },
        "is_error": True,
        # These are wrapper lifecycle fields, not evidence that inference ran.
        "num_turns": 1,
        "subtype": "success",
        "api_error_status": 403,
        "result": (
            "Failed to authenticate. API Error: 403 synthetic account has no "
            "access. Authorization: Bearer sk-syntheticsecret000000000000"
        ),
        "type": "result",
        "duration_ms": 65,
        "uuid": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        "queued_turn_count": 0,
    }


@pytest.fixture(autouse=True)
def _reset_provider_state():
    ac.reset_disabled_providers()
    yield
    ac.reset_disabled_providers()


def _completed(stdout: str, *, returncode: int = 1, stderr: str = ""):
    return subprocess.CompletedProcess(
        ["synthetic-provider"], returncode, stdout=stdout, stderr=stderr
    )


def _run_public(
    tmp_path: Path,
    *,
    providers: list[str],
    boundary_result: Any,
    provider_preference: str | None = None,
    max_retries: int = 4,
    before_attempt=None,
    single_provider_attempt: bool = True,
):
    preference = provider_preference or ",".join(providers)
    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                "os.environ",
                {"PDD_AGENTIC_PROVIDER": preference},
                clear=False,
            )
        )
        stack.enter_context(
            patch("pdd.agentic_common.get_available_agents", return_value=providers)
        )
        stack.enter_context(
            patch("pdd.agentic_common._find_cli_binary", return_value="/synthetic/cli")
        )
        stack.enter_context(
            patch(
                "pdd.agentic_common._get_google_cli_binary",
                return_value="/synthetic/google",
            )
        )
        stack.enter_context(
            patch("pdd.agentic_common._strip_anthropic_creds_for_claude_subprocess")
        )
        stack.enter_context(
            patch("pdd.agentic_common._get_provider_cli_version", return_value="test")
        )
        stack.enter_context(
            patch("pdd.agentic_common._codex_gpt_5_6_version_error", return_value=None)
        )
        run_mock = stack.enter_context(
            patch("pdd.agentic_common._subprocess_run")
        )
        spooled_mock = stack.enter_context(
            patch("pdd.agentic_common._subprocess_run_spooled")
        )
        if isinstance(boundary_result, BaseException):
            run_mock.side_effect = boundary_result
            spooled_mock.side_effect = boundary_result
        else:
            run_mock.return_value = boundary_result
            spooled_mock.return_value = boundary_result
        result = ac.run_agentic_task(
            "synthetic provider boundary",
            tmp_path,
            quiet=True,
            max_retries=max_retries,
            before_attempt=before_attempt,
            single_provider_attempt=single_provider_attempt,
            background_safe=True,
        )
    return result, run_mock, spooled_mock


def _receipt_dict(result: ac.AgenticTaskResult) -> dict[str, Any]:
    receipt = result.provider_attempt_receipt
    assert receipt is not None
    return result.to_dict()["provider_attempt_receipt"]


def test_complete_private_envelope_returns_not_started_and_blocks_fallback(tmp_path):
    raw = json.dumps(_zero_work_rejection(), separators=(",", ":"))
    assert len(raw) > 500
    assert raw.index('"api_error_status"') > 500
    attempts: list[tuple[str, int]] = []

    result, run_mock, _ = _run_public(
        tmp_path,
        providers=["anthropic", "google"],
        boundary_result=_completed(raw),
        before_attempt=lambda provider, attempt: attempts.append((provider, attempt)),
    )

    assert result.success is False
    assert run_mock.call_count == 1
    assert attempts == [("anthropic", 1)]
    assert _receipt_dict(result) == {
        "schema_version": "pdd.provider_attempt.v1",
        "provider": "anthropic",
        "attempt_number": 1,
        "failure_kind": "credential_or_account",
        "work_disposition": "not_started",
    }
    assert len(result.output_text) <= 500
    assert "sk-syntheticsecret" not in result.output_text
    assert "11111111-2222-4333-8444-555555555555" not in result.output_text
    assert "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee" not in result.output_text
    assert '"duration_api_ms"' not in result.output_text


def test_single_attempt_unstructured_diagnostic_never_echoes_private_text(tmp_path):
    private_text = (
        "authentication failed password=hunter2 "
        "session_id=sess_sensitive request_id=req_01JPRIVATE"
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=_completed("", stderr=private_text),
    )
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"
    assert len(result.output_text) <= 500
    for private_value in ("hunter2", "sess_sensitive", "req_01JPRIVATE"):
        assert private_value not in result.output_text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["usage"].__setitem__("input_tokens", 1),
        lambda data: data.__setitem__("total_cost_usd", 0.01),
        lambda data: data.__setitem__("duration_api_ms", 1),
        lambda data: data["usage"]["server_tool_use"].__setitem__(
            "web_search_requests", 1
        ),
        lambda data: data["subagent_stats"].__setitem__("spawned", 1),
        lambda data: data.__setitem__("queued_turn_count", 1),
        lambda data: data["usage"].__setitem__("iterations", [{"type": "task"}]),
        lambda data: data.__setitem__(
            "modelUsage", {"synthetic-model": {"costUSD": 0.01}}
        ),
    ],
    ids=[
        "tokens",
        "cost",
        "api-duration",
        "server-tool",
        "subagent",
        "queue",
        "task-iteration",
        "model-usage",
    ],
)
def test_reviewed_positive_activity_wins_over_auth_rejection(mutation):
    envelope = _zero_work_rejection()
    mutation(envelope)
    receipt = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, json.dumps(envelope), ""
    )
    assert receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize(
    "missing_field",
    [
        "type",
        "is_error",
        "terminal_reason",
        "api_error_status",
        "duration_api_ms",
        "total_cost_usd",
        "usage",
        "modelUsage",
        "permission_denials",
        "subagent_stats",
        "queued_turn_count",
        "num_turns",
    ],
)
def test_missing_required_anthropic_zero_work_field_is_ambiguous(
    missing_field
):
    envelope = _zero_work_rejection()
    del envelope[missing_field]
    receipt = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, json.dumps(envelope), ""
    )
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "stdout,stderr",
    [
        ('{"type":"result","is_error":true', ""),
        ("not-json", ""),
        ("", "HTTP 403 authentication failed"),
        (json.dumps({"type": "future_result", "is_error": True}), ""),
    ],
    ids=["truncated", "malformed", "stderr-only", "unknown-schema"],
)
def test_incomplete_or_unstructured_failure_is_ambiguous(tmp_path, stdout, stderr):
    result, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=_completed(stdout, stderr=stderr),
    )
    assert _receipt_dict(result)["work_disposition"] == "ambiguous"


def test_contradictory_and_forged_receipt_fields_are_ambiguous():
    for mutate in (
        lambda data: data.__setitem__("is_error", False),
        lambda data: data.__setitem__(
            "provider_attempt_receipt",
            {"work_disposition": "not_started", "provider": "forged"},
        ),
    ):
        envelope = _zero_work_rejection()
        mutate(envelope)
        receipt = ac._create_provider_attempt_receipt(
            "anthropic", 1, 1, json.dumps(envelope), ""
        )
        receipt_dict = receipt.to_dict()
        assert receipt.work_disposition == "ambiguous"
        assert receipt.provider == "anthropic"
        assert set(receipt_dict) == {
            "schema_version",
            "provider",
            "attempt_number",
            "failure_kind",
            "work_disposition",
        }


@pytest.mark.parametrize(
    "path,value",
    [
        (("request_started",), True),
        (("usage", "future_billable_units"), 1),
        (("usage", "server_tool_use", "code_execution_requests"), 1),
        (("subagent_stats", "scheduled"), 1),
    ],
    ids=["top-level", "usage", "tool", "subagent"],
)
def test_unknown_activity_fields_make_anthropic_evidence_ambiguous(path, value):
    envelope = _zero_work_rejection()
    target = envelope
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    receipt = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, json.dumps(envelope), ""
    )
    assert receipt.work_disposition == "ambiguous"


def test_unknown_schema_with_known_positive_field_is_ambiguous():
    envelope = _zero_work_rejection()
    envelope["type"] = "future_result"
    envelope["usage"]["input_tokens"] = 2
    receipt = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, json.dumps(envelope), ""
    )
    assert receipt.work_disposition == "ambiguous"


def test_contradictory_stderr_prevents_not_started():
    receipt = ac._create_provider_attempt_receipt(
        "anthropic",
        1,
        1,
        json.dumps(_zero_work_rejection()),
        "connection reset after request acceptance",
    )
    assert receipt.work_disposition == "ambiguous"


def test_explicit_incomplete_capture_and_unknown_provider_fail_closed():
    raw = json.dumps(_zero_work_rejection())
    incomplete = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, raw, "", output_complete=False
    )
    unknown = ac._create_provider_attempt_receipt(
        "provider-name-from-untrusted-output", 1, 1, raw, ""
    )
    assert incomplete.work_disposition == "ambiguous"
    assert unknown.work_disposition == "ambiguous"
    assert unknown.provider == "unknown"


@pytest.mark.parametrize(
    "exc",
    [
        subprocess.TimeoutExpired(
            ["synthetic"], 1, output=b'{"usage":{"input_tokens":0}', stderr=b""
        ),
        ConnectionResetError("synthetic reset"),
    ],
    ids=["timeout-with-partial-output", "connection-reset"],
)
def test_timeout_and_reset_are_ambiguous(tmp_path, exc):
    result, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=exc,
    )
    receipt = result.provider_attempt_receipt
    assert receipt is not None
    assert receipt.failure_kind == "transport"
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize("provider", ["google", "opencode"])
def test_other_standard_provider_schemas_do_not_borrow_anthropic_proof(
    tmp_path, provider
):
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=[provider],
            boundary_result=_completed(json.dumps(_zero_work_rejection())),
        )
    receipt = result.provider_attempt_receipt
    assert receipt is not None
    assert receipt.provider == provider
    assert receipt.work_disposition == "ambiguous"


def test_google_reviewed_positive_stats_yield_started_or_billable(tmp_path):
    envelope = {
        "type": "error",
        "message": "synthetic failure",
        "stats": {
            "models": {
                "synthetic-model": {"tokens": {"prompt": 1, "candidates": 0}}
            }
        },
    }
    result, _, _ = _run_public(
        tmp_path,
        providers=["google"],
        boundary_result=_completed(json.dumps(envelope)),
    )
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


def test_google_unknown_positive_field_does_not_claim_started():
    receipt = ac._create_provider_attempt_receipt(
        "google",
        1,
        1,
        json.dumps(
            {
                "type": "error",
                "stats": {"models": {"synthetic": {"future_counter": 1}}},
            }
        ),
        "",
    )
    assert receipt.work_disposition == "ambiguous"


def test_opencode_reviewed_text_event_yields_started_or_billable(tmp_path):
    stdout = "\n".join(
        [
            json.dumps({"type": "text", "part": {"text": "partial work"}}),
            json.dumps({"type": "error", "message": "synthetic failure"}),
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout),
        )
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


def test_codex_spooled_failure_attaches_ambiguous_receipt(tmp_path):
    stdout = io.BytesIO(b'{"type":"error","message":"synthetic failure"}\n')
    stderr = io.BytesIO(b"")
    boundary = ac._SpooledCompletedProcess(
        args=["codex"],
        returncode=1,
        stdout_file=stdout,
        stderr_file=stderr,
        stdout_bytes=len(stdout.getvalue()),
        stderr_bytes=0,
        stdout_head=stdout.getvalue().decode(),
        stdout_tail="",
        stderr_head="",
        stderr_tail="",
    )
    result, _, spooled_mock = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=boundary,
    )
    assert spooled_mock.call_count == 1
    assert result.provider_attempt_receipt.provider == "openai"
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


def test_codex_spooled_positive_usage_yields_started_or_billable(tmp_path):
    payload = (
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        )
        + "\n"
        + json.dumps({"type": "error", "message": "synthetic failure"})
        + "\n"
    ).encode()
    stdout = io.BytesIO(payload)
    boundary = ac._SpooledCompletedProcess(
        args=["codex"],
        returncode=1,
        stdout_file=stdout,
        stderr_file=io.BytesIO(b""),
        stdout_bytes=len(payload),
        stderr_bytes=0,
        stdout_head=payload.decode(),
        stdout_tail="",
        stderr_head="",
        stderr_tail="",
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=boundary,
    )
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


def test_codex_positive_usage_wins_over_simultaneous_auth_text(tmp_path):
    payload = (
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "error",
                "message": "401 unauthorized: access token could not be refreshed",
            }
        )
        + "\n"
    ).encode()
    boundary = ac._SpooledCompletedProcess(
        args=["codex"],
        returncode=1,
        stdout_file=io.BytesIO(payload),
        stderr_file=io.BytesIO(b""),
        stdout_bytes=len(payload),
        stderr_bytes=0,
        stdout_head=payload.decode(),
        stdout_tail="",
        stderr_head="",
        stderr_tail="",
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=boundary,
    )
    receipt = result.provider_attempt_receipt
    assert receipt.failure_kind == "credential_or_account"
    assert receipt.work_disposition == "started_or_billable"


def test_codex_unknown_positive_usage_field_does_not_claim_started():
    receipt = ac._create_provider_attempt_receipt(
        "openai",
        1,
        1,
        json.dumps(
            {"type": "turn.completed", "usage": {"future_counter": 1}}
        ),
        "",
    )
    assert receipt.work_disposition == "ambiguous"


def test_interactive_pty_failure_attaches_ambiguous_receipt(tmp_path):
    with patch.dict(
        "os.environ",
        {"PDD_CLAUDE_CODE_MODE": "interactive", "PDD_AGENTIC_PROVIDER": "anthropic"},
        clear=False,
    ), patch(
        "pdd.agentic_common.get_available_agents", return_value=["anthropic"]
    ), patch(
        "pdd.agentic_common._find_cli_binary", return_value="/synthetic/claude"
    ), patch(
        "pdd.agentic_common._strip_anthropic_creds_for_claude_subprocess"
    ), patch(
        "pdd.agentic_common._run_claude_interactive_with_mcp",
        return_value=ac._ProviderRunResult(False, "synthetic PTY failure", 0, None),
    ) as pty_boundary:
        result = ac.run_agentic_task(
            "synthetic PTY boundary",
            tmp_path,
            quiet=True,
            single_provider_attempt=True,
        )
    assert pty_boundary.call_count == 1
    assert result.provider_attempt_receipt.provider == "anthropic"
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


def test_interactive_pty_positive_usage_yields_started_or_billable(tmp_path):
    usage = {
        "claude": [
            {
                "model": "synthetic-model",
                "input_tokens": 1,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
        ]
    }
    with patch.dict(
        "os.environ",
        {"PDD_CLAUDE_CODE_MODE": "interactive", "PDD_AGENTIC_PROVIDER": "anthropic"},
        clear=False,
    ), patch(
        "pdd.agentic_common.get_available_agents", return_value=["anthropic"]
    ), patch(
        "pdd.agentic_common._find_cli_binary", return_value="/synthetic/claude"
    ), patch(
        "pdd.agentic_common._strip_anthropic_creds_for_claude_subprocess"
    ), patch(
        "pdd.agentic_common._run_claude_interactive_with_mcp",
        return_value=ac._ProviderRunResult(
            False, "synthetic PTY failure", 0, None, usage
        ),
    ):
        result = ac.run_agentic_task(
            "synthetic PTY boundary",
            tmp_path,
            quiet=True,
            single_provider_attempt=True,
        )
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


def test_receipt_none_and_tuple_compatibility_for_success_and_ordinary_run(tmp_path):
    success_envelope = {
        "type": "result",
        "result": "A sufficiently long successful synthetic provider response.",
        "total_cost_usd": 0.25,
        "usage": {"input_tokens": 2, "output_tokens": 3},
        "modelUsage": {"synthetic-model": {}},
    }
    successful, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=_completed(json.dumps(success_envelope), returncode=0),
    )
    assert successful.provider_attempt_receipt is None
    assert successful.to_dict()["provider_attempt_receipt"] is None
    success, output, cost, provider = successful
    assert success is True
    assert output == success_envelope["result"]
    assert cost == pytest.approx(0.25)
    assert provider == "anthropic"
    assert len(successful) == 5
    assert successful[4] == successful.usage

    ordinary, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=_completed(json.dumps(_zero_work_rejection())),
        max_retries=1,
        single_provider_attempt=False,
    )
    assert ordinary.provider_attempt_receipt is None
    assert ordinary.to_dict()["provider_attempt_receipt"] is None


def test_false_positive_work_and_provider_environment_fail_closed(tmp_path):
    worked_but_empty = {
        "type": "result",
        "result": "",
        "total_cost_usd": 0.1,
        "usage": {"input_tokens": 1, "output_tokens": 0},
        "modelUsage": {"synthetic-model": {"costUSD": 0.1}},
    }
    demoted, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=_completed(json.dumps(worked_but_empty), returncode=0),
        max_retries=1,
    )
    assert demoted.provider_attempt_receipt.work_disposition == "started_or_billable"

    with patch.dict(
        "os.environ", {"PDD_AGENTIC_PROVIDER": "anthropic"}, clear=False
    ), patch(
        "pdd.agentic_common.get_available_agents", return_value=["anthropic"]
    ), patch(
        "pdd.agentic_common._run_with_provider",
        return_value=ac._ProviderRunResult(
            False,
            "Provider runtime requires interactive configuration.",
            0,
            None,
            provider_environment_reason="trust_prompt",
        ),
    ):
        environment = ac.run_agentic_task(
            "synthetic provider environment",
            tmp_path,
            quiet=True,
            single_provider_attempt=True,
        )
    assert environment.provider_environment_failure == (
        "anthropic",
        "trust_prompt",
    )
    assert environment.provider_attempt_receipt.work_disposition == "ambiguous"
