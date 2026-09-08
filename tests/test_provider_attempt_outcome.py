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
    with patch(
        "pdd.agentic_common._get_provider_cli_version",
        return_value="2.1.263 (Claude Code)",
    ):
        ac.reset_disabled_providers()
        yield
        ac.reset_disabled_providers()


def _completed(stdout: str, *, returncode: int = 1, stderr: str = ""):
    return subprocess.CompletedProcess(
        ["synthetic-provider"], returncode, stdout=stdout, stderr=stderr
    )


def _spooled(stdout: str, *, returncode: int) -> ac._SpooledCompletedProcess:
    payload = stdout.encode()
    return ac._SpooledCompletedProcess(
        args=["codex"],
        returncode=returncode,
        stdout_file=io.BytesIO(payload),
        stderr_file=io.BytesIO(b""),
        stdout_bytes=len(payload),
        stderr_bytes=0,
        stdout_head=stdout,
        stdout_tail="",
        stderr_head="",
        stderr_tail="",
    )


def _opencode_step_start() -> dict[str, Any]:
    return {
        "type": "step_start",
        "timestamp": 1,
        "sessionID": "ses_synthetic",
        "part": {
            "id": "prt_start",
            "sessionID": "ses_synthetic",
            "messageID": "msg_synthetic",
            "type": "step-start",
        },
    }


def _opencode_tool_use() -> dict[str, Any]:
    return {
        "type": "tool_use",
        "timestamp": 2,
        "sessionID": "ses_synthetic",
        "part": {
            "id": "prt_tool",
            "sessionID": "ses_synthetic",
            "messageID": "msg_synthetic",
            "type": "tool",
            "callID": "call_synthetic",
            "tool": "synthetic-tool",
            "state": {"status": "completed"},
        },
    }


def _codex_current_item_pair(item_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if item_type == "command_execution":
        base = {
            "id": "item_command",
            "type": item_type,
            "command": "true",
            "aggregated_output": "",
            "exit_code": None,
        }
        return (
            {**base, "status": "in_progress"},
            {**base, "status": "completed", "exit_code": 0},
        )
    if item_type == "file_change":
        base = {
            "id": "item_file",
            "type": item_type,
            "changes": [{"path": "synthetic.txt", "kind": "update"}],
        }
        completed = {**base, "status": "completed"}
        return (completed, completed)
    if item_type == "mcp_tool_call":
        base = {
            "id": "item_mcp",
            "type": item_type,
            "server": "synthetic-server",
            "tool": "synthetic-tool",
            "arguments": {},
        }
        return (
            {**base, "status": "in_progress", "result": None, "error": None},
            {
                **base,
                "status": "completed",
                "result": {"content": [], "structured_content": None},
                "error": None,
            },
        )
    if item_type == "collab_tool_call":
        base = {
            "id": "item_collab",
            "type": item_type,
            "tool": "spawn_agent",
            "sender_thread_id": "thread_parent",
            "receiver_thread_ids": ["thread_child"],
            "prompt": "synthetic task",
            "agents_states": {
                "thread_child": {"status": "running", "message": None}
            },
        }
        return (
            {**base, "status": "in_progress"},
            {**base, "status": "completed"},
        )
    if item_type == "web_search":
        item = {
            "id": "item_search",
            "type": item_type,
            "query": "synthetic",
            "action": {"type": "search", "query": "synthetic"},
        }
        return (item, item)
    raise AssertionError(f"unsupported synthetic item type: {item_type}")


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
            patch(
                "pdd.agentic_common._get_provider_cli_version",
                return_value="2.1.263 (Claude Code)",
            )
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


@pytest.mark.parametrize(
    "path,value",
    [
        (("usage", "service_tier"), {"future_billable_units": 1}),
        (("session_id",), {"request_started": True}),
        (("fast_mode_state",), ["unknown-schema"]),
    ],
    ids=["nested-metadata", "session-id", "fast-mode"],
)
def test_wrong_reviewed_field_types_are_ambiguous(path, value):
    envelope = _zero_work_rejection()
    target = envelope
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    receipt = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, json.dumps(envelope), ""
    )
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "version",
    ["2.1.264 (Claude Code)", "2.1.263.1", "2.1.263-beta", "2.1.263"],
)
def test_unreviewed_anthropic_cli_version_is_ambiguous(version):
    with patch(
        "pdd.agentic_common._get_provider_cli_version", return_value=version
    ):
        receipt = ac._create_provider_attempt_receipt(
            "anthropic", 1, 1, json.dumps(_zero_work_rejection()), ""
        )
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonstandard_anthropic_numbers_are_ambiguous(bad_value):
    envelope = _zero_work_rejection()
    envelope["duration_ms"] = bad_value
    receipt = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, json.dumps(envelope), ""
    )
    assert receipt.work_disposition == "ambiguous"


def test_duplicate_anthropic_json_keys_are_ambiguous():
    raw = json.dumps(_zero_work_rejection(), separators=(",", ":"))
    contradictory = '{"usage":{"input_tokens":9},' + raw[1:]
    receipt = ac._create_provider_attempt_receipt(
        "anthropic", 1, 1, contradictory, ""
    )
    assert receipt.work_disposition == "ambiguous"


def test_duplicate_anthropic_success_bypass_is_demoted_at_public_boundary(tmp_path):
    raw = json.dumps(_zero_work_rejection(), separators=(",", ":"))
    contradictory = raw[:-1] + ',"is_error":false}'
    result, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=_completed(contradictory, returncode=0),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"
    assert "sk-syntheticsecret" not in result.output_text


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


def test_google_unknown_schema_with_known_tokens_is_ambiguous():
    receipt = ac._create_provider_attempt_receipt(
        "google",
        1,
        1,
        json.dumps(
            {
                "type": "future_result",
                "stats": {
                    "models": {"synthetic": {"tokens": {"prompt": 1}}}
                },
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


def test_opencode_nonfinite_counter_fails_closed(tmp_path):
    stdout = "\n".join(
        [
            '{"type":"step_finish","part":{"usage":{"input":1e309}}}',
            json.dumps({"type": "error", "message": "synthetic failure"}),
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout),
        )
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "tail",
    ['{"type":', json.dumps({"type": "future.event", "cost": 1})],
    ids=["malformed", "unknown-event"],
)
def test_opencode_mixed_unreviewed_jsonl_is_ambiguous(tail):
    stdout = "\n".join(
        [json.dumps({"type": "text", "part": {"text": "partial"}}), tail]
    )
    receipt = ac._create_provider_attempt_receipt(
        "opencode", 1, 1, stdout, ""
    )
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "tail",
    ['{"type":', json.dumps({"type": "future.event"})],
    ids=["malformed", "unknown-event"],
)
def test_opencode_exit_zero_unreviewed_tail_is_demoted(tmp_path, tail):
    stdout = "\n".join(
        [
            json.dumps({"type": "text", "part": {"text": "useful output"}}),
            json.dumps({"type": "step_finish", "part": {"cost": 0.01}}),
            tail,
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout, returncode=0),
        )
    assert result.success is False
    assert result.cost_usd == 0
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "event",
    [_opencode_step_start(), _opencode_tool_use()],
    ids=["request-accepted", "tool-used"],
)
def test_opencode_reviewed_boundary_activity_claims_started(event):
    stdout = "\n".join(
        [json.dumps(event), json.dumps({"type": "error", "message": "failed"})]
    )
    receipt = ac._create_provider_attempt_receipt("opencode", 1, 1, stdout, "")
    assert receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "step_start"},
        {
            **_opencode_tool_use(),
            "part": {**_opencode_tool_use()["part"], "callID": 1},
        },
        {
            **_opencode_tool_use(),
            "part": {
                **_opencode_tool_use()["part"],
                "state": {"status": "running"},
            },
        },
        {
            **_opencode_step_start(),
            "part": {
                **_opencode_step_start()["part"],
                "sessionID": "ses_other",
            },
        },
    ],
    ids=["missing-envelope", "bad-call-id", "unemitted-status", "session-mismatch"],
)
def test_opencode_unreviewed_activity_shape_is_ambiguous(event):
    receipt = ac._create_provider_attempt_receipt(
        "opencode", 1, 1, json.dumps(event), ""
    )
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "malformed_event",
    [
        {"type": "text", "part": {"type": "text", "text": {"forged": "shape"}}},
        {"type": "step_finish", "part": {"cost": "0.01"}},
        {"type": "error", "message": {"forged": "shape"}},
        {"type": "session.end", "model": {"forged": "shape"}},
        {"type": "text", "part": 17, "text": "OK"},
        {"type": "session.end", "session": {"model": {"forged": "shape"}}},
        {
            "type": "error",
            "message": "synthetic failure",
            "error": {"forged": "shape"},
        },
    ],
    ids=[
        "text",
        "step-finish",
        "error",
        "session-end",
        "scalar-text-part",
        "nested-session-model",
        "dual-error",
    ],
)
@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_opencode_malformed_known_event_overrides_activity(
    tmp_path, malformed_event, returncode
):
    stdout = "\n".join(
        [
            json.dumps(_opencode_step_start()),
            json.dumps(malformed_event),
            json.dumps({"type": "error", "message": "synthetic failure"}),
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout, returncode=returncode),
        )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


def test_opencode_supported_session_end_remains_successful(tmp_path):
    stdout = "\n".join(
        [
            json.dumps(_opencode_step_start()),
            json.dumps(
                {
                    "type": "text",
                    "part": {"text": "A sufficiently long OpenCode answer."},
                }
            ),
            json.dumps({"type": "step_finish", "part": {"cost": 0.01}}),
            json.dumps({"type": "session.end", "model": "synthetic/model"}),
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout, returncode=0),
        )
    assert result.success is True
    assert result.provider_attempt_receipt is None


def test_opencode_short_text_false_positive_preserves_started_receipt(tmp_path):
    stdout = "\n".join(
        [
            json.dumps(_opencode_step_start()),
            json.dumps({"type": "text", "part": {"text": "OK"}}),
            json.dumps({"type": "step_finish", "part": {"cost": 0}}),
            json.dumps({"type": "session.end", "model": "synthetic/model"}),
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout, returncode=0),
        )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


def test_opencode_nested_error_overrides_partial_success_without_leak(tmp_path):
    private_detail = "synthetic private provider detail Bearer sk-not-a-real-secret"
    stdout = "\n".join(
        [
            json.dumps(_opencode_step_start()),
            json.dumps(
                {
                    "type": "text",
                    "part": {
                        "text": "A sufficiently long partial synthetic answer."
                    },
                }
            ),
            json.dumps(
                {
                    "type": "step_finish",
                    "part": {
                        "cost": 0.01,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "name": "ProviderError",
                        "data": {"message": private_detail},
                    },
                }
            ),
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout, returncode=0),
        )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"
    assert private_detail not in result.output_text
    assert "sk-not-a-real-secret" not in result.output_text


@pytest.mark.parametrize(
    "event",
    [_opencode_step_start(), _opencode_tool_use()],
    ids=["request-accepted", "tool-used"],
)
def test_opencode_exit_zero_activity_preserves_started_receipt(tmp_path, event):
    stdout = "\n".join(
        [
            json.dumps(event),
            json.dumps({"type": "session.end", "model": "synthetic/model"}),
        ]
    )
    with patch.dict("os.environ", {"OPENCODE_MODEL": "synthetic/model"}, clear=False):
        result, _, _ = _run_public(
            tmp_path,
            providers=["opencode"],
            boundary_result=_completed(stdout, returncode=0),
        )
    assert result.success is False
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


def test_codex_exit_zero_message_then_failure_preserves_started_receipt(tmp_path):
    payload = (
        "\n".join(
            [
                json.dumps({"type": "init"}),
                json.dumps(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "A substantive synthetic answer.",
                    }
                ),
                json.dumps({"type": "turn.failed"}),
            ]
        )
        + "\n"
    ).encode()
    boundary = ac._SpooledCompletedProcess(
        args=["codex"],
        returncode=0,
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
    assert result.success is False
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


@pytest.mark.parametrize(
    "item",
    [
        {"type": "tool_call", "tool": "synthetic-tool"},
        {
            "type": "tool_output",
            "tool_calls": [{"function": {"name": "synthetic-tool"}}],
        },
    ],
    ids=["tool-call", "tool-output"],
)
def test_codex_reviewed_tool_item_claims_started(item):
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": item}),
            json.dumps({"type": "turn.failed"}),
        ]
    )
    receipt = ac._create_provider_attempt_receipt("openai", 1, 1, stdout, "")
    assert receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
@pytest.mark.parametrize(
    "item",
    [
        {"type": "agent_message", "text": {"forged": "shape"}},
        {"type": "tool_call", "tool": {"forged": "shape"}},
        {"type": "tool_output", "tool_calls": ["not-an-object"]},
    ],
    ids=["agent-message-text", "tool-name", "tool-calls"],
)
def test_codex_malformed_modern_item_is_ambiguous(tmp_path, item, returncode):
    stdout = (
        "\n".join(
            [
                json.dumps({"type": "item.completed", "item": item}),
                json.dumps({"type": "turn.failed"}),
            ]
        )
        + "\n"
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_codex_started_tool_item_claims_started(tmp_path, returncode):
    stdout = (
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {"type": "tool_call", "tool": "synthetic-tool"},
                    }
                ),
                json.dumps({"type": "turn.failed"}),
            ]
        )
        + "\n"
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
@pytest.mark.parametrize(
    "event",
    [
        {"type": "item.started", "item": {"type": "future.item"}},
        {
            "type": "item.completed",
            "item": {
                "id": "item_message",
                "type": "agent_message",
                "text": "synthetic output",
                "tool_calls": [{"function": {"name": "forged"}}],
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_reasoning",
                "type": "reasoning",
                "text": "synthetic reasoning",
                "tool": "cross-variant",
            },
        },
        {
            "type": "item.completed",
            "item": {
                **_codex_current_item_pair("command_execution")[1],
                "tool_calls": [{"name": "cross-variant"}],
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "tool_call", "text": "cross variant"},
        },
        {
            "type": "item.completed",
            "item": {"type": "tool_output", "tool": "cross-variant"},
        },
        {
            "type": "item.completed",
            "item": {
                "type": "tool_output",
                "tool_calls": [
                    {
                        "tool": "synthetic-tool",
                        "function": {"name": {"forged": "shape"}},
                    }
                ],
            },
        },
        {"type": "item.started", "item": {"type": "tool_call"}},
    ],
    ids=[
        "unknown-started-item",
        "agent-message-cross-field",
        "reasoning-cross-field",
        "command-cross-field",
        "tool-call-cross-field",
        "tool-output-cross-field",
        "contradictory-call-name",
        "missing-started-tool",
    ],
)
def test_codex_unreviewed_item_overrides_prior_activity(
    tmp_path, event, returncode
):
    stdout = (
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "synthetic progress",
                    }
                ),
                json.dumps(event),
                json.dumps({"type": "turn.failed"}),
            ]
        )
        + "\n"
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_codex_started_agent_message_proves_acceptance(tmp_path, returncode):
    stdout = (
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {"type": "agent_message"},
                    }
                ),
                json.dumps({"type": "turn.failed"}),
            ]
        )
        + "\n"
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
@pytest.mark.parametrize(
    "item_type",
    [
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "collab_tool_call",
        "web_search",
    ],
)
def test_codex_current_tool_items_claim_started(tmp_path, item_type, returncode):
    started_item, completed_item = _codex_current_item_pair(item_type)
    stdout = (
        "\n".join(
            [
                json.dumps({"type": "item.started", "item": started_item}),
                json.dumps({"type": "item.completed", "item": completed_item}),
                json.dumps({"type": "turn.failed"}),
            ]
        )
        + "\n"
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


def test_codex_current_tool_stream_preserves_exit_zero_success(tmp_path):
    started_item, completed_item = _codex_current_item_pair("command_execution")
    stdout = (
        "\n".join(
            [
                json.dumps({"type": "item.started", "item": started_item}),
                json.dumps({"type": "item.completed", "item": completed_item}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item_message",
                            "type": "agent_message",
                            "text": "A sufficiently long synthetic Codex answer.",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
            ]
        )
        + "\n"
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=0),
    )
    assert result.success is True
    assert result.provider_attempt_receipt is None


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_codex_item_updated_todo_claims_started(tmp_path, returncode):
    stdout = json.dumps(
        {
            "type": "item.updated",
            "item": {
                "id": "item_todo",
                "type": "todo_list",
                "items": [{"text": "synthetic step", "completed": False}],
            },
        }
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
@pytest.mark.parametrize(
    "item",
    [
        {
            **_codex_current_item_pair("command_execution")[0],
            "status": "declined",
        },
        {
            **_codex_current_item_pair("file_change")[0],
            "status": "in_progress",
        },
        {
            **_codex_current_item_pair("mcp_tool_call")[0],
            "status": "failed",
            "result": None,
            "error": {"message": "synthetic tool failure"},
        },
    ],
    ids=["declined-command", "in-progress-file", "failed-mcp"],
)
def test_codex_provider_real_status_variants_claim_started(
    tmp_path, returncode, item
):
    stdout = json.dumps({"type": "item.completed", "item": item})
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
@pytest.mark.parametrize(
    "action",
    [
        {"type": "search", "queries": ["synthetic"]},
        {"type": "open_page", "url": "https://example.invalid"},
        {
            "type": "find_in_page",
            "url": "https://example.invalid",
            "pattern": "synthetic",
        },
        {"type": "other"},
    ],
    ids=["search", "open-page", "find-in-page", "other"],
)
def test_codex_web_search_action_variants_claim_started(
    tmp_path, returncode, action
):
    item = {
        **_codex_current_item_pair("web_search")[0],
        "action": action,
    }
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(
            json.dumps({"type": "item.completed", "item": item}),
            returncode=returncode,
        ),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_codex_provider_real_web_search_double_id_claims_started(
    tmp_path, returncode
):
    # Codex 0.153.2 flattens outer ThreadItem.id and inner WebSearchItem.id.
    stdout = (
        '{"type":"item.completed","item":{"id":"item_outer",'
        '"type":"web_search","id":"item_inner","query":"synthetic",'
        '"action":{"type":"other"}}}'
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize(
    "stdout",
    [
        '{"type":"item.completed","item":{"id":"one",'
        '"type":"agent_message","id":"two","text":"synthetic"}}',
        '{"type":"item.completed","item":{"id":"one",'
        '"type":"web_search","id":7,"query":"synthetic",'
        '"action":{"type":"other"}}}',
    ],
    ids=["duplicate-id-wrong-variant", "duplicate-id-wrong-type"],
)
def test_codex_unreviewed_duplicate_ids_remain_ambiguous(tmp_path, stdout):
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=1),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
@pytest.mark.parametrize(
    "item",
    [
        {
            key: value
            for key, value in _codex_current_item_pair("command_execution")[1].items()
            if key != "exit_code"
        },
        {
            **_codex_current_item_pair("command_execution")[0],
            "exit_code": 0,
        },
        {
            **_codex_current_item_pair("mcp_tool_call")[0],
            "status": "completed",
        },
        {
            "id": "item_search",
            "type": "web_search",
            "query": "synthetic",
        },
        {
            "id": "item_message",
            "type": "agent_message",
            "text": "synthetic",
            "tool_calls": [{"tool": "forged"}],
        },
    ],
    ids=[
        "completed-command-no-exit",
        "started-command-with-exit",
        "completed-mcp-no-result",
        "web-search-no-action",
        "agent-message-cross-field",
    ],
)
def test_codex_malformed_current_items_remain_ambiguous(
    tmp_path, returncode, item
):
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(
            json.dumps({"type": "item.completed", "item": item}),
            returncode=returncode,
        ),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
@pytest.mark.parametrize(
    "event_type", ["turn.failed", "session.failed", "task.failed", "error"]
)
@pytest.mark.parametrize(
    "evidence",
    [
        {"usage": {"input_tokens": 1, "output_tokens": 0}},
        {"cost": 0.01},
    ],
    ids=["usage", "cost"],
)
def test_codex_failure_event_accounting_claims_started(
    tmp_path, event_type, evidence, returncode
):
    event = {"type": event_type, **evidence}
    if event_type == "error":
        event["message"] = "synthetic failure"
    stdout = json.dumps(event) + "\n"
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize(
    "event",
    [
        {
            "type": "error",
            "message": {"forged": "shape"},
            "cost": 0.01,
        },
        {
            "type": "turn.failed",
            "error": 17,
            "usage": {"input_tokens": 1},
        },
    ],
    ids=["malformed-error-message", "malformed-failure-error"],
)
@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_codex_malformed_failure_shape_overrides_accounting(
    tmp_path, event, returncode
):
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(json.dumps(event), returncode=returncode),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "event",
    [
        {
            "type": "turn.completed",
            "usage": {"reasoning_output_tokens": 1},
        },
        {
            "type": "item.completed",
            "item": _codex_current_item_pair("command_execution")[1],
        },
    ],
    ids=["current-usage", "current-item"],
)
def test_codex_classification_is_newline_invariant(event):
    raw = json.dumps(event)
    without_newline = ac._create_provider_attempt_receipt(
        "openai", 1, 1, raw, ""
    )
    with_newline = ac._create_provider_attempt_receipt(
        "openai", 1, 1, raw + "\n", ""
    )
    assert without_newline.work_disposition == "started_or_billable"
    assert with_newline.work_disposition == without_newline.work_disposition


@pytest.mark.parametrize("terminator", ["", "\n"], ids=["no-newline", "newline"])
def test_codex_public_nonspooled_single_event_is_schema_validated(
    tmp_path, terminator
):
    valid = {
        "type": "item.completed",
        "item": {
            "id": "item_message",
            "type": "agent_message",
            "text": "A complete synthetic Codex answer. " * 4,
        },
    }
    successful, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_completed(json.dumps(valid) + terminator, returncode=0),
    )
    assert successful.success is True
    assert successful.provider_attempt_receipt is None

    malformed = {
        "type": "item.completed",
        "usage": {"input_tokens": 1},
        "item": {
            "id": "item_message",
            "type": "agent_message",
            "text": {"forged": ["not provider-valid text"]},
        },
    }
    rejected, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_completed(json.dumps(malformed) + terminator, returncode=0),
    )
    assert rejected.success is False
    assert rejected.provider_attempt_receipt.work_disposition == "ambiguous"


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


def test_codex_legacy_assistant_message_before_failure_claims_started():
    stdout = "\n".join(
        [
            json.dumps({"type": "init"}),
            json.dumps(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "A substantive synthetic answer.",
                }
            ),
            json.dumps({"type": "turn.failed"}),
        ]
    )
    receipt = ac._create_provider_attempt_receipt("openai", 1, 1, stdout, "")
    assert receipt.work_disposition == "started_or_billable"


@pytest.mark.parametrize(
    "role,content",
    [
        ("assistant", ["not", "a", "string"]),
        ("assistant", {"forged": "payload"}),
        ("assistant", 1),
        ("user", "synthetic prompt"),
    ],
)
def test_codex_unreviewed_legacy_message_shape_is_ambiguous(role, content):
    stdout = "\n".join(
        [
            json.dumps({"type": "init"}),
            json.dumps({"type": "message", "role": role, "content": content}),
            json.dumps({"type": "turn.failed"}),
        ]
    )
    receipt = ac._create_provider_attempt_receipt("openai", 1, 1, stdout, "")
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "tail",
    ['{"type":', json.dumps({"type": "future.event", "usage": {}})],
    ids=["malformed", "unknown-event"],
)
def test_codex_mixed_unreviewed_jsonl_is_ambiguous(tail):
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            ),
            tail,
        ]
    )
    receipt = ac._create_provider_attempt_receipt("openai", 1, 1, stdout, "")
    assert receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize(
    "tail",
    ['{"type":', json.dumps({"type": "future.event"})],
    ids=["malformed", "unknown-event"],
)
def test_codex_exit_zero_unreviewed_tail_is_demoted(tmp_path, tail):
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "A sufficiently long synthetic answer.",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
            tail,
        ]
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_completed(stdout, returncode=0),
    )
    assert result.success is False
    assert result.cost_usd == 0
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


def test_codex_supported_legacy_init_remains_successful(tmp_path):
    stdout = "\n".join(
        [
            json.dumps({"type": "init"}),
            json.dumps(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "synthetic progress",
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "output": "A sufficiently long legacy Codex answer.",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
        ]
    )
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_completed(stdout, returncode=0),
    )
    assert result.success is True
    assert result.provider_attempt_receipt is None


@pytest.mark.parametrize(
    "stdout",
    [
        json.dumps(
            {
                "type": "result",
                "result": "A sufficiently long synthetic Codex answer.",
                "usage": {"input_tokens": float("inf"), "output_tokens": 0},
            }
        ),
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "A sufficiently long synthetic Codex answer.",
                        },
                    }
                ),
                '{"type":"turn.completed","usage":{"input_tokens":1e309}}',
            ]
        ),
    ],
    ids=["single-json", "jsonl"],
)
def test_codex_nonfinite_exit_zero_usage_is_demoted(tmp_path, stdout):
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_completed(stdout, returncode=0),
    )
    assert result.success is False
    assert result.cost_usd == 0
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


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
        return_value=ac._ProviderRunResult(
            False,
            "synthetic PTY failure request_id=req_PRIVATE session_id=sess_PRIVATE",
            0,
            None,
        ),
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
    assert "req_PRIVATE" not in result.output_text
    assert "sess_PRIVATE" not in result.output_text


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


def test_single_attempt_false_positive_diagnostic_does_not_echo_private_ids(tmp_path):
    envelope = {
        "type": "result",
        "result": "request_id=req_PRIVATE session_id=sess_PRIVATE",
        "total_cost_usd": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "modelUsage": {},
    }
    result, _, _ = _run_public(
        tmp_path,
        providers=["anthropic"],
        boundary_result=_completed(json.dumps(envelope), returncode=0),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"
    assert "req_PRIVATE" not in result.output_text
    assert "sess_PRIVATE" not in result.output_text
