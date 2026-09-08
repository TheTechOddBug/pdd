"""Provider-free story regressions for PDD issue #2422."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

import pdd.agentic_common as ac


pytestmark = [
    pytest.mark.timeout(60),
    pytest.mark.story(story_id="provider_attempt_outcome"),
]


@pytest.fixture(autouse=True)
def _reset_provider_state():
    ac.reset_disabled_providers()
    yield
    ac.reset_disabled_providers()


from tests.test_provider_attempt_outcome import (
    _codex_current_item_pair,
    _completed,
    _run_public,
    _spooled,
)


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
        {"type": "tool_call", "name": "shell", "output": "synthetic.txt"},
        {
            "type": "tool_output",
            "tool_calls": [{"function": {"name": "synthetic-tool"}}],
        },
        {"type": "tool_output", "text": "synthetic tool output"},
    ],
    ids=[
        "tool-call",
        "legacy-named-tool-call",
        "tool-output",
        "legacy-text-tool-output",
    ],
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
        {"type": "tool_call", "name": "shell", "output": 7},
        {"type": "tool_output", "text": {"forged": "shape"}},
        {
            "type": "tool_call",
            "tool": "synthetic-tool",
            "name": "cross-variant",
            "output": "cross-variant",
        },
    ],
    ids=[
        "agent-message-text",
        "tool-name",
        "tool-calls",
        "legacy-tool-output-type",
        "legacy-output-text-type",
        "mixed-tool-call-unions",
    ],
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
        '{"type":"message","role":"assistant","content":"worked"}\n'
        '{"type":"error","message":"failed","extra":{"type":"web_search",'
        '"id":"nested-one","id":"nested-two"}}',
    ],
    ids=[
        "duplicate-id-wrong-variant",
        "duplicate-id-wrong-type",
        "duplicate-id-nested-out-of-position",
    ],
)
@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_codex_unreviewed_duplicate_ids_remain_ambiguous(
    tmp_path, stdout, returncode
):
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
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
        {
            **_codex_current_item_pair("command_execution")[0],
            "status": [],
        },
        {
            **_codex_current_item_pair("file_change")[0],
            "changes": [{"path": "synthetic.txt", "kind": {}}],
        },
        {
            **_codex_current_item_pair("mcp_tool_call")[0],
            "status": {},
        },
        {
            **_codex_current_item_pair("collab_tool_call")[0],
            "tool": [],
        },
        {
            **_codex_current_item_pair("web_search")[0],
            "action": {"type": []},
        },
    ],
    ids=[
        "completed-command-no-exit",
        "started-command-with-exit",
        "completed-mcp-no-result",
        "web-search-no-action",
        "agent-message-cross-field",
        "command-unhashable-status",
        "file-unhashable-kind",
        "mcp-unhashable-status",
        "collab-unhashable-tool",
        "web-unhashable-action",
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
def test_codex_unhashable_event_type_remains_ambiguous(tmp_path, returncode):
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(
            json.dumps({"type": [], "usage": {"input_tokens": 1}}),
            returncode=returncode,
        ),
    )
    assert result.success is False
    assert result.provider_attempt_receipt.work_disposition == "ambiguous"


@pytest.mark.parametrize("returncode", [0, 1], ids=["exit-zero", "nonzero-exit"])
def test_codex_deeply_nested_json_remains_ambiguous(tmp_path, returncode):
    stdout = "[" * 900 + "0" + "]" * 900
    result, _, _ = _run_public(
        tmp_path,
        providers=["openai"],
        boundary_result=_spooled(stdout, returncode=returncode),
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
