"""Provider-free story regressions for PDD issue #2422."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import pdd.agentic_common as ac


@pytest.fixture(autouse=True)
def _reset_provider_state():
    ac.reset_disabled_providers()
    yield
    ac.reset_disabled_providers()


from tests.test_provider_attempt_outcome import (
    _completed,
    _opencode_step_start,
    _opencode_tool_use,
    _run_public,
)


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
            **_opencode_tool_use(),
            "part": {
                **_opencode_tool_use()["part"],
                "state": {"status": []},
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
    ids=[
        "missing-envelope",
        "bad-call-id",
        "unemitted-status",
        "unhashable-status",
        "session-mismatch",
    ],
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
