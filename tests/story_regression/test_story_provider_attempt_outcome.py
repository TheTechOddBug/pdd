"""Generated story-backed regression tests.

This file is deterministic and safe to run without LLM/cloud credentials.
"""

from pathlib import Path

import pytest

PDD_STORY_ID = "provider_attempt_outcome"
PDD_STORY_HASH = "6b118aed50d9366f"
STORY_PATH = (
    Path(__file__).resolve().parent
    / "../../user_stories/story__provider_attempt_outcome.md"
)
CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "../../user_stories/contracts/provider_attempt_outcome.contract.md"
)


def _story_bundle() -> str:
    story = STORY_PATH.read_text(encoding="utf-8")
    if CONTRACT_PATH is not None and CONTRACT_PATH.exists():
        return story + "\n\n" + CONTRACT_PATH.read_text(encoding="utf-8")
    return story


def _bundle_hash() -> str:
    # Reuse the canonical primitive so the recorded PDD_STORY_HASH and
    # the gate's freshness check can never drift (pdd#1889). A
    # metadata-only prompt relink does not change this value.
    from pdd.story_test_generation import story_bundle_hash

    return story_bundle_hash(STORY_PATH)


@pytest.mark.story(story_id=PDD_STORY_ID)
def test_story_provider_attempt_outcome_oracle_contract():
    assert _bundle_hash() == PDD_STORY_HASH
    expected = [
        "These details matter for pass/fail:",
        "The execution output contains a private, bounded, and secret-safe disposition field bound to the specific provider attempt.",
        'The disposition resolves to a conclusively "not-started" (zero-work) state if and only if there is complete, untruncated evidence (such as zero cost, zero token/tool activity, and a validated pre-inference error envelope).',
        'Any incomplete, malformed, or ambiguous results (such as general HTTP status codes alone, process exit statuses, or truncated JSON) resolve to an "ambiguous" state.',
        "Exactly one attempt is executed when `single_provider_attempt=True` is active; no retries, provider fallback, or credential rotations are performed by PDD.",
        "Public diagnostics do not leak credential payloads, secret values, or raw private identifiers.",
    ]
    bundle = _story_bundle()
    assert expected, "story has no Oracle or Acceptance Criteria clauses"
    for clause in expected:
        assert clause in bundle


@pytest.mark.story(story_id=PDD_STORY_ID)
def test_story_provider_attempt_outcome_negative_cases():
    assert _bundle_hash() == PDD_STORY_HASH
    expected = [
        "PDD retries, switches credentials, or shifts to alternative models when `single_provider_attempt=True` is specified.",
        'A process timeout or a truncated JSON response is classified as "conclusive zero-work" evidence, leading to an unsafe retry.',
        "Credential payloads, API keys, or raw secrets are leaked in the public diagnostic text or the private disposition contract.",
    ]
    bundle = _story_bundle()
    for clause in expected:
        assert clause in bundle
