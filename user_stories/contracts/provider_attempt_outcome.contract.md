<!-- pdd-story-contract derived-from-story="../story__provider_attempt_outcome.md" story-hash="40598df88317a233" issue-ref="promptdriven/pdd#2422" -->

# Contract: Trustworthy provider attempt outcomes

> Generated from the human-verified user story + issue. Do not hand-edit:
> it is regenerated to align whenever the Story changes. Humans verify the
> Story (`../story__provider_attempt_outcome.md`), not this contract.

## Covers
- AC1: Single-attempt execution constraint (`single_provider_attempt=True` disables retry/rotation).
- AC2: Bounded trustworthy private attempt disposition (structured, secret-safe evaluation of work).
- AC3: Strict "zero-work" verification (distinguishing authenticated pre-inference failure from any possible started work).
- AC4: Sanitized public diagnostics (preventing truncated JSON and secret leakage).

## Context
Downstream applications (like Generative-Video-Studio) invoke PDD's agentic execution path with a single attempt requested. When a provider CLI fails, the application needs a trustworthy indication of whether any work started (which could bill the account or change state) to safely decide whether to try an alternative credential.

## Acceptance Criteria
1. Given a task invocation with `single_provider_attempt=True` and a mock CLI that fails with a complete, untruncated JSON envelope indicating a pre-inference 403 API rejection, when PDD executes the task, then exactly one provider attempt is executed, and PDD returns a result with a trustworthy disposition confirming zero work started.
2. Given a task invocation with `single_provider_attempt=True` and a mock CLI that exits nonzero with truncated JSON, a process timeout, or stderr fallback, when PDD executes the task, then PDD returns a disposition of "ambiguous" rather than "zero-work" (failing closed).
3. Given a nonzero provider CLI exit containing raw secrets or credentials, when PDD formats the failure for public diagnostics, then the public error string is sanitized and bounded (never raw JSON or credential payloads), while the private contract preserves the trustworthy disposition details.
4. Given a standard task invocation where `single_provider_attempt` is false or omitted, when a provider fails, then existing success, usage, cost, tuple-unpacking, and default retry/fallback behaviors remain fully backward compatible.

## Oracle
These details matter for pass/fail:
- The execution output contains a private, bounded, and secret-safe disposition field bound to the specific provider attempt.
- The disposition resolves to a conclusively "not-started" (zero-work) state if and only if there is complete, untruncated evidence (such as zero cost, zero token/tool activity, and a validated pre-inference error envelope).
- Any incomplete, malformed, or ambiguous results (such as general HTTP status codes alone, process exit statuses, or truncated JSON) resolve to an "ambiguous" state.
- Exactly one attempt is executed when `single_provider_attempt=True` is active; no retries, provider fallback, or credential rotations are performed by PDD.
- Public diagnostics do not leak credential payloads, secret values, or raw private identifiers.

## Non-Oracle
These details should not matter:
- The internal shape of the classifier (whether it uses a versioned receipt structure or a parsed envelope classifier).
- The exact diagnostic phrasing of the public error text, provided it is bounded and sanitized.
- Differences in internal execution pathways between standard noninteractive CLIs and interactive PTY routes, so long as both enforce their respective disposition contracts.

## Negative Cases
- PDD retries, switches credentials, or shifts to alternative models when `single_provider_attempt=True` is specified.
- A process timeout or a truncated JSON response is classified as "conclusive zero-work" evidence, leading to an unsafe retry.
- Credential payloads, API keys, or raw secrets are leaked in the public diagnostic text or the private disposition contract.

## Non-Goals
- Building a universal GVS provider transport (direct SDK calls are out of scope).
- Having PDD itself own credential rotation or billing policy decisions for the caller.
- Repurposing the existing provider-environment-failure sink for zero-work evidence.

## Candidate Prompts
- `prompts/agentic_common_python.prompt` — Shared agentic CLI execution route that handles provider attempts and parsing. (primary)
- `prompts/agentic_multishot_python.prompt` — Wraps task runs and controls multi-shot repeat-runs, which must respect the single-attempt bounds. (related)

## Notes
- GVS issue context: GVS Authentication lifecycle #3726, Credential-exhaustion regression #3510, and downstream blocker PR #3803.
- Standard fallback handling currently truncates stdout to 500 characters (`result.stdout[:500]`), which cuts off trailing JSON fields. The new parser must handle complete envelopes safely.
