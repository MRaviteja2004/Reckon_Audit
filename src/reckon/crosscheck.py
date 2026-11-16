"""Stage 2.5 — cross-check pass with a second model (Phase 6).

The deterministic aligner can only confirm an assertion from what Stage 2 wrote down.
When Stage 2 missed an on-screen message — a common failure at low resolution or low
frame rate — the aligner reports assertion_not_observed even though the message did
render. That is a false finding on a faithful recording, the mirror image of the false
pass this audit exists to catch.

This stage routes the highest-stakes questions to a second model, framed as grounded
yes/no over specific frames pulled at full resolution:

  - Assertion checks — for every assertion the aligner could not confirm, "does text or
    a UI state matching X appear in these frames?" A confident "yes" overturns the
    finding; a "no" confirms it.
  - Low-confidence steps — observed actions the extractor itself was unsure about are
    re-examined before their match is published.

Only assertion verdicts modify the audit result, and only a confident "present" does so
— a single uncertain second opinion never silently flips a finding.
"""

import base64
import logging
import os
from enum import Enum
from pathlib import Path

import instructor
import openai
from instructor.core import InstructorRetryException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reckon.aligner import _order_navigation_supports_open
from reckon.frames import extract_evidence_frames_in_window, frame_to_jpeg_bytes
from reckon.latency import (
    LatencyMetrics,
    bind_retry_callback,
    elapsed_ms,
    monotonic_ms,
    resolve_latency,
)
from reckon.parser import make_openai_client
from reckon.retry import RetryCallback, call_with_retry

logger = logging.getLogger(__name__)
from reckon.schema import (
    ActionType,
    AuditResult,
    CrossCheckAnswer,
    CrossCheckKind,
    CrossCheckReport,
    CrossCheckTarget,
    CrossCheckVerdict,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    ObservedAction,
    ObservedTrace,
    SemanticDiagnostic,
)
from reckon.tolerance import (
    ASSERTION_ACTIONS,
    _DIGIT_RUN,
    _introduces_negation,
    _normalise_assertion_text,
    actions_compatible,
    verdict_for,
)

CROSSCHECK_MODEL = os.environ.get("CROSSCHECK_MODEL", "gpt-5.5")

# Observed actions the extractor scored below this are re-examined before publishing.
LOW_CONFIDENCE_THRESHOLD = 0.7

# Minimum cross-check confidence in a "present" answer before an assertion finding is
# overturned. A weak second opinion must not silently flip a finding to a pass.
OVERTURN_CONFIDENCE = 0.7

# Frames sampled per target window, and how far either side of the anchor to look.
FRAMES_PER_TARGET = 3
WINDOW_PAD_MS = 2500

# temperature is omitted (None → not sent): determinism is the goal, but the
# recommended GPT-5-family cross-check model accepts only its default temperature and
# rejects any explicit value.
_TEMPERATURE: float | None = None

_OPPOSITE_STATE_PHRASES = (
    (("enabled",), ("disabled",)),
    (("checked",), ("unchecked",)),
    (("success",), ("failure",)),
    (("completed",), ("pending",)),
    (("logged", "in"), ("logged", "out")),
    (("approved",), ("declined",)),
)

_CROSSCHECK_SYSTEM = """\
You are a careful visual verifier for a test-recording audit. You are shown a few \
frames from a UI recording and asked one grounded yes/no question about what is \
visible in them.

Answer only from what the frames actually show:
- "present" only if one frame directly and unambiguously shows the requested element \
or state in the relevant UI context.
- "absent" if the frames clearly do not show it.
- "unclear" if the frames are ambiguous or you cannot tell.

Text appearing on an unrelated screen does not prove that a requested UI element is \
visible. An unlabeled or ambiguous menu item is not direct evidence of a specifically \
named item. Identify the single frame that directly supports a "present" answer.

Do not assume an outcome because a test would expect it. Report only what you see.\
"""

_SEMANTIC_SYSTEM = """\
You compare two short descriptions from an already-aligned UI test step.

Return "equivalent" only when they express the same UI target, visible text, or state. \
Return "different" when their outcomes, values, or meanings materially differ. Return \
"unclear" when the descriptions do not provide enough information.

For assertion comparisons, use the supplied screen and page context when deciding \
whether different wording describes the same UI state.

Do not create actions, evidence, timestamps, indices, alignments, findings, or verdicts.\
"""


class _SemanticCandidateKind(str, Enum):
    target = "target"
    assertion = "assertion"


class _SemanticAnswer(str, Enum):
    equivalent = "equivalent"
    different = "different"
    unclear = "unclear"


class _EvidenceSupport(str, Enum):
    direct = "direct"
    indirect = "indirect"
    ambiguous = "ambiguous"
    none = "none"


class _SemanticCandidate(BaseModel):
    """One existing deterministic pair whose wording remains unresolved."""

    kind: _SemanticCandidateKind
    expected_step_index: int = Field(ge=1)
    observed_action_index: int = Field(ge=1)
    expected_action: ActionType
    observed_action: ActionType
    expected_text: str = Field(min_length=1)
    observed_text: str = Field(min_length=1)
    observed_visible_text: str | None = None
    observed_evidence_description: str | None = None
    observed_screen_description: str | None = None


class _SemanticDecision(BaseModel):
    """The only fields the semantic model may return."""

    model_config = ConfigDict(extra="forbid")

    answer: _SemanticAnswer
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class _AssertionCheck(BaseModel):
    """The cross-check model's structured answer for one target."""

    answer: CrossCheckAnswer
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the answer (0–1)")
    rationale: str = Field(description="One sentence grounded in what the frames show")
    supporting_frame_index: int | None = Field(
        None,
        ge=1,
        description="1-based frame index directly supporting a present answer",
    )
    evidence_support: _EvidenceSupport = Field(
        description="Whether the selected frame is direct, indirect, ambiguous, or absent evidence"
    )


def _describe_action_question(target: str | None, action: str) -> str:
    where = f" on {target!r}" if target else ""
    return (
        f"Does the screen show the user performing a {action} action{where}? "
        "Answer from the frames only."
    )


def _contains_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(
        tuple(tokens[start:start + width]) == phrase
        for start in range(len(tokens) - width + 1)
    )


def _texts_have_protected_contradiction(
    expected_text: str,
    observed_text: str,
) -> bool:
    expected_digits = _DIGIT_RUN.findall(expected_text)
    observed_digits = _DIGIT_RUN.findall(observed_text)
    if expected_digits and observed_digits and expected_digits != observed_digits:
        return True
    if (
        _introduces_negation(expected_text, observed_text)
        or _introduces_negation(observed_text, expected_text)
    ):
        return True

    expected_tokens = _normalise_assertion_text(expected_text)
    observed_tokens = _normalise_assertion_text(observed_text)
    return any(
        (
            _contains_phrase(expected_tokens, left)
            and _contains_phrase(observed_tokens, right)
        )
        or (
            _contains_phrase(expected_tokens, right)
            and _contains_phrase(observed_tokens, left)
        )
        for left, right in _OPPOSITE_STATE_PHRASES
    )


def _protected_contradiction_indices(
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
    audit_result: AuditResult,
) -> tuple[set[int], set[int]]:
    protected_steps = {
        finding.expected_step_index
        for finding in audit_result.findings
        if finding.type == FindingType.wrong_input_value
        and finding.expected_step_index is not None
    }
    protected_actions = {
        finding.observed_action_index
        for finding in audit_result.findings
        if finding.type == FindingType.wrong_input_value
        and finding.observed_action_index is not None
    }
    step_by_index = {step.index: step for step in expected_trace.steps}
    action_by_index = {action.index: action for action in observed_trace.actions}

    for finding in audit_result.findings:
        if (
            finding.type != FindingType.assertion_not_observed
            or finding.expected_step_index is None
            or finding.observed_action_index is None
        ):
            continue
        step = step_by_index.get(finding.expected_step_index)
        action = action_by_index.get(finding.observed_action_index)
        if step is None or step.assertion is None or action is None:
            continue
        evidence = action.assertion_evidence
        if evidence is None:
            continue
        observed_text = evidence.visible_text or evidence.description
        if _texts_have_protected_contradiction(step.assertion, observed_text):
            protected_steps.add(step.index)
            protected_actions.add(action.index)

    return protected_steps, protected_actions


def select_semantic_candidates(
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
    audit_result: AuditResult,
) -> list[_SemanticCandidate]:
    """Select ambiguous wording only from pairs already chosen by the aligner."""
    step_by_index = {step.index: step for step in expected_trace.steps}
    action_by_index = {action.index: action for action in observed_trace.actions}
    protected_steps, protected_actions = _protected_contradiction_indices(
        expected_trace, observed_trace, audit_result
    )
    findings_by_pair: dict[tuple[int, int], set[FindingType]] = {}
    for finding in audit_result.findings:
        if (
            finding.expected_step_index is None
            or finding.observed_action_index is None
        ):
            continue
        pair = (finding.expected_step_index, finding.observed_action_index)
        findings_by_pair.setdefault(pair, set()).add(finding.type)

    candidates: list[_SemanticCandidate] = []
    for step_verdict in audit_result.step_verdicts:
        action_index = step_verdict.matched_observed_index
        if action_index is None:
            continue
        step = step_by_index.get(step_verdict.expected_step_index)
        action = action_by_index.get(action_index)
        if step is None or action is None:
            continue
        if step.index in protected_steps or action.index in protected_actions:
            continue
        if not actions_compatible(step.action, action.action):
            continue

        pair_findings = findings_by_pair.get((step.index, action.index), set())
        if FindingType.wrong_input_value in pair_findings:
            continue

        if (
            FindingType.assertion_not_observed in pair_findings
            and step.assertion is not None
            and action.assertion_evidence is not None
        ):
            evidence = action.assertion_evidence
            observed_text = evidence.visible_text or evidence.description
            if observed_text.strip():
                candidates.append(_SemanticCandidate(
                    kind=_SemanticCandidateKind.assertion,
                    expected_step_index=step.index,
                    observed_action_index=action.index,
                    expected_action=step.action,
                    observed_action=action.action,
                    expected_text=step.assertion,
                    observed_text=observed_text,
                    observed_visible_text=evidence.visible_text,
                    observed_evidence_description=evidence.description,
                    observed_screen_description=action.screen_description,
                ))
                continue

        if (
            FindingType.wrong_target in pair_findings
            and step.target is not None
            and action.target is not None
            and not _texts_have_protected_contradiction(
                step.target, action.target
            )
        ):
            candidates.append(_SemanticCandidate(
                kind=_SemanticCandidateKind.target,
                expected_step_index=step.index,
                observed_action_index=action.index,
                expected_action=step.action,
                observed_action=action.action,
                expected_text=step.target,
                observed_text=action.target,
            ))

    return candidates


def verify_semantic_candidate(
    openai_client: instructor.Instructor,
    candidate: _SemanticCandidate,
    *,
    model: str = CROSSCHECK_MODEL,
    on_retry: RetryCallback | None = None,
) -> _SemanticDecision:
    """Ask for a text-only judgement without exposing code-owned identifiers."""
    user_content = (
        f"Comparison kind: {candidate.kind.value}\n"
        f"Expected action: {candidate.expected_action.value}\n"
        f"Observed action: {candidate.observed_action.value}\n"
        f"Expected text: {candidate.expected_text!r}\n"
        f"Observed text: {candidate.observed_text!r}"
    )
    if candidate.kind == _SemanticCandidateKind.assertion:
        user_content += (
            f"\nObserved visible text: {candidate.observed_visible_text!r}"
            "\nObserved evidence description: "
            f"{candidate.observed_evidence_description!r}"
            f"\nObserved screen context: {candidate.observed_screen_description!r}"
        )
    kwargs: dict = {
        "model": model,
        "response_model": _SemanticDecision,
        "messages": [
            {"role": "system", "content": _SEMANTIC_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    }
    if _TEMPERATURE is not None:
        kwargs["temperature"] = _TEMPERATURE
    response = call_with_retry(
        lambda: openai_client.chat.completions.create(**kwargs),
        on_retry=on_retry,
    )
    return _SemanticDecision.model_validate(response)


def _semantic_candidate_key(
    candidate: _SemanticCandidate,
) -> tuple[_SemanticCandidateKind, int, int, str, str]:
    return (
        candidate.kind,
        candidate.expected_step_index,
        candidate.observed_action_index,
        candidate.expected_text,
        candidate.observed_text,
    )


def _semantic_finding_present(
    audit_result: AuditResult,
    candidate: _SemanticCandidate,
) -> bool:
    finding_type = (
        FindingType.wrong_target
        if candidate.kind == _SemanticCandidateKind.target
        else FindingType.assertion_not_observed
    )
    return any(
        finding.type == finding_type
        and finding.expected_step_index == candidate.expected_step_index
        and finding.observed_action_index == candidate.observed_action_index
        for finding in audit_result.findings
    )


def _rebuild_audit_result(
    audit_result: AuditResult,
    findings: list[Finding],
) -> AuditResult:
    finding_types_by_step: dict[int, list[FindingType]] = {}
    for finding in findings:
        if finding.expected_step_index is not None:
            finding_types_by_step.setdefault(
                finding.expected_step_index, []
            ).append(finding.type)

    step_verdicts = []
    for step_verdict in audit_result.step_verdicts:
        finding_types = finding_types_by_step.get(
            step_verdict.expected_step_index, []
        )
        step_verdicts.append(step_verdict.model_copy(update={
            "finding_types": finding_types,
            "passed": len(finding_types) == 0,
        }))

    total = len(step_verdicts)
    passed = sum(1 for verdict in step_verdicts if verdict.passed)
    return AuditResult(
        verdict=verdict_for(findings),
        fidelity_score=passed / total if total > 0 else 1.0,
        findings=findings,
        step_verdicts=step_verdicts,
    )


def _visual_pair_action_state_compatible(
    step: ExpectedStep,
    action: ObservedAction,
) -> bool:
    return (
        actions_compatible(step.action, action.action)
        or _order_navigation_supports_open(step, action)
        or step.action in ASSERTION_ACTIONS
    )


def _apply_semantic_decisions(
    audit_result: AuditResult,
    decisions: list[tuple[_SemanticCandidate, _SemanticDecision]],
    *,
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
) -> AuditResult:
    """Apply confident target equivalence; assertions require direct visual support."""
    eligible = {
        _semantic_candidate_key(candidate)
        for candidate in select_semantic_candidates(
            expected_trace, observed_trace, audit_result
        )
    }
    accepted_targets: set[tuple[int, int]] = set()

    for candidate, decision in decisions:
        if _semantic_candidate_key(candidate) not in eligible:
            continue
        if (
            decision.answer != _SemanticAnswer.equivalent
            or decision.confidence < OVERTURN_CONFIDENCE
        ):
            continue
        if candidate.kind == _SemanticCandidateKind.target:
            accepted_targets.add(
                (candidate.expected_step_index, candidate.observed_action_index)
            )

    if not accepted_targets:
        return audit_result

    new_findings = []
    for finding in audit_result.findings:
        pair = (finding.expected_step_index, finding.observed_action_index)
        remove_target = (
            finding.type == FindingType.wrong_target
            and pair in accepted_targets
        )
        if not remove_target:
            new_findings.append(finding)

    return _rebuild_audit_result(audit_result, new_findings)


def select_crosscheck_targets(
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
    audit_result: AuditResult,
    *,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    window_pad_ms: int = WINDOW_PAD_MS,
) -> list[CrossCheckTarget]:
    """Select the assertion findings and low-confidence actions to route to the second model.

    No model calls are made here — this is the deterministic routing decision.
    """
    targets: list[CrossCheckTarget] = []
    step_by_index = {step.index: step for step in expected_trace.steps}
    protected_steps, protected_actions = _protected_contradiction_indices(
        expected_trace, observed_trace, audit_result
    )

    for finding in audit_result.findings:
        if finding.type != FindingType.assertion_not_observed:
            continue
        if finding.expected_step_index in protected_steps:
            continue
        step = step_by_index.get(finding.expected_step_index)
        assertion = step.assertion if step else None
        center = finding.evidence_timestamp_ms or 0
        targets.append(CrossCheckTarget(
            kind=CrossCheckKind.assertion,
            question=(
                f"Does text or a UI state matching {assertion!r} appear on screen in "
                "these frames?"
            ),
            window_start_ms=max(0, center - window_pad_ms),
            window_end_ms=center + window_pad_ms,
            anchor_timestamp_ms=center,
            expected_step_index=finding.expected_step_index,
            observed_action_index=finding.observed_action_index,
        ))

    already_targeted = {t.observed_action_index for t in targets}
    for action in observed_trace.actions:
        if action.confidence >= confidence_threshold:
            continue
        if action.index in protected_actions:
            continue
        if action.index in already_targeted:
            continue
        center = action.timestamp_ms
        targets.append(CrossCheckTarget(
            kind=CrossCheckKind.low_confidence,
            question=_describe_action_question(action.target, action.action.value),
            window_start_ms=max(0, center - window_pad_ms),
            window_end_ms=center + window_pad_ms,
            anchor_timestamp_ms=center,
            expected_step_index=None,
            observed_action_index=action.index,
        ))

    return targets


def _image_content(jpeg_bytes: bytes) -> dict:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def _ask_vision(
    openai_client: instructor.Instructor,
    question: str,
    jpeg_frames: list[bytes],
    model: str,
    on_retry: RetryCallback | None = None,
) -> _AssertionCheck:
    content: list[dict] = [{"type": "text", "text": question}]
    for index, frame in enumerate(jpeg_frames, start=1):
        content.append({"type": "text", "text": f"Frame {index}"})
        content.append(_image_content(frame))
    kwargs: dict = {
        "model": model,
        "response_model": _AssertionCheck,
        "messages": [
            {"role": "system", "content": _CROSSCHECK_SYSTEM},
            {"role": "user", "content": content},
        ],
    }
    if _TEMPERATURE is not None:
        kwargs["temperature"] = _TEMPERATURE
    return call_with_retry(
        lambda: openai_client.chat.completions.create(**kwargs),
        on_retry=on_retry,
    )


def verify_target(
    openai_client: instructor.Instructor,
    video_path: Path,
    target: CrossCheckTarget,
    *,
    model: str = CROSSCHECK_MODEL,
    frames_per_target: int = FRAMES_PER_TARGET,
    on_retry: RetryCallback | None = None,
) -> CrossCheckVerdict:
    """Pull the target's frames and put its grounded yes/no question to the second model."""
    samples = extract_evidence_frames_in_window(
        video_path, target.window_start_ms, target.window_end_ms, frames_per_target
    )
    if not samples:
        raise ValueError("No timestamped evidence frames available for cross-check")
    frame_timestamps = [ts for ts, _ in samples]
    jpeg_frames = [frame_to_jpeg_bytes(image) for _, image in samples]

    check = _ask_vision(
        openai_client, target.question, jpeg_frames, model, on_retry=on_retry
    )
    answer = check.answer
    if check.answer == CrossCheckAnswer.present:
        support_index = check.supporting_frame_index
        support_is_valid = (
            check.evidence_support == _EvidenceSupport.direct
            and support_index is not None
            and support_index <= len(frame_timestamps)
        )
        if support_is_valid and target.anchor_timestamp_ms is not None:
            support_is_valid = (
                frame_timestamps[support_index - 1]
                >= target.anchor_timestamp_ms
            )
        if not support_is_valid:
            answer = CrossCheckAnswer.unclear

    return CrossCheckVerdict(
        question=target.question,
        answer=answer,
        confidence=check.confidence,
        rationale=check.rationale,
        frame_timestamps_ms=frame_timestamps,
        expected_step_index=target.expected_step_index,
        observed_action_index=target.observed_action_index,
    )


def _apply_verdicts(
    audit_result: AuditResult,
    verdicts: list[CrossCheckVerdict],
    *,
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
) -> AuditResult:
    """Refine the audit result using the cross-check verdicts.

    A qualifying "present" can resolve assertion_not_observed for its expected step and
    wrong_target for its exact expected/observed pair. Other findings remain unchanged.
    Low-confidence verdicts are recorded for the operator but do not change the result.
    """
    assertion_steps = {
        f.expected_step_index
        for f in audit_result.findings
        if f.type == FindingType.assertion_not_observed
    }
    wrong_target_pairs = {
        (f.expected_step_index, f.observed_action_index)
        for f in audit_result.findings
        if f.type == FindingType.wrong_target
        and f.expected_step_index is not None
        and f.observed_action_index is not None
    }
    protected_steps, _ = _protected_contradiction_indices(
        expected_trace, observed_trace, audit_result
    )
    step_by_index = {step.index: step for step in expected_trace.steps}
    action_by_index = {action.index: action for action in observed_trace.actions}
    qualifying_verdicts = [
        verdict
        for verdict in verdicts
        if verdict.answer == CrossCheckAnswer.present
        and verdict.confidence >= OVERTURN_CONFIDENCE
        and verdict.expected_step_index not in protected_steps
        and verdict.expected_step_index in step_by_index
        and verdict.observed_action_index in action_by_index
        and _visual_pair_action_state_compatible(
            step_by_index[verdict.expected_step_index],
            action_by_index[verdict.observed_action_index],
        )
        and (
            verdict.expected_step_index in assertion_steps
            or (
                verdict.expected_step_index,
                verdict.observed_action_index,
            )
            in wrong_target_pairs
        )
    ]
    if not qualifying_verdicts:
        return audit_result

    overturned_assertion_steps = {
        verdict.expected_step_index for verdict in qualifying_verdicts
    }
    overturned_target_pairs = {
        (verdict.expected_step_index, verdict.observed_action_index)
        for verdict in qualifying_verdicts
    }
    new_findings = [
        f for f in audit_result.findings
        if not (
            (
                f.type == FindingType.assertion_not_observed
                and f.expected_step_index in overturned_assertion_steps
            )
            or (
                f.type == FindingType.wrong_target
                and (f.expected_step_index, f.observed_action_index)
                in overturned_target_pairs
            )
        )
    ]

    return _rebuild_audit_result(audit_result, new_findings)


def run_cross_check(
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
    audit_result: AuditResult,
    video_path: Path,
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = CROSSCHECK_MODEL,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    frames_per_target: int = FRAMES_PER_TARGET,
    window_pad_ms: int = WINDOW_PAD_MS,
    latency: LatencyMetrics | None = None,
) -> CrossCheckReport:
    """Run the full cross-check pass and return verdicts plus a refined audit result.

    Ambiguous wording in existing deterministic pairs is checked first. The visual
    cross-check then receives that refined result. Pass openai_client to inject a test
    double; otherwise a default instructor-patched OpenAI client is created.

    When latency is provided (argument or active latency_scope), records
    semantic/visual wall times, candidate/target counts, and retry backoff totals.
    Logs numeric fields only.
    """
    latency = resolve_latency(latency)
    retry_callback = bind_retry_callback(latency, None)

    semantic_candidates = select_semantic_candidates(
        expected_trace, observed_trace, audit_result
    )
    if latency is not None:
        latency.stage4_semantic_candidate_count = len(semantic_candidates)

    semantic_decisions: list[tuple[_SemanticCandidate, _SemanticDecision]] = []
    diagnostic_records: list[tuple[_SemanticCandidate, SemanticDiagnostic]] = []
    semantic_started = monotonic_ms()
    if semantic_candidates:
        if openai_client is None:
            openai_client = make_openai_client()
        for candidate in semantic_candidates:
            try:
                decision = verify_semantic_candidate(
                    openai_client,
                    candidate,
                    model=model,
                    on_retry=retry_callback,
                )
            except (
                openai.APIError,
                InstructorRetryException,
                ValidationError,
            ) as exc:
                diagnostic_records.append((candidate, SemanticDiagnostic(
                    expected_step_index=candidate.expected_step_index,
                    observed_action_index=candidate.observed_action_index,
                    comparison_kind=candidate.kind.value,
                    applied=False,
                    error_type=type(exc).__name__,
                )))
                continue
            semantic_decisions.append((candidate, decision))
            diagnostic_records.append((candidate, SemanticDiagnostic(
                expected_step_index=candidate.expected_step_index,
                observed_action_index=candidate.observed_action_index,
                comparison_kind=candidate.kind.value,
                answer=decision.answer.value,
                confidence=decision.confidence,
                applied=False,
            )))
    if latency is not None:
        latency.stage4_semantic_ms = elapsed_ms(semantic_started)
        logger.info(
            "stage4.semantic duration_ms=%.1f candidate_count=%d",
            latency.stage4_semantic_ms,
            latency.stage4_semantic_candidate_count,
        )

    semantic_result = _apply_semantic_decisions(
        audit_result,
        semantic_decisions,
        expected_trace=expected_trace,
        observed_trace=observed_trace,
    )
    semantic_diagnostics = [
        diagnostic.model_copy(update={
            "applied": (
                diagnostic.error_type is None
                and _semantic_finding_present(audit_result, candidate)
                and not _semantic_finding_present(semantic_result, candidate)
            ),
        })
        for candidate, diagnostic in diagnostic_records
    ]
    targets = select_crosscheck_targets(
        expected_trace,
        observed_trace,
        semantic_result,
        confidence_threshold=confidence_threshold,
        window_pad_ms=window_pad_ms,
    )
    if latency is not None:
        latency.stage4_visual_target_count = len(targets)

    if not targets:
        if latency is not None:
            latency.stage4_visual_ms = 0.0
            logger.info(
                "stage4.visual duration_ms=0.0 target_count=0",
            )
        return CrossCheckReport(
            verdicts=[],
            refined_result=semantic_result,
            semantic_diagnostics=semantic_diagnostics,
        )

    if openai_client is None:
        openai_client = make_openai_client()

    visual_started = monotonic_ms()
    verdicts = [
        verify_target(
            openai_client,
            video_path,
            target,
            model=model,
            frames_per_target=frames_per_target,
            on_retry=retry_callback,
        )
        for target in targets
    ]
    if latency is not None:
        latency.stage4_visual_ms = elapsed_ms(visual_started)
        logger.info(
            "stage4.visual duration_ms=%.1f target_count=%d retry_count=%d "
            "retry_sleep_s=%.3f",
            latency.stage4_visual_ms,
            latency.stage4_visual_target_count,
            latency.retry_count,
            latency.retry_sleep_s,
        )
    refined = _apply_verdicts(
        semantic_result,
        verdicts,
        expected_trace=expected_trace,
        observed_trace=observed_trace,
    )
    return CrossCheckReport(
        verdicts=verdicts,
        refined_result=refined,
        semantic_diagnostics=semantic_diagnostics,
    )
