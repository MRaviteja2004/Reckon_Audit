"""Claude role-label proposals for experimental evidence-aware Stage 3 assist."""

import os

import instructor
from pydantic import BaseModel, ConfigDict, Field

from reckon.assist_evidence_select import EvidenceBundleCandidate
from reckon.parser import make_openai_client
from reckon.retry import RetryCallback, call_with_retry
from reckon.schema import EvidenceRole

ALIGN_EVIDENCE_MODEL = os.environ.get(
    "ALIGN_EVIDENCE_MODEL",
    os.environ.get("ALIGN_ASSIST_MODEL", "claude-opus-5-5"),
)
EVIDENCE_CONFIDENCE_FLOOR = 0.7

_TEMPERATURE: float | None = None

_EVIDENCE_SYSTEM = """\
You assist a deterministic UI-test aligner with coarse expected steps that may \
map to several fine-grained observations. You never invent events, timestamps, \
or typed values. You never declare the audit faithful or unfaithful.

For each candidate, assign every listed observation a role:
- initiation: workflow start / entry (e.g. opening an app)
- progress: intermediate steps (forms, continue, spinners)
- completion: successful outcome state
- assertion_evidence: observation whose visible text supports the expected assertion
- distractor: similar tokens but wrong meaning (spinner ≠ app, settings ≠ logged-in home)

Rules:
- Initiation alone never proves successful completion or an assertion.
- Never treat a click on an app icon, a spinner, or splash as proof of login success.
- Never treat negative evidence (failure, error opposite the claim) as assertion_evidence.
- Prefer distractor when unsure whether an observation truly supports the claim.
- best_* ids must be from the listed observations or null.
- Return one decision per candidate_id.
"""


class _EvidenceRoleAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_event_id: str = Field(min_length=2)
    role: EvidenceRole


class _EvidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    labels: list[_EvidenceRoleAssignment] = Field(default_factory=list)
    best_initiation_id: str | None = None
    best_completion_id: str | None = None
    best_assertion_support_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class _EvidenceBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[_EvidenceDecision]


def _format_event(event) -> str:
    return (
        f"- O{event.observed_index}: action={event.action} target={event.target!r} "
        f"ts_ms={event.timestamp_ms} "
        f"visible_text={event.visible_text!r} "
        f"evidence_description={event.evidence_description!r} "
        f"screen_description={event.screen_description!r}"
    )


def build_evidence_user_content(candidates: list[EvidenceBundleCandidate]) -> str:
    """Build the batched user message for evidence-bundle candidates."""
    parts = ["Label observations for these coarse expected steps:\n"]
    for candidate in candidates:
        parts.append(f"candidate_id: {candidate.candidate_id}")
        parts.append(f"expected_step_index: {candidate.expected_step_index}")
        parts.append(f"expected_action: {candidate.expected_action}")
        parts.append(f"expected_text: {candidate.expected_text!r}")
        parts.append(f"expected_assertion: {candidate.expected_assertion!r}")
        parts.append(
            f"matched_observed_index: {candidate.matched_observed_index}"
        )
        parts.append(f"suspicious_pass: {candidate.suspicious_pass}")
        parts.append("observations:")
        for event in candidate.evidence_events:
            parts.append(_format_event(event))
        parts.append("")
    return "\n".join(parts).strip()


def propose_evidence_batch(
    candidates: list[EvidenceBundleCandidate],
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = ALIGN_EVIDENCE_MODEL,
    on_retry: RetryCallback | None = None,
) -> _EvidenceBatchResponse:
    """Ask Claude once for role labels on all evidence-bundle candidates."""
    if not candidates:
        raise ValueError("propose_evidence_batch requires at least one candidate")
    if openai_client is None:
        openai_client = make_openai_client()

    kwargs: dict = {
        "model": model,
        "response_model": _EvidenceBatchResponse,
        "messages": [
            {"role": "system", "content": _EVIDENCE_SYSTEM},
            {"role": "user", "content": build_evidence_user_content(candidates)},
        ],
    }
    if _TEMPERATURE is not None:
        kwargs["temperature"] = _TEMPERATURE

    return call_with_retry(
        lambda: openai_client.chat.completions.create(**kwargs),
        on_retry=on_retry,
    )
