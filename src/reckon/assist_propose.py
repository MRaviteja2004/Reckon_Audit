"""Batched Claude proposals for Stage 3 assist candidates."""

import os

import instructor
from pydantic import BaseModel, ConfigDict, Field

from reckon.assist_select import AssistCandidate
from reckon.parser import make_openai_client
from reckon.retry import RetryCallback, call_with_retry
from reckon.schema import AssistCandidateKind

ALIGN_ASSIST_MODEL = os.environ.get("ALIGN_ASSIST_MODEL", "claude-opus-5-5")
ASSIST_CONFIDENCE_FLOOR = 0.7

# temperature omitted: some gateway models reject any explicit value.
_TEMPERATURE: float | None = None

_ASSIST_SYSTEM = """\
You assist a deterministic UI-test aligner. You receive already-matched expected steps \
and candidate observed text. You never invent events, timestamps, or typed values.

For wrong_target: answer "equivalent" only when expected and observed name the same UI \
control or place despite wording differences. Answer "different" when they are distinct. \
Answer "unresolved" when unsure.

For delayed_assertion: decide whether any listed later observation supports the expected \
assertion text. Answer "equivalent" only with a supporting_event_id from the list. Answer \
"unsupported" when none support it. Answer "unresolved" when unsure. Never treat a click, \
spinner, or intermediate state as proof of a successful final outcome. Never treat \
negative evidence (failure, error opposite the claim) as support.

Return one decision per candidate_id. Do not invent candidate or event ids.\
"""


class _AssistDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    supporting_event_id: str | None = None


class _AssistBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[_AssistDecision]


def _format_delayed_events(candidate: AssistCandidate) -> str:
    lines: list[str] = []
    for event in candidate.delayed_events:
        event_id = f"O{event.observed_index}"
        lines.append(
            f"- {event_id}: action={event.action} target={event.target!r} "
            f"visible_text={event.visible_text!r} "
            f"evidence_description={event.evidence_description!r} "
            f"screen_description={event.screen_description!r}"
        )
    return "\n".join(lines)


def _format_candidate(candidate: AssistCandidate) -> str:
    block = (
        f"candidate_id: {candidate.candidate_id}\n"
        f"kind: {candidate.kind.value}\n"
        f"expected_action: {candidate.expected_action}\n"
        f"observed_action: {candidate.observed_action}\n"
        f"expected_text: {candidate.expected_text!r}\n"
        f"observed_text: {candidate.observed_text!r}"
    )
    if candidate.kind == AssistCandidateKind.delayed_assertion:
        block += (
            f"\nmatched_visible_text: {candidate.observed_visible_text!r}"
            f"\nmatched_evidence_description: {candidate.observed_evidence_description!r}"
            f"\nmatched_screen_description: {candidate.observed_screen_description!r}"
            f"\nlater_observations:\n{_format_delayed_events(candidate)}"
        )
    return block


def build_assist_user_content(candidates: list[AssistCandidate]) -> str:
    """Build the batched user message for eligible LLM candidates only."""
    parts = ["Review these alignment candidates:\n"]
    for candidate in candidates:
        parts.append(_format_candidate(candidate))
        parts.append("")
    return "\n".join(parts).strip()


def propose_assist_batch(
    candidates: list[AssistCandidate],
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = ALIGN_ASSIST_MODEL,
    on_retry: RetryCallback | None = None,
) -> _AssistBatchResponse:
    """Ask Claude once for all LLM-eligible candidates. Caller must not pass empty list."""
    if not candidates:
        raise ValueError("propose_assist_batch requires at least one candidate")
    if openai_client is None:
        openai_client = make_openai_client()

    kwargs: dict = {
        "model": model,
        "response_model": _AssistBatchResponse,
        "messages": [
            {"role": "system", "content": _ASSIST_SYSTEM},
            {"role": "user", "content": build_assist_user_content(candidates)},
        ],
    }
    if _TEMPERATURE is not None:
        kwargs["temperature"] = _TEMPERATURE

    response = call_with_retry(
        lambda: openai_client.chat.completions.create(**kwargs),
        on_retry=on_retry,
    )
    return _AssistBatchResponse.model_validate(response)
