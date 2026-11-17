"""Stage 2C LLM proposals: frame IDs only — never timestamps or verdicts."""

import os

import instructor
from pydantic import BaseModel, ConfigDict, Field

from reckon.parser import make_openai_client
from reckon.retry import RetryCallback, call_with_retry
from reckon.schema import CandidateEvidenceType, CandidateSearchStatus, FrameManifest
from reckon.stage2c.select import Stage2CSearchTarget

STAGE2C_MODEL = os.environ.get(
    "STAGE2C_MODEL",
    os.environ.get("ALIGN_ASSIST_MODEL", "claude-opus-5-5"),
)

_SYSTEM = """\
You search labelled UI-test recording frames for visual evidence that might support \
specific expected steps. You never invent frame labels, timestamps, or audit verdicts.

For each expected step, return frame_ids from the supplied labels only (e.g. F12).
Set evidence_type to ACTION for user actions (click, type, navigate) and \
ASSERTION_OUTCOME for resulting screen state or visible text.
Set search_status to candidate_found, not_found, or ambiguous.
Do not declare the test passed, faithful, or failed.
Do not clear findings. Confidence reflects visual certainty only.
"""


class RawFrameEvidenceCandidate(BaseModel):
    """LLM-facing candidate — timestamps are forbidden here."""

    model_config = ConfigDict(extra="forbid")

    expected_step_id: str = Field(min_length=1)
    frame_ids: list[str] = Field(default_factory=list)
    evidence_type: CandidateEvidenceType
    observed_description: str = Field(default="")
    visible_text: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    search_status: CandidateSearchStatus


class RawFrameEvidenceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[RawFrameEvidenceCandidate]


def build_stage2c_user_content(
    targets: list[Stage2CSearchTarget],
    manifest: FrameManifest,
    *,
    observed_context: str | None = None,
) -> str:
    """Build the text portion of the Stage 2C user message (frames attached separately)."""
    parts = [
        "Search these expected steps in the labelled frames that follow.",
        "Return frame_ids from the labels only. Never invent timestamps.",
        "",
        "Available frame labels:",
    ]
    for entry in manifest.frames:
        parts.append(f"- {entry.frame_id}")
    if observed_context:
        parts.append("")
        parts.append("Optional ObservedTrace context (may be incomplete):")
        parts.append(observed_context)
    parts.append("")
    parts.append("Expected steps to search:")
    for target in targets:
        parts.append(f"expected_step_id: {target.expected_step_id}")
        parts.append(f"preferred_evidence_type: {target.evidence_type.value}")
        parts.append(f"expected_action: {target.expected_action}")
        parts.append(f"expected_text: {target.expected_text!r}")
        parts.append(f"expected_assertion: {target.expected_assertion!r}")
        parts.append(
            "finding_types: "
            + ", ".join(ft.value for ft in target.finding_types)
        )
        parts.append("")
    return "\n".join(parts).strip()


def propose_frame_evidence_batch(
    targets: list[Stage2CSearchTarget],
    manifest: FrameManifest,
    frame_content: list[dict],
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = STAGE2C_MODEL,
    on_retry: RetryCallback | None = None,
    observed_context: str | None = None,
) -> RawFrameEvidenceBatch:
    """Ask the vision model for frame-ID candidates. Timestamps stay in Python."""
    if openai_client is None:
        openai_client = make_openai_client()
    text = build_stage2c_user_content(
        targets, manifest, observed_context=observed_context
    )
    content: list[dict] = [{"type": "text", "text": text}, *frame_content]
    return call_with_retry(
        lambda: openai_client.chat.completions.create(
            model=model,
            response_model=RawFrameEvidenceBatch,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": content},
            ],
        ),
        on_retry=on_retry,
    )
