import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _blank_str_to_none(value: object) -> object:
    """Treat blank/whitespace-only strings as absent.

    LLM extraction (Stages 1 and 2) commonly emits "" for optional fields it
    cannot fill rather than omitting them. Without this, an empty target, input
    value, or assertion would be compared downstream as a real (mismatching)
    value and produce spurious findings — including a fabricated
    assertion_not_observed on an otherwise faithful recording.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


class ActionType(str, Enum):
    click = "click"
    type_text = "type_text"
    select = "select"
    navigate = "navigate"
    scroll = "scroll"
    hover = "hover"
    assert_visible = "assert_visible"
    assert_text = "assert_text"
    assert_state = "assert_state"
    other = "other"


# ---------------------------------------------------------------------------
# Expected trace — parsed from the scenario / test case text
# ---------------------------------------------------------------------------

class ExpectedStep(BaseModel):
    index: int = Field(ge=1, description="1-based position in the scenario")
    action: ActionType
    target: Optional[str] = Field(
        None,
        description="UI element, page, or URL the action applies to",
    )
    input_value: Optional[str] = Field(
        None,
        description="Text typed, option selected, or value asserted",
    )
    assertion: Optional[str] = Field(
        None,
        description="What must be true after this step (visible text, state, URL, etc.)",
    )
    preconditions: Optional[str] = Field(
        None,
        description="Any state that must hold before this step executes",
    )
    raw_text: Optional[str] = Field(
        None,
        description="The original sentence or Gherkin line this step was parsed from",
    )

    _normalize_blanks = field_validator(
        "target", "input_value", "assertion", "preconditions", "raw_text",
        mode="before",
    )(_blank_str_to_none)


class ExpectedTrace(BaseModel):
    scenario_title: str
    steps: list[ExpectedStep]

    @model_validator(mode="after")
    def steps_must_be_sequential(self) -> "ExpectedTrace":
        for i, step in enumerate(self.steps, start=1):
            if step.index != i:
                raise ValueError(
                    f"steps must be in order: expected index {i}, got {step.index}"
                )
        return self


# ---------------------------------------------------------------------------
# Observed trace — extracted from the video by the model (Stage 2)
# ---------------------------------------------------------------------------

class AssertionEvidence(BaseModel):
    description: str = Field(
        description="What was observed on screen (e.g. 'Error message: Invalid email')"
    )
    visible_text: Optional[str] = Field(
        None,
        description="Exact text read from the screen, if legible",
    )

    _normalize_blanks = field_validator("visible_text", mode="before")(_blank_str_to_none)


class ObservedAction(BaseModel):
    index: int = Field(ge=1, description="1-based order of this action in the recording")
    timestamp_ms: int = Field(ge=0, description="Millisecond offset from clip start")
    action: ActionType
    target: Optional[str] = Field(
        None,
        description="UI element, page, or URL acted upon",
    )
    input_value: Optional[str] = Field(
        None,
        description="Text entered or value observed",
    )
    assertion_evidence: Optional[AssertionEvidence] = Field(
        None,
        description="Assertion-relevant evidence observed at this step",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in this observation (0–1); low values are routed to cross-check",
    )
    screen_description: Optional[str] = Field(
        None,
        description="Brief description of the screen state at this moment",
    )

    _normalize_blanks = field_validator(
        "target", "input_value", "screen_description", mode="before",
    )(_blank_str_to_none)

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_nan(cls, v: object) -> object:
        if isinstance(v, float) and math.isnan(v):
            raise ValueError("confidence must be a finite number, got NaN")
        return v


class ObservedTrace(BaseModel):
    source_file: str = Field(description="Filename or File API name of the source recording")
    clip_offset_ms: int = Field(
        default=0,
        ge=0,
        description="Start offset of this clip within the full recording, in milliseconds",
    )
    clip_end_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="End offset of this clip within the full recording, in milliseconds. None means the clip runs to the end of the recording.",
    )
    actions: list[ObservedAction]

    @model_validator(mode="after")
    def clip_end_must_be_after_offset(self) -> "ObservedTrace":
        if self.clip_end_ms is not None and self.clip_end_ms <= self.clip_offset_ms:
            raise ValueError(
                f"clip_end_ms ({self.clip_end_ms}) must be greater than clip_offset_ms ({self.clip_offset_ms})"
            )
        return self


# ---------------------------------------------------------------------------
# Audit result — produced by Stage 3 (the deterministic aligner)
# ---------------------------------------------------------------------------

class FindingType(str, Enum):
    missing_action = "missing_action"
    extra_action = "extra_action"
    wrong_target = "wrong_target"
    wrong_input_value = "wrong_input_value"
    assertion_not_observed = "assertion_not_observed"


class FindingSeverity(str, Enum):
    """Whether a finding bears on fidelity or only on wording and grain.

    critical — the recording does not demonstrate something the scenario claimed.
    advisory — a real difference, but one that recurs on recordings which are faithful,
               because Stage 2 describes a recording at a finer grain than anyone writes
               a test at. Reported, never fatal.
    """

    critical = "critical"
    advisory = "advisory"


class AuditVerdict(str, Enum):
    faithful = "faithful"
    unfaithful = "unfaithful"


class Finding(BaseModel):
    type: FindingType
    severity: FindingSeverity = Field(
        default=FindingSeverity.advisory,
        description="Whether this finding bears on fidelity; set by the aligner",
    )
    expected_step_index: Optional[int] = Field(
        None,
        description="1-based index of the expected step involved (None for extra_action findings)",
    )
    observed_action_index: Optional[int] = Field(
        None,
        description="1-based index of the observed action involved (None for missing_action findings)",
    )
    evidence_timestamp_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Absolute millisecond offset in the recording that supports this finding",
    )
    description: str = Field(description="Human-readable explanation of the finding")


class StepVerdict(BaseModel):
    expected_step_index: int = Field(ge=1)
    passed: bool
    matched_observed_index: Optional[int] = Field(
        None,
        description="1-based index of the observed action this step was matched to",
    )
    evidence_timestamp_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Absolute millisecond offset in the recording for this step's match",
    )
    finding_types: list[FindingType] = Field(
        default_factory=list,
        description="Finding types raised for this step; empty if passed",
    )


class AuditResult(BaseModel):
    verdict: AuditVerdict = Field(
        default=AuditVerdict.faithful,
        description=(
            "The headline result: unfaithful when any critical finding is present. This, "
            "not fidelity_score, is what a consumer should act on"
        ),
    )
    fidelity_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of expected steps that passed. A secondary detail, not a verdict — "
            "it moves with how finely the model happened to describe the recording"
        ),
    )
    findings: list[Finding]
    step_verdicts: list[StepVerdict]

    @property
    def critical_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == FindingSeverity.critical]


# ---------------------------------------------------------------------------
# Cross-check pass — second-model assertion verification (Stage 2.5)
# ---------------------------------------------------------------------------

class CrossCheckAnswer(str, Enum):
    present = "present"
    absent = "absent"
    unclear = "unclear"


class CrossCheckVerdict(BaseModel):
    """A second model's grounded yes/no judgement on a single assertion or low-confidence step.

    The cross-check is framed as "does text/state matching X appear in these frames?"
    rather than a holistic match, because that is where false passes hide. answer,
    confidence, and rationale come from the model; question, the frame timestamps, and
    the step/action indices are set in code so the evidence pointer stays authoritative.
    """

    question: str = Field(description="The grounded yes/no question put to the cross-check model")
    answer: CrossCheckAnswer
    confidence: float = Field(ge=0.0, le=1.0, description="Cross-check model confidence (0–1)")
    rationale: str = Field(description="The model's short justification for its answer")
    frame_timestamps_ms: list[int] = Field(
        default_factory=list,
        description="Absolute millisecond offsets of the frames the model was shown",
    )
    expected_step_index: Optional[int] = Field(
        None,
        description="1-based expected step this check relates to, if any",
    )
    observed_action_index: Optional[int] = Field(
        None,
        description="1-based observed action this check relates to, if any",
    )


class CrossCheckKind(str, Enum):
    assertion = "assertion"
    low_confidence = "low_confidence"


class CrossCheckTarget(BaseModel):
    """A single thing the cross-check model should re-examine, with the frame window.

    Assertion targets re-examine an assertion the deterministic aligner could not
    confirm (the high-stakes false-pass case); low_confidence targets re-examine an
    observed action the extractor itself was unsure about, before its match is
    published.
    """

    kind: CrossCheckKind
    question: str = Field(description="The grounded yes/no question to put to the model")
    window_start_ms: int = Field(ge=0, description="Absolute start of the frame window")
    window_end_ms: int = Field(ge=0, description="Absolute end of the frame window")
    anchor_timestamp_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Earliest timestamp relevant to the UI state being verified",
    )
    expected_step_index: Optional[int] = None
    observed_action_index: Optional[int] = None

    @model_validator(mode="after")
    def window_must_be_ordered(self) -> "CrossCheckTarget":
        if self.window_end_ms <= self.window_start_ms:
            raise ValueError(
                f"window_end_ms ({self.window_end_ms}) must be greater than "
                f"window_start_ms ({self.window_start_ms})"
            )
        if (
            self.anchor_timestamp_ms is not None
            and not self.window_start_ms
            <= self.anchor_timestamp_ms
            <= self.window_end_ms
        ):
            raise ValueError(
                "anchor_timestamp_ms must fall within the cross-check window"
            )
        return self


class SemanticDiagnostic(BaseModel):
    """Non-sensitive outcome metadata for one semantic verification candidate."""

    expected_step_index: int = Field(ge=1)
    observed_action_index: int = Field(ge=1)
    comparison_kind: str
    answer: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    applied: bool = False
    error_type: Optional[str] = None


class CrossCheckReport(BaseModel):
    """The output of the cross-check pass: the per-target verdicts and the refined audit."""

    verdicts: list[CrossCheckVerdict]
    refined_result: AuditResult
    semantic_diagnostics: list[SemanticDiagnostic] = Field(default_factory=list)


def merge_observed_chunks(chunks: list[ObservedTrace]) -> ObservedTrace:
    """Merge ordered chunks of the same recording into a single ObservedTrace.

    Each action's timestamp_ms is converted from clip-relative to recording-absolute
    by adding the chunk's clip_offset_ms. Actions are reindexed sequentially.
    Chunks must all refer to the same source_file and be passed in time order.
    """
    if not chunks:
        raise ValueError("at least one chunk is required")

    source_files = {c.source_file for c in chunks}
    if len(source_files) > 1:
        raise ValueError(f"all chunks must share the same source_file, got: {source_files}")

    sorted_chunks = sorted(chunks, key=lambda c: c.clip_offset_ms)

    absolute_actions: list[ObservedAction] = []
    for chunk in sorted_chunks:
        for action in chunk.actions:
            absolute_actions.append(
                action.model_copy(
                    update={"timestamp_ms": action.timestamp_ms + chunk.clip_offset_ms}
                )
            )

    reindexed = [
        a.model_copy(update={"index": i})
        for i, a in enumerate(absolute_actions, start=1)
    ]

    last_chunk = sorted_chunks[-1]
    return ObservedTrace(
        source_file=sorted_chunks[0].source_file,
        clip_offset_ms=0,
        clip_end_ms=last_chunk.clip_end_ms,
        actions=reindexed,
    )
