import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, NonNegativeInt, field_validator, model_validator


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

class StepSource(str, Enum):
    """Where an expected step came from in Stage 1 modeling."""

    scenario = "scenario"
    macro = "macro"


class StepObligation(str, Enum):
    """Whether a macro-derived step is required or conditional."""

    mandatory = "mandatory"
    conditional = "conditional"
    explicit = "explicit"


class MacroMemberDecision(BaseModel):
    """One expansion decision recorded for provenance / audits."""

    member_id: str = Field(description="Registry member slot identifier")
    decision: str = Field(
        description=(
            "Outcome such as emitted, annotated_existing, "
            "skipped_explicit_cover, suppressed_by_negation, "
            "not_expanded_insufficient_detail"
        )
    )
    detail: Optional[str] = Field(
        None,
        description="Optional human-readable reason for the decision",
    )


class MacroApplicationRecord(BaseModel):
    """Provenance for one Domain Macro application on an ExpectedTrace."""

    macro_id: str
    version: str
    domain: str
    source_text: Optional[str] = Field(
        None,
        description="Scenario span / raw_text that triggered the macro",
    )
    decisions: list[MacroMemberDecision] = Field(default_factory=list)


class IntentOperation(str, Enum):
    """UI operation on a structured intent. Not an English verb."""

    activate = "activate"
    enter = "enter"
    choose = "choose"
    traverse = "traverse"
    dismiss = "dismiss"
    assert_ = "assert"
    unspecified = "unspecified"


class ObjectKind(str, Enum):
    application = "application"
    control = "control"
    field = "field"
    option = "option"
    page = "page"
    dialog = "dialog"
    text = "text"
    button = "button"
    link = "link"
    icon = "icon"
    menu_item = "menu_item"
    unspecified = "unspecified"


class ElementKind(str, Enum):
    icon = "icon"
    button = "button"
    link = "link"
    menu_item = "menu_item"
    field = "field"
    option = "option"
    page = "page"
    dialog = "dialog"
    text = "text"
    control = "control"
    application = "application"
    unspecified = "unspecified"


class CitationSurface(str, Enum):
    target = "target"
    input_value = "input_value"
    visible_text = "visible_text"


class SlotCitation(BaseModel):
    """Exact slice of an original source string. ``end`` is exclusive."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def end_is_exclusive(self) -> "SlotCitation":
        if self.end <= self.start:
            raise ValueError("citation end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("citation text length must equal end - start")
        return self


class ExpectedObject(BaseModel):
    kind: ObjectKind = ObjectKind.unspecified
    kind_citation: Optional[SlotCitation] = None
    name: Optional[str] = None
    name_citation: Optional[SlotCitation] = None
    product: Optional[str] = None
    product_citation: Optional[SlotCitation] = None
    instance: Optional[str] = None
    instance_citation: Optional[SlotCitation] = None
    value: Optional[str] = None
    value_citation: Optional[SlotCitation] = None


class ExpectedActionIntent(BaseModel):
    """Structured expected intent. Offsets refer to the original scenario text."""

    schema_version: str = "1"
    operation: IntentOperation = IntentOperation.unspecified
    operation_citation: Optional[SlotCitation] = None
    object: ExpectedObject = Field(default_factory=ExpectedObject)
    sentence_start: int = Field(ge=0)
    sentence_end: int = Field(ge=0)

    @model_validator(mode="after")
    def sentence_bounds(self) -> "ExpectedActionIntent":
        if self.sentence_end <= self.sentence_start:
            raise ValueError("sentence_end must be greater than sentence_start")
        return self


class IntentParseStatus(str, Enum):
    bound = "bound"
    omitted = "omitted"
    rejected = "rejected"


class IntentRejectionReason(str, Enum):
    """Why application-side validation refused an intent the model did offer."""

    missing_field = "missing_field"
    unsupported_operation = "unsupported_operation"
    unsupported_kind = "unsupported_kind"
    step_span_unavailable = "step_span_unavailable"
    invalid_citation = "invalid_citation"
    ambiguous_citation = "ambiguous_citation"
    binding_failed = "binding_failed"


class IntentDiagnostic(BaseModel):
    """Outcome of binding one step's intent. Holds no scenario text, only vocabulary tokens."""

    step_index: int = Field(ge=1)
    status: IntentParseStatus
    reason: IntentRejectionReason | None = Field(
        default=None,
        description="Set exactly when status is rejected. An omitted intent has no reason.",
    )
    slot: str | None = Field(
        default=None,
        description="Slot that failed: operation, operation_text, kind, kind_text, name, ...",
    )
    offered_operation: str | None = Field(
        default=None, description="Normalised operation token the model offered, if sane"
    )
    offered_kind: str | None = Field(
        default=None, description="Normalised kind token the model offered, if sane"
    )

    @model_validator(mode="after")
    def reason_matches_status(self) -> "IntentDiagnostic":
        if (self.status == IntentParseStatus.rejected) != (self.reason is not None):
            raise ValueError("reason is required when rejected and forbidden otherwise")
        return self


class IntentParseReport(BaseModel):
    """Stage 1 intent outcomes, one per scenario step the parser bound or refused."""

    contract_version: str = Field(min_length=1)
    diagnostics: list[IntentDiagnostic] = Field(default_factory=list)


class ObservedElement(BaseModel):
    kind: ElementKind = ElementKind.unspecified
    kind_citation: Optional[SlotCitation] = None
    name: Optional[str] = None
    name_citation: Optional[SlotCitation] = None
    product: Optional[str] = None
    product_citation: Optional[SlotCitation] = None
    instance: Optional[str] = None
    instance_citation: Optional[SlotCitation] = None
    value: Optional[str] = None
    value_citation: Optional[SlotCitation] = None


class ObservedInteraction(BaseModel):
    """Structured observation. Citations point into one stored surface, not screen_description."""

    schema_version: str = "1"
    operation: IntentOperation = IntentOperation.unspecified
    operation_citation: Optional[SlotCitation] = None
    element: ObservedElement = Field(default_factory=ObservedElement)
    citation_surface: CitationSurface
    cited_frame_index: int = Field(ge=1)
    context_episode_id: Optional[int] = Field(
        default=None,
        description="Screen episode for review. Not a control identity.",
    )


class InteractionParseStatus(str, Enum):
    bound = "bound"
    omitted = "omitted"
    rejected = "rejected"


class InteractionRejectionReason(str, Enum):
    """Why application-side validation refused an interaction the model did report."""

    missing_field = "missing_field"
    unsupported_operation = "unsupported_operation"
    unsupported_kind = "unsupported_kind"
    unsupported_surface = "unsupported_surface"
    surface_unavailable = "surface_unavailable"
    operation_contradicts_action = "operation_contradicts_action"
    invalid_citation = "invalid_citation"
    ambiguous_citation = "ambiguous_citation"
    binding_failed = "binding_failed"


class ObservedInteractionDiagnostic(BaseModel):
    """Outcome of binding one observation's interaction. Holds no screen text."""

    contract_version: str = Field(min_length=1)
    status: InteractionParseStatus
    reason: InteractionRejectionReason | None = Field(
        default=None,
        description="Set exactly when status is rejected. An omitted interaction has no reason.",
    )
    slot: str | None = Field(
        default=None, description="Slot that failed: operation, kind, kind_text, name, ..."
    )
    offered_operation: str | None = Field(
        default=None, description="Normalised operation token the model offered, if sane"
    )
    offered_kind: str | None = Field(
        default=None, description="Normalised kind token the model offered, if sane"
    )
    offered_surface: str | None = Field(
        default=None, description="Normalised surface token the model offered, if sane"
    )

    @model_validator(mode="after")
    def reason_matches_status(self) -> "ObservedInteractionDiagnostic":
        if (self.status == InteractionParseStatus.rejected) != (self.reason is not None):
            raise ValueError("reason is required when rejected and forbidden otherwise")
        return self


class EquivalenceMatch(BaseModel):
    """One accepted action realization. The observation id and timestamp are the proof."""

    relation_id: str
    expected_step_index: int = Field(ge=1)
    terminal_observed_index: int = Field(ge=1)
    evidence_timestamp_ms: int = Field(ge=0)
    evidence_id: str
    cited_frame_index: int = Field(ge=1)
    evidence_kind: str = "action"
    compared_slots: list[str] = Field(default_factory=list)
    context_episode_id: Optional[int] = None


class EquivalenceDecision(BaseModel):
    """One Base assignment outcome. Agent fields are filled only after Base commits."""

    expected_step_index: int = Field(ge=1)
    status: str
    observed_index: Optional[int] = None
    relation_id: Optional[str] = None
    agent_status: Optional[str] = None
    agent_observed_index: Optional[int] = None
    agent_rejection: Optional[str] = None


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
    source: StepSource = Field(
        default=StepSource.scenario,
        description="Whether this step came from scenario text or macro expansion",
    )
    macro_id: Optional[str] = Field(
        None,
        description="Domain macro ID when source is macro",
    )
    macro_version: Optional[str] = Field(
        None,
        description="Domain macro version when source is macro",
    )
    obligation: Optional[StepObligation] = Field(
        None,
        description="mandatory / conditional / explicit when macro-derived",
    )
    action_intent: Optional[ExpectedActionIntent] = Field(
        default=None,
        description="Structured intent with citations into the original scenario. Unset on legacy traces.",
    )

    _normalize_blanks = field_validator(
        "target", "input_value", "assertion", "preconditions", "raw_text",
        "macro_id", "macro_version",
        mode="before",
    )(_blank_str_to_none)


class ExpectedTrace(BaseModel):
    scenario_title: str
    steps: list[ExpectedStep]
    source_text: Optional[str] = Field(
        default=None,
        description="Original scenario input. Citation offsets refer to this string.",
    )
    macros_applied: list[MacroApplicationRecord] = Field(
        default_factory=list,
        description="Domain macros applied during Stage 1 (opt-in provenance)",
    )
    intent_report: IntentParseReport | None = Field(
        default=None,
        description="Why each step's intent was bound, omitted or rejected. Unset on legacy traces.",
    )

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
    interaction: Optional[ObservedInteraction] = Field(
        default=None,
        description="Structured interaction with citations into target, input_value, or visible_text.",
    )
    interaction_diagnostic: ObservedInteractionDiagnostic | None = Field(
        default=None,
        description="Why the interaction was bound, omitted or rejected. Unset on legacy traces.",
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
    gap_sample_timestamps_ms: list[NonNegativeInt] = Field(
        default_factory=list,
        description=(
            "Recording-absolute times of frames sampled to cover quiet stretches. "
            "Visual evidence for Stage 2C to search, never an observation."
        ),
    )

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
    cleared_by_assist: bool = Field(
        default=False,
        description=(
            "True when Stage 3 assist cleared findings on this step. "
            "Stage 4 may still visually review low-confidence clears."
        ),
    )
    assist_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM confidence for the assist clear, when applicable",
    )
    recovery_reason: Optional[str] = Field(
        default=None,
        description=(
            "How this step's evidence was selected "
            "(e.g. later_observed_action, frame_evidence)"
        ),
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="Id of the evidence this verdict uses, such as O3 or F4,F5",
    )
    frame_ids: list[str] = Field(
        default_factory=list,
        description="Frame ids when the evidence is frame text",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Why this evidence was selected. It refers to evidence_id and evidence_timestamp_ms.",
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
    equivalence_decisions: list[EquivalenceDecision] = Field(
        default_factory=list,
        description="Base equivalence assignment. Empty when no structured intent was present.",
    )

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
    supporting_frame_timestamp_ms: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Absolute millisecond offset of the frame that directly supported a "
            "present answer, when the model named one"
        ),
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


# ---------------------------------------------------------------------------
# Stage 3 episodes — temporal clusters of observed micro-actions
# ---------------------------------------------------------------------------

class ScreenEpisode(BaseModel):
    """A contiguous screen state spanning one or more ObservedAction items."""

    episode_id: int = Field(ge=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    start_index: int = Field(ge=1)
    end_index: int = Field(ge=1)
    screen_summary: str = Field(
        description="Representative screen/target description for the episode",
    )
    visible_texts: list[str] = Field(default_factory=list)
    actions_summary: list[str] = Field(
        default_factory=list,
        description="Short action labels (action on target) inside the episode",
    )
    is_ambient_system: bool = Field(
        default=False,
        description=(
            "True when the episode looks like an OS-level interaction "
            "(notification shade, lock screen, system UI)"
        ),
    )

    @model_validator(mode="after")
    def bounds_must_be_ordered(self) -> "ScreenEpisode":
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be >= start_ms ({self.start_ms})"
            )
        if self.end_index < self.start_index:
            raise ValueError(
                f"end_index ({self.end_index}) must be >= start_index "
                f"({self.start_index})"
            )
        return self


# ---------------------------------------------------------------------------
# Stage 3 LLM assist — optional semantic matching on top of deterministic align()
# ---------------------------------------------------------------------------

class AssistCandidateKind(str, Enum):
    """Kinds handled by the Stage 3 assistant.

    evidence_bundle is the experimental many-observation support path. It does not
    change deterministic align(); it only runs when callers opt into evidence-aware
    assist.

    episode_verification remaps a DP-anchored (or unmatched) assertion step onto
    the terminal ObservedAction inside a proving ScreenEpisode.

    frame_evidence is Stage 2C expected-guided frame evidence accepted only after
    Python validation against the frame manifest.
    """

    wrong_target = "wrong_target"
    delayed_assertion = "delayed_assertion"
    evidence_bundle = "evidence_bundle"
    episode_verification = "episode_verification"
    frame_evidence = "frame_evidence"


class CandidateEvidenceType(str, Enum):
    """What a Stage 2C candidate claims to show in frames."""

    ACTION = "ACTION"
    ASSERTION_OUTCOME = "ASSERTION_OUTCOME"


class CandidateSearchStatus(str, Enum):
    """LLM search outcome for one expected step — never a pass/fail verdict."""

    candidate_found = "candidate_found"
    not_found = "not_found"
    ambiguous = "ambiguous"


class CandidateValidationStatus(str, Enum):
    """Python validation outcome for a Stage 2C candidate."""

    accepted = "accepted"
    rejected = "rejected"


class FrameManifestEntry(BaseModel):
    """One labelled frame and its authoritative decoder timestamp."""

    frame_id: str = Field(min_length=1, description="Label such as F1, F2")
    timestamp_ms: int = Field(ge=0)


class FrameManifest(BaseModel):
    """Authoritative frame_id → timestamp map. LLM never invents times."""

    frames: list[FrameManifestEntry] = Field(min_length=1)

    def timestamp_for(self, frame_id: str) -> int | None:
        key = frame_id.strip().upper()
        for entry in self.frames:
            if entry.frame_id.strip().upper() == key:
                return entry.timestamp_ms
        return None


class CandidateEvidenceItem(BaseModel):
    """One expected-guided evidence candidate after Python materialization/validation."""

    expected_step_id: str = Field(min_length=1, description="Stable id such as E5")
    expected_step_index: int = Field(ge=1)
    frame_ids: list[str] = Field(default_factory=list)
    timestamp_start_ms: int | None = Field(
        default=None,
        ge=0,
        description="Python-derived from frame manifest; None when frames invalid",
    )
    timestamp_end_ms: int | None = Field(
        default=None,
        ge=0,
        description="Python-derived from frame manifest; None when frames invalid",
    )
    evidence_type: CandidateEvidenceType
    observed_description: str = Field(default="")
    visible_text: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    search_status: CandidateSearchStatus
    validation_status: CandidateValidationStatus
    rejection_reason: str | None = Field(
        default=None,
        description="Explicit reason when rejected; required for auditability",
    )

    @model_validator(mode="after")
    def rejected_must_have_reason(self) -> "CandidateEvidenceItem":
        if (
            self.validation_status == CandidateValidationStatus.rejected
            and not (self.rejection_reason or "").strip()
        ):
            raise ValueError("rejected candidates require rejection_reason")
        if (
            self.timestamp_start_ms is not None
            and self.timestamp_end_ms is not None
            and self.timestamp_end_ms < self.timestamp_start_ms
        ):
            raise ValueError("timestamp_end_ms must be >= timestamp_start_ms")
        return self


class CandidateEvidenceTrace(BaseModel):
    """Stage 2C output: candidate evidence only — never a verdict."""

    source_file: str = Field(description="Recording identifier matching ObservedTrace")
    candidates: list[CandidateEvidenceItem] = Field(default_factory=list)


class EvidenceRole(str, Enum):
    """Role of one observed action inside an evidence-aware support window."""

    initiation = "initiation"
    progress = "progress"
    completion = "completion"
    assertion_evidence = "assertion_evidence"
    distractor = "distractor"


class AssistProposalStatus(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    skipped = "skipped"
    failed = "failed"


class AssistMatchMethod(str, Enum):
    deterministic = "deterministic"
    llm_assisted = "llm_assisted"
    unresolved = "unresolved"
    rejected_proposal = "rejected_proposal"


class AssistReviewedPair(BaseModel):
    """A Stage 3-assisted pair Stage 4 semantic review should not re-ask."""

    expected_step_index: int = Field(ge=1)
    observed_action_index: int = Field(ge=1)
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Assist confidence for this pair. Stage 4 semantic skip requires "
            ">= 0.85 for LLM-assisted pairs; deterministic clears omit this."
        ),
    )


class AssistProposalRecord(BaseModel):
    """Audit trail for one assist candidate: accepted, rejected, skipped, or failed."""

    kind: AssistCandidateKind
    status: AssistProposalStatus
    expected_step_index: int = Field(ge=1)
    matched_observed_index: int | None = None
    supporting_observed_index: int | None = None
    initiation_observed_index: int | None = None
    completion_observed_index: int | None = None
    finding_types: list[FindingType] = Field(default_factory=list)
    model_answer: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    reason: str | None = None
    matching_method: AssistMatchMethod
    challenged_false_pass: bool = False
    evidence_timestamp_ms: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Python-derived evidence timestamp for frame_evidence clears "
            "(never LLM-invented)"
        ),
    )
    frame_ids: list[str] = Field(
        default_factory=list,
        description="Stage 2C frame labels that grounded this proposal",
    )
    candidate_evidence_type: CandidateEvidenceType | None = Field(
        default=None,
        description="ACTION or ASSERTION_OUTCOME when kind is frame_evidence",
    )


class AssistMetrics(BaseModel):
    """Numeric Stage 3 assist diagnostics — counts and durations only."""

    expected_event_count: int = Field(ge=0)
    observed_event_count: int = Field(ge=0)
    deterministic_match_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    llm_request_count: int = Field(ge=0)
    proposals_accepted: int = Field(ge=0)
    proposals_rejected: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    deterministic_ms: float = Field(ge=0.0)
    llm_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class AssistReport(BaseModel):
    """Diagnostics for one align_with_assist run. Never replaces AuditResult alone."""

    model: str
    proposals: list[AssistProposalRecord] = Field(default_factory=list)
    metrics: AssistMetrics
    reviewed_semantic_pairs: list[AssistReviewedPair] = Field(default_factory=list)
    error: str | None = None
    candidate_evidence: CandidateEvidenceTrace | None = Field(
        default=None,
        description="Stage 2C candidate trace when frame_evidence assist ran",
    )
    stage2c_skip_visual_steps: list[int] = Field(
        default_factory=list,
        description=(
            "Expected step indices whose Stage 2C ASSERTION_OUTCOME evidence "
            "was accepted at high confidence; Stage 4 may skip redundant visual recheck"
        ),
    )


class AssistedAlignResult(BaseModel):
    """Opt-in Stage 3 output: deterministic baseline plus validated assisted result."""

    deterministic_result: AuditResult
    result: AuditResult
    assist: AssistReport


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
