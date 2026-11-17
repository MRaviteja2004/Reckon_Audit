from typing import Literal

from pydantic import BaseModel, Field

from reckon.schema import FindingType


class StepTruth(BaseModel):
    """Human-authored expected result for one scenario step."""

    index: int = Field(
        ge=1,
        description="1-based expected step index",
    )

    should_pass: bool = Field(
        description="Whether this expected step should pass",
    )

    expected_finding_types: list[FindingType] = Field(
        default_factory=list,
        description="Finding types that Reckon should produce for this step",
    )


class CaseTruth(BaseModel):
    """Human-authored ground truth for one evaluation case."""

    case_id: str

    polarity: Literal["faithful", "broken"]

    steps: list[StepTruth]

    expected_case_findings: list[FindingType] = Field(
        default_factory=list,
        description=(
            "Expected findings that are not tied to an expected step, "
            "such as extra_action."
        ),
    )


class AssistCaseTruth(BaseModel):
    """Ground truth for Stage 3 assist evaluation cases under eval/aligner_assist/."""

    case_id: str
    notes: str = ""
    baseline_polarity: Literal["faithful", "broken"]
    assisted_polarity: Literal["faithful", "broken"]
    baseline_steps: list[StepTruth]
    assisted_steps: list[StepTruth]
    baseline_case_findings: list[FindingType] = Field(default_factory=list)
    assisted_case_findings: list[FindingType] = Field(default_factory=list)
    assist_kind: Literal["wrong_target", "delayed_assertion", "none"]
    candidate_expected: bool
    llm_required: bool
    expected_resolution: Literal["accept", "reject", "no_candidate"]
    scripted_answer: str | None = Field(
        default=None,
        description="Mock Claude answer for offline scripted mode when llm_required",
    )
    scripted_supporting_event_id: str | None = None
    scripted_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    improvement_class: Literal["deterministic_delayed", "llm_semantic", "none"] = "none"


class EvidenceRoleLabel(BaseModel):
    """Scripted role assignment for one observed event in evidence-aware eval."""

    observed_index: int = Field(ge=1)
    role: Literal[
        "initiation",
        "progress",
        "completion",
        "assertion_evidence",
        "distractor",
    ]


class EvidenceCaseTruth(BaseModel):
    """Ground truth for Stage 3 evidence-aware cases under eval/aligner_evidence/."""

    case_id: str
    notes: str = ""
    baseline_polarity: Literal["faithful", "broken"]
    evidence_polarity: Literal["faithful", "broken"]
    baseline_steps: list[StepTruth]
    evidence_steps: list[StepTruth]
    baseline_case_findings: list[FindingType] = Field(default_factory=list)
    evidence_case_findings: list[FindingType] = Field(default_factory=list)
    candidate_expected: bool
    llm_required: bool
    expected_resolution: Literal["accept", "reject", "challenge", "no_candidate"]
    scripted_labels: list[EvidenceRoleLabel] = Field(default_factory=list)
    scripted_best_initiation_id: str | None = None
    scripted_best_completion_id: str | None = None
    scripted_best_assertion_support_id: str | None = None
    scripted_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    improvement_class: Literal[
        "evidence_support",
        "evidence_challenge",
        "none",
    ] = "none"
