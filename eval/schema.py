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
