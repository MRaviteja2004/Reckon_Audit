"""Stage 2C — expected-guided frame-level candidate evidence discovery."""

from reckon.stage2c.discover import discover_candidate_evidence, load_frame_jpegs
from reckon.stage2c.manifest import build_frame_manifest, materialize_timestamp_range
from reckon.stage2c.propose import (
    STAGE2C_MODEL,
    RawFrameEvidenceBatch,
    RawFrameEvidenceCandidate,
)
from reckon.stage2c.select import Stage2CSearchTarget, select_stage2c_targets
from reckon.stage2c.to_assist import (
    high_confidence_visual_skip_steps,
    proposals_from_candidate_trace,
)
from reckon.stage2c.roles import (
    SemanticRole,
    evidence_semantic_role,
    expectation_semantic_role,
    role_mismatch_reason,
)
from reckon.stage2c.validate import STAGE2C_CONFIDENCE_FLOOR, validate_stage2c_batch

__all__ = [
    "STAGE2C_CONFIDENCE_FLOOR",
    "STAGE2C_MODEL",
    "RawFrameEvidenceBatch",
    "RawFrameEvidenceCandidate",
    "SemanticRole",
    "Stage2CSearchTarget",
    "build_frame_manifest",
    "discover_candidate_evidence",
    "evidence_semantic_role",
    "expectation_semantic_role",
    "high_confidence_visual_skip_steps",
    "load_frame_jpegs",
    "materialize_timestamp_range",
    "proposals_from_candidate_trace",
    "role_mismatch_reason",
    "select_stage2c_targets",
    "validate_stage2c_batch",
]
