"""Run the Stage 2.5 cross-check pass over an existing audit.

Usage:
    python scripts/cross_check.py <expected.json> <observed.json> <recording.mp4> [refined.json]

Loads an expected trace, an observed trace, and the recording, runs the deterministic
aligner (Stage 3), then routes the assertion findings and low-confidence steps to the
second model on targeted frames. Prints the per-target verdicts and the refined audit.

Stage 3 is deterministic; the cross-check uses USSO_TOKEN through Uber AI Gateway
(with vision-capable CROSSCHECK_MODEL) and reads frames locally with PyAV — no second
video upload.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from reckon.aligner import align  # noqa: E402
from reckon.crosscheck import run_cross_check  # noqa: E402
from reckon.schema import ExpectedTrace, ObservedTrace  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        raise SystemExit("error: expected, observed, and recording paths are required")

    expected_path = Path(sys.argv[1])
    observed_path = Path(sys.argv[2])
    recording_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    for path in (expected_path, observed_path, recording_path):
        if not path.exists():
            raise SystemExit(f"file not found: {path}")

    expected = ExpectedTrace.model_validate(json.loads(expected_path.read_text()))
    observed = ObservedTrace.model_validate(json.loads(observed_path.read_text()))

    audit = align(expected, observed)
    print(f"Stage 3 fidelity score: {audit.fidelity_score:.0%}", file=sys.stderr)

    print("Running cross-check pass …", file=sys.stderr)
    report = run_cross_check(expected, observed, audit, recording_path)

    for verdict in report.verdicts:
        print(
            f"  [{verdict.answer.value}] ({verdict.confidence:.2f}) "
            f"step={verdict.expected_step_index} obs={verdict.observed_action_index} "
            f"frames={verdict.frame_timestamps_ms}: {verdict.rationale}",
            file=sys.stderr,
        )

    print(
        f"\nRefined fidelity score: {report.refined_result.fidelity_score:.0%} "
        f"({len(report.refined_result.findings)} finding(s) after cross-check)",
        file=sys.stderr,
    )

    result_json = report.model_dump_json(indent=2)
    print(result_json)
    if output_path is not None:
        output_path.write_text(result_json)
        print(f"\nReport written to {output_path}", file=sys.stderr)
