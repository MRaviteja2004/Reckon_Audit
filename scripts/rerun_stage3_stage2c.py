"""Re-run Stage 3 (+ optional Stage 2C / Stage 4) on frozen traces + recording.

Usage:
  python scripts/rerun_stage3_stage2c.py expected.json observed.json recording.mp4 \\
      --out result.json [--no-cross-check]
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from reckon.aligner import align  # noqa: E402
from reckon.assist import align_with_assist  # noqa: E402
from reckon.crosscheck import run_cross_check  # noqa: E402
from reckon.extractor import make_client  # noqa: E402
from reckon.schema import ExpectedTrace, ObservedTrace  # noqa: E402
from reckon.stage2c import build_frame_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-cross-check", action="store_true")
    args = parser.parse_args()

    expected = ExpectedTrace.model_validate_json(args.expected.read_text())
    observed = ObservedTrace.model_validate_json(args.observed.read_text())
    if not args.recording.exists():
        raise SystemExit(f"recording not found: {args.recording}")

    baseline = align(expected, observed)
    timestamps = sorted({action.timestamp_ms for action in observed.actions})
    manifest = build_frame_manifest(timestamps) if timestamps else None
    client = make_client()
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=client,
        evidence_aware=True,
        frame_evidence=True,
        frame_manifest=manifest,
        frame_video_path=args.recording,
    )
    result = assisted.result
    stage4 = None
    if not args.no_cross_check:
        stage4 = run_cross_check(
            expected,
            observed,
            result,
            args.recording,
            skip_semantic_pairs={
                (p.expected_step_index, p.observed_action_index)
                for p in assisted.assist.reviewed_semantic_pairs
            },
            skip_visual_steps=set(assisted.assist.stage2c_skip_visual_steps),
        )
        result = stage4.refined_result

    payload = {
        "scenario_title": expected.scenario_title,
        "baseline": baseline.model_dump(),
        "assisted": assisted.result.model_dump(),
        "assist_report": assisted.assist.model_dump(),
        "final": result.model_dump(),
        "cross_check": stage4.model_dump() if stage4 is not None else None,
        "stage2c_executed": assisted.assist.candidate_evidence is not None,
        "stage2c_candidate_count": (
            len(assisted.assist.candidate_evidence.candidates)
            if assisted.assist.candidate_evidence is not None
            else 0
        ),
        "assist_error": assisted.assist.error,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"baseline={baseline.fidelity_score:.3f}/{baseline.verdict.value} "
        f"assisted={assisted.result.fidelity_score:.3f}/{assisted.result.verdict.value} "
        f"final={result.fidelity_score:.3f}/{result.verdict.value} "
        f"stage2c={payload['stage2c_executed']} "
        f"candidates={payload['stage2c_candidate_count']} "
        f"error={assisted.assist.error!r}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
