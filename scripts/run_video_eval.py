"""Score Stage 2 against the synthetic video corpus.

Usage:
    PYTHONPATH=src:. python scripts/run_video_eval.py [--build] [--only <name>]

Options:
    --build     Re-render the corpus before scoring (needed on first run).
    --only      Score a single case by name.
    --out       Write the per-case results as JSON.

The corpus is generated, so its ground truth is exact. Three things are measured:

  detection   false positives — events reported where nothing happened. This is the
              noise-robustness number: a moving cursor, a spinner, and long idle
              stretches are all in the corpus specifically to try to produce them.
  extraction  action recall — for each action that genuinely occurred, did an observed
              action land near it, and was its type right.
  audit       for cases carrying a scenario, whether Stage 3's findings match what the
              recording actually demonstrates. login_missing_assertion is the one that
              matters: the recording omits the outcome its scenario claims, and an
              audit that passes it is worthless.

Real API calls are made, so this is a script, never a test.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rapidfuzz import fuzz  # noqa: E402

from eval.video.build import build_case  # noqa: E402
from eval.video.cases import Case, build_corpus  # noqa: E402
from eval.video.long_flow import build_long_cases  # noqa: E402
from reckon.aligner import align  # noqa: E402
from reckon.crosscheck import run_cross_check  # noqa: E402
from reckon.extractor import make_client  # noqa: E402
from reckon.keyframes import detect_change_events  # noqa: E402
from reckon.parser import make_openai_client, parse_scenario  # noqa: E402
from reckon.sampler import (  # noqa: E402
    FALLBACK_FRAME_COUNT,
    derive_action_windows,
    extract_observed_trace,
    select_window_frames,
)

CORPUS_DIR = Path("eval/video/corpus")

# A detection may lag its change by up to one sampling interval, and debouncing can
# hold it back further; anything outside this window is not explained by either.
_MATCH_EARLY_MS = 150
_MATCH_LATE_MS = 750

# How far an observed action may sit from the truth action it is credited with. Frames
# are chosen at change events, so a correct action lands on one of a few nearby frames.
_ACTION_TOLERANCE_MS = 2200

_TARGET_SIMILARITY = 60


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _covers(detections_ms: list[int], change_ms: int) -> bool:
    return any(
        change_ms - _MATCH_EARLY_MS <= d <= change_ms + _MATCH_LATE_MS
        for d in detections_ms
    )


def _is_spurious(detection_ms: int, change_times_ms: list[int]) -> bool:
    return not any(
        c - _MATCH_EARLY_MS <= detection_ms <= c + _MATCH_LATE_MS
        for c in change_times_ms
    )


def _score_case(case: Case, video_path: Path, client, cross_check: bool = False) -> dict:
    detections = detect_change_events(video_path)

    # Mirror the sampler's no-detection fallback so the reported frame count is what was
    # actually sent, not zero.
    frame_sources = detections
    if not frame_sources:
        stride = case.duration_ms // (FALLBACK_FRAME_COUNT + 1)
        frame_sources = [stride * i for i in range(1, FALLBACK_FRAME_COUNT + 1)]
    windows = derive_action_windows(frame_sources)
    frames = sum(len(select_window_frames(s, e, frame_sources)) for s, e in windows)

    spurious = [d for d in detections if _is_spurious(d, case.change_times_ms)]
    covered = [c for c in case.change_times_ms if _covers(detections, c)]

    trace = extract_observed_trace(client, video_path, source_file=video_path.name)
    observed = trace.actions

    matched, type_hits, target_hits = [], 0, 0
    for truth in case.actions:
        near = [
            a for a in observed
            if abs(a.timestamp_ms - truth.at_ms) <= _ACTION_TOLERANCE_MS
        ]
        if not near:
            continue
        matched.append(truth)
        if any(a.action.value == truth.action for a in near):
            type_hits += 1
        if truth.target and any(
            a.target and fuzz.partial_ratio(truth.target.lower(), a.target.lower()) >= _TARGET_SIMILARITY
            for a in near
        ):
            target_hits += 1

    result = {
        "case": case.name,
        "note": case.note,
        "truth_changes": len(case.change_times_ms),
        "detections": len(detections),
        "changes_covered": len(covered),
        "spurious_detections": len(spurious),
        "spurious_times_ms": spurious,
        "windows": len(windows),
        "frames_sent": frames,
        "truth_actions": len(case.actions),
        "observed_actions": len(observed),
        "actions_matched": len(matched),
        "action_type_correct": type_hits,
        "action_target_correct": target_hits,
        "audit": None,
    }

    if case.scenario:
        expected = parse_scenario(case.scenario, openai_client=make_openai_client())
        audit = align(expected, trace)
        if cross_check:
            # Stage 4 exists precisely for the failure this corpus exposes: the model
            # cannot know which on-screen evidence an assertion cares about, because the
            # scenario is withheld from it by design.
            audit = run_cross_check(expected, trace, audit, video_path).refined_result
        score = audit.fidelity_score
        # The substantive product claim is not a score threshold but whether the step
        # carrying the scenario's outcome is flagged. A recording that omits its outcome
        # must flag it; one that demonstrates it must not.
        # Severity now comes from the aligner, so the eval scores exactly what a
        # consumer sees rather than reimplementing the policy.
        flagged = sorted({f.type.value for f in audit.critical_findings})
        # Scored on the verdict alone. A score bar was tried and removed: the faithful
        # recording's fidelity range overlapped the broken ones across repeated runs, so
        # any threshold would have failed good recordings and passed bad ones.
        if case.expect_finding is not None:
            ok = bool(set(case.expect_finding) & set(flagged))
        else:
            ok = not flagged
        result["audit"] = {
            "verdict": audit.verdict.value,
            "fidelity_score": round(score, 3),
            "expect_finding": list(case.expect_finding) if case.expect_finding else None,
            "defect_findings": flagged,
            "found_findings": sorted({f.type.value for f in audit.findings}),
            "correct": ok,
        }
    return result


def _print_report(results: list[dict]) -> None:
    print()
    print("=" * 104)
    print(f"{'case':26s} {'chg cov':>9s} {'spur':>5s} {'win':>4s} {'frm':>4s} "
          f"{'act':>7s} {'type':>5s} {'tgt':>5s}  note")
    print("-" * 104)
    for r in results:
        print(
            f"{r['case']:26s} "
            f"{r['changes_covered']:>4d}/{r['truth_changes']:<4d} "
            f"{r['spurious_detections']:>5d} {r['windows']:>4d} {r['frames_sent']:>4d} "
            f"{r['actions_matched']:>3d}/{r['truth_actions']:<3d} "
            f"{r['action_type_correct']:>5d} {r['action_target_correct']:>5d}  "
            f"{r['note'][:34]}"
        )
    print("-" * 104)

    truth_actions = sum(r["truth_actions"] for r in results)
    matched = sum(r["actions_matched"] for r in results)
    types = sum(r["action_type_correct"] for r in results)
    targets = sum(r["action_target_correct"] for r in results)
    spurious = sum(r["spurious_detections"] for r in results)
    changes = sum(r["truth_changes"] for r in results)
    covered = sum(r["changes_covered"] for r in results)

    print(f"change coverage      {covered}/{changes}  ({covered / changes:.0%})")
    print(f"spurious detections  {spurious}")
    print(f"action recall        {matched}/{truth_actions}  ({matched / truth_actions:.0%})")
    print(f"action type correct  {types}/{truth_actions}  ({types / truth_actions:.0%})")
    print(f"target match         {targets}/{truth_actions}  ({targets / truth_actions:.0%})")
    print(f"frames sent (total)  {sum(r['frames_sent'] for r in results)}")

    audited = [r for r in results if r["audit"]]
    if audited:
        print()
        print("End-to-end audit verdicts:")
        for r in audited:
            a = r["audit"]
            mark = "PASS" if a["correct"] else "FAIL"
            want = " | ".join(a["expect_finding"]) if a["expect_finding"] else "no defect finding"
            print(f"  [{mark}] {r['case']:30s} {a['verdict']:10s} score={a['fidelity_score']:.2f}  "
                  f"needs {want:22s} found={a['defect_findings'] or 'none'}")
    print("=" * 104)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="Re-render the corpus first")
    parser.add_argument("--cross-check", action="store_true",
                        help="Run the Stage 4 cross-check before scoring the audit")
    parser.add_argument("--suite", choices=["probe", "long", "all"], default="all",
                        help="probe = short hazard cases, long = full multi-page flow")
    parser.add_argument("--only", default=None, help="Score only this case")
    parser.add_argument("--out", type=Path, default=None, help="Write results JSON here")
    args = parser.parse_args()

    corpus = []
    if args.suite in ("probe", "all"):
        corpus += build_corpus()
    if args.suite in ("long", "all"):
        corpus += build_long_cases()
    if args.only:
        corpus = [c for c in corpus if c.name == args.only]
        if not corpus:
            raise SystemExit(f"no such case: {args.only}")

    if args.build:
        _log(f"Rendering {len(corpus)} case(s) to {CORPUS_DIR} …")
        for case in corpus:
            build_case(case, CORPUS_DIR)

    client = make_client()
    results = []
    for case in corpus:
        video_path = CORPUS_DIR / f"{case.name}.mp4"
        if not video_path.exists():
            raise SystemExit(f"{video_path} missing — run with --build first")
        _log(f"scoring {case.name} …")
        results.append(_score_case(case, video_path, client, cross_check=args.cross_check))

    _print_report(results)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        _log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
