"""Run the full Recording Fidelity pipeline end to end on one recording + scenario.

Usage:
    python scripts/run_pipeline.py <recording.mp4> <scenario.txt> [options]
    python scripts/run_pipeline.py <recording.mp4> -                [options]   # scenario on stdin

Options:
    --no-cross-check     Skip the Stage 2.5 cross-check pass. It runs by default:
                         Stage 2 is never told the scenario, so it cannot know which
                         on-screen evidence an assertion cares about, and without this
                         pass faithful recordings draw false assertion findings.
    --out <path>         Write the final audit (or cross-check report) JSON to <path>.

Stages:
    1. Parse the scenario into an ExpectedTrace          (OpenAI, OPENAI_MODEL)
    2. Extract an ObservedTrace from change-event frames (OpenAI, EXTRACTION_MODEL;
       frames are decoded locally)
    3. Align the two traces deterministically            (no model)
    4. Cross-check assertions / low-confidence steps on frames (OpenAI)

USSO_TOKEN comes from the environment / .env and covers Stages 1, 2 and 4 through
Uber AI Gateway.
Stage 3 needs no key.
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from reckon.aligner import align  # noqa: E402
from reckon.crosscheck import run_cross_check  # noqa: E402
from reckon.extractor import make_client  # noqa: E402
from reckon.latency import (  # noqa: E402
    LatencyMetrics,
    elapsed_ms,
    latency_scope,
    log_metrics,
    monotonic_ms,
)
from reckon.parser import parse_scenario  # noqa: E402
from reckon.sampler import extract_observed_trace  # noqa: E402
from reckon.schema import AuditResult, ExpectedTrace  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("reckon.pipeline")


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _format_report(title: str, result: AuditResult, expected: ExpectedTrace) -> str:
    critical = result.critical_findings
    lines = [
        "",
        "=" * 60,
        f"Scenario:       {title}",
        f"VERDICT:        {result.verdict.value.upper()}"
        + (f"  ({len(critical)} critical finding(s))" if critical else ""),
        f"Steps passed:   {sum(1 for v in result.step_verdicts if v.passed)}/{len(expected.steps)}"
        f"  (fidelity {result.fidelity_score:.0%} — secondary, see README)",
        "=" * 60,
    ]
    if critical:
        lines.append("Critical findings — the recording does not demonstrate these:")
        for finding in critical:
            ts = f"@{finding.evidence_timestamp_ms}ms" if finding.evidence_timestamp_ms is not None else ""
            lines.append(f"  ! {finding.type.value} {ts}: {finding.description}")
        lines.append("")
    lines.append("Per-step verdicts:")
    for verdict in result.step_verdicts:
        mark = "PASS" if verdict.passed else "FAIL"
        ts = (
            f"@{verdict.evidence_timestamp_ms}ms"
            if verdict.evidence_timestamp_ms is not None
            else "(no evidence)"
        )
        issues = (
            "  " + ", ".join(ft.value for ft in verdict.finding_types)
            if verdict.finding_types else ""
        )
        lines.append(f"  [{mark}] step {verdict.expected_step_index} {ts}{issues}")

    advisory = [f for f in result.findings if f not in critical]
    lines.append("")
    lines.append(f"Advisory findings ({len(advisory)}) — wording and grain, not fidelity:")
    if not advisory:
        lines.append("  (none)")
    for finding in advisory:
        ts = f"@{finding.evidence_timestamp_ms}ms" if finding.evidence_timestamp_ms is not None else ""
        lines.append(f"  - {finding.type.value} {ts}: {finding.description}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path, help="Path to the recording (mp4/mov/...)")
    parser.add_argument("scenario", help="Path to a scenario text file, or '-' for stdin")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="Skip the cross-check pass (it runs by default)")
    parser.add_argument("--out", type=Path, default=None, help="Write the result JSON here")
    args = parser.parse_args()

    if not args.recording.exists():
        raise SystemExit(f"recording not found: {args.recording}")
    scenario_text = sys.stdin.read() if args.scenario == "-" else Path(args.scenario).read_text()

    latency = LatencyMetrics()
    pipeline_started = monotonic_ms()

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    _log("Stage 1 — parsing scenario …")
    stage1_started = monotonic_ms()
    expected = parse_scenario(scenario_text)
    latency.stage1_ms = elapsed_ms(stage1_started)
    logger.info("stage1 duration_ms=%.1f step_count=%d", latency.stage1_ms, len(expected.steps))
    _log(f"  → {len(expected.steps)} step(s): {expected.scenario_title!r}")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    _log("Stage 2 — detecting screen changes and extracting observed trace …")
    with latency_scope(latency):
        observed = extract_observed_trace(
            make_client(),
            args.recording,
            source_file=args.recording.name,
        )
    _log(f"  → {len(observed.actions)} observed action(s)")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    _log("Stage 3 — aligning …")
    stage3_started = monotonic_ms()
    result = align(expected, observed)
    latency.stage3_ms = elapsed_ms(stage3_started)
    logger.info(
        "stage3 duration_ms=%.1f finding_count=%d",
        latency.stage3_ms,
        len(result.findings),
    )

    output_obj = result
    # ── Stage 4 ──────────────────────────────────────────────────────────────
    if not args.no_cross_check:
        _log("Stage 4 — cross-checking assertions and low-confidence steps …")
        with latency_scope(latency):
            report = run_cross_check(expected, observed, result, args.recording)
        for verdict in report.verdicts:
            _log(
                f"  [{verdict.answer.value}] ({verdict.confidence:.2f}) "
                f"step={verdict.expected_step_index} obs={verdict.observed_action_index}: "
                f"{verdict.rationale}"
            )
        result = report.refined_result
        output_obj = report
        _log(f"  → refined verdict: {result.verdict.value}")

    latency.total_ms = elapsed_ms(pipeline_started)
    log_metrics(latency, where="run_pipeline")
    _log(
        "Latency — "
        + " ".join(f"{k}={v}" for k, v in latency.summary_fields().items())
    )

    print(_format_report(expected.scenario_title, result, expected))

    if args.out is not None:
        args.out.write_text(output_obj.model_dump_json(indent=2))
        _log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
