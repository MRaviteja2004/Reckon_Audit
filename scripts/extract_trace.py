"""Extract an observed trace from a video recording.

Usage:
    python scripts/extract_trace.py <recording.mp4> [output.json]

Scans the recording locally for the moments the screen changed, then sends the frames
at those moments to the vision model one window at a time.

The expected scenario is intentionally NOT supplied to the model. See CLAUDE.md
and reconPoc.md §2 for why this separation is critical.
"""

import sys
from pathlib import Path

# load_dotenv must run before reckon imports so that module-level os.environ.get
# calls (e.g. EXTRACTION_MODEL) pick up values from .env
from dotenv import load_dotenv
load_dotenv()

from reckon.extractor import make_client  # noqa: E402
from reckon.sampler import extract_observed_trace  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit("error: recording path required")

    recording_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Detecting screen changes and extracting from {recording_path} …", file=sys.stderr)
    trace = extract_observed_trace(
        make_client(),
        recording_path,
        source_file=recording_path.name,
    )

    result_json = trace.model_dump_json(indent=2)
    print(result_json)

    if output_path is not None:
        output_path.write_text(result_json)
        print(f"\nTrace written to {output_path}", file=sys.stderr)

    print(
        f"\nExtracted {len(trace.actions)} action(s)."
        "\nVerify each action against the recording manually before proceeding.",
        file=sys.stderr,
    )
