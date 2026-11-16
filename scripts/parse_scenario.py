"""Parse an expected scenario into an ExpectedTrace via Stage 1 (OpenAI).

Usage:
    python scripts/parse_scenario.py <scenario.txt> [output.json]
    python scripts/parse_scenario.py -    # read scenario from stdin

Reads scenario text (free text, structured steps, or Gherkin), runs the Stage 1
parser, and prints the resulting ExpectedTrace as JSON. If an output path is
given the JSON is also written there.
"""

import sys
from pathlib import Path

# load_dotenv must run before reckon imports so that module-level os.environ.get
# calls in parser.py (e.g. OPENAI_MODEL) pick up values from .env
from dotenv import load_dotenv
load_dotenv()

from reckon.parser import parse_scenario  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit("error: scenario path required (or '-' for stdin)")

    source = sys.argv[1]
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if source == "-":
        scenario_text = sys.stdin.read()
    else:
        scenario_text = Path(source).read_text()

    print("Parsing scenario via Stage 1 …", file=sys.stderr)
    trace = parse_scenario(scenario_text)

    result_json = trace.model_dump_json(indent=2)
    print(result_json)

    if output_path is not None:
        output_path.write_text(result_json)
        print(f"\nTrace written to {output_path}", file=sys.stderr)

    print(
        f"\nParsed {len(trace.steps)} step(s) for scenario: {trace.scenario_title!r}",
        file=sys.stderr,
    )
