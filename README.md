# Reckon — Recording Fidelity Audit (POC)

Reckon audits a test recording against a set of expected scenarios and reports whether the recording actually demonstrates what the test claims. Each finding is backed by a timestamp in the video, making results human-verifiable in seconds.

The full solution design is in [`reconPoc.md`](./reconPoc.md). The phase-by-phase build checklist is in [`reconPoc-todos.md`](./reconPoc-todos.md).

---

## How it works

The expected scenario and the video are each converted into a structured trace, then compared by deterministic code. An optional second-model cross-check pass refines the highest-stakes findings.

```
Scenario text ─► Stage 1 (OpenAI) ─► ExpectedTrace ─┐
                                                     ├─► Stage 3 (aligner) ─► AuditResult ─► Stage 4 (cross-check, optional) ─► refined AuditResult
Video file ─► Stage 2 (local frames + OpenAI vision) ─► ObservedTrace ─┘
```

**Stage 1 — Scenario → ExpectedTrace (OpenAI).** A single LLM call converts the test scenario text into an ordered list of expected steps. It accepts free text, structured step lists, and Gherkin. For clean Gherkin the step lines are extracted deterministically in code and only those lines are sent to the model; anything that can't be reduced to Gherkin step lines without losing content (prose, data tables) is sent whole so nothing is dropped. Asserted/expected text always lands in the `assertion` field — never `input_value` — so Stage 3 can detect an assertion the recording never demonstrates. Generic value placeholders ("a password", "an invalid username") are **not** captured as `input_value`; only concrete literal values ("tomsmith") are, so a masked or unspecified value never produces a spurious mismatch.

**Stage 2 — Video → ObservedTrace (local frames + OpenAI vision).** The recording never leaves the machine as video. It is decoded locally, and only the frames that matter are sent as full-resolution stills:

1. **Change detection** (`keyframes.py`) scans the recording with PyAV and returns the exact moments the screen changed. No model, no API key, no network — a screen recording is static but for discrete state changes, so this is where the information is.
2. **Action windows** are derived around each change event, merged where they overlap and split where one grows too long to cover with its frame budget.
3. Per window, frames are taken **at the change events themselves** (plus the window edges for context) and sent to the vision model at high detail, so error banners, toasts, and field values are read at OCR grade.
4. The chunks are merged into one trace, each clip-relative timestamp converted to recording-absolute.

**Timestamps are never reported by the model.** It names a *frame* ("Frame 3"), and the timestamp is looked up in code from that frame's decoder presentation time. A model that misreads a frame yields a wrong-but-real pointer to a frame a human can open; it cannot invent a moment that never existed. This is the difference between an evidence pointer and a guess, and the audit rests on it.

The expected scenario is **intentionally withheld** from the model at this stage; giving it the expectation beforehand causes hallucination (the model "sees" what it expects to see), which is exactly the false-positive this tool exists to catch.

**Stage 3 — Align and score (no model).** A deterministic aligner compares the two traces. It sequence-matches expected steps to observed actions (Needleman–Wunsch with RapidFuzz fuzzy comparison) and detects five mismatch types: missing action, extra action, wrong target, wrong input value, and — the key false-positive case — an assertion the recording never demonstrates. Each finding is attached to the supporting video timestamp and classified `critical` or `advisory`, and the result carries a **`verdict`** (`faithful` / `unfaithful`) derived from the critical findings. A `fidelity_score` is also reported, but it is a secondary detail — see [Why `verdict` is the headline](#key-design-decisions).

**Stage 4 — Cross-check (optional, OpenAI vision).** The highest-stakes questions are routed to a second model on targeted, full-resolution frames pulled locally with PyAV: every assertion the aligner could *not* confirm, plus any low-confidence observed action. Each is framed as a grounded yes/no ("does text/UI matching X appear in these frames?"). A confident "present" overturns a false `assertion_not_observed` (an assertion the first pass missed at low resolution) while leaving genuine findings intact — so a faithful recording isn't penalised for an extraction miss.

**Phases 0–7 are built and working end to end. Phases 8–9 (evaluation set, persistence) are pending.** See [`reconPoc-todos.md`](./reconPoc-todos.md) for the full roadmap.

---

## Prerequisites

- Python 3.12+
- `ffmpeg` installed system-wide (`brew install ffmpeg` on Mac)
- Uber AI Gateway access and a USSO token (see [Environment variables](#environment-variables))

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/ujjwal502/Reckon.git
cd Reckon

# Create a virtual environment and install dependencies
# The project uses uv — install it first if needed: https://docs.astral.sh/uv/
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Copy the env template and fill in your API keys
cp .env.example .env
```

---

## Environment variables

All configuration lives in `.env`. Copy `.env.example` to get started — it documents every variable.

| Variable | Required | Description |
|---|---|---|
| `USSO_TOKEN` | Yes | USSO token used by every model stage through Uber AI Gateway |
| `OPENAI_MODEL` | No | Model for Stage 1 scenario parsing. Default: `gpt-5.5` |
| `EXTRACTION_MODEL` | No | Vision-capable model for Stage 2 frame extraction. Default: `gpt-5.5` |
| `CROSSCHECK_MODEL` | No | Vision-capable model for the Stage 4 cross-check. Default: `gpt-5.5` |

> `.env` is gitignored. Never commit it.

All model requests use `https://genai-api.uberinternal.com/v1`; there is no public
OpenAI endpoint fallback.

### Model notes

- **Vision is required.** `EXTRACTION_MODEL` and `CROSSCHECK_MODEL` are both sent images and must be vision-capable. A key that works for text can still fail here — `scripts/verify_openai.py <recording>` checks the vision path specifically.
- **Temperature.** The GPT-5 family (default `gpt-5.5`) accepts only its default temperature, so the pipeline omits the `temperature` parameter rather than sending `0`. Older models that support `temperature=0` still work and can be pinned via the API.
- **Retries.** Stages 2 and 4 automatically retry transient `429`/`5xx` responses, honoring machine-readable gateway retry headers and otherwise using bounded exponential backoff; only a sustained outage surfaces as an error.
- **Why not Gemini.** Gemini leads on general video benchmarks and was the original Stage 2 provider. Neither OpenAI nor Anthropic accepts video via API, but that turned out not to matter: those benchmarks measure natural video, while a screen recording is static but for discrete changes. Decoding locally gives exact timestamps, no OCR downsampling, and a free change-detection pass — see [Stage 2](#how-it-works) above.

---

## Running tests

```bash
source .venv/bin/activate
pytest
```

---

## Evaluating Stage 2 against synthetic recordings

Unit tests prove the code does what it was written to do. They cannot tell you whether the audit is *right* about a recording. That needs recordings whose truth is known exactly, which means generating them:

```bash
# First run renders the corpus (~1 min); afterwards it is reused.
PYTHONPATH=src:. python scripts/run_video_eval.py --build --cross-check

PYTHONPATH=src:. python scripts/run_video_eval.py --suite long --cross-check
PYTHONPATH=src:. python scripts/run_video_eval.py --only checkbox_toggle
```

This makes **real API calls**, so it is a script and never part of `pytest`.

The corpus (`eval/video/`) is rendered at 1920×1080 with real text at real sizes, and every frame carries a moving cursor — so a UI change occupies a fraction of a percent of the frame and detection has a genuine noise floor to discriminate against, exactly as on a real capture. It has two halves:

- **Probe cases** (`cases.py`) — ten short recordings, each isolating one hazard: a 26×26px checkbox, a 450 ms toast, a spinner animating throughout, eleven idle seconds, scrolling, a keystroke burst.
- **The long flow** (`long_flow.py`) — a 90-second, five-page refund journey in four variants sharing one scenario: faithful, outcome-never-renders, one-field-wrong (£8.90 instead of £89.00), and opposite-outcome ("Refund failed"). The last two matter most. A recording that is wholly wrong is easy to catch; a recording that is wrong in one field, or one word, is what ships.

What it measures: **spurious detections** (events reported where nothing happened), **action recall** (did each real action get observed), and per case whether the audit reaches the right verdict — which finding type, not just a score.

All tests are mocked — **no real API or network calls are made**. Tests cover the schema and cross-check models, the Stage 1 parser (format detection, Gherkin extraction, data-loss guards, prompt contracts, generic-value handling), change detection (scene boundaries, localised changes, debouncing, calibration guards), the extractor (frame-index resolution, clip-relative timestamps, request construction, transient-error retry), the sampler (window derivation and frame selection), PyAV frame extraction, and the Stage 3 aligner and Stage 4 cross-check (target selection, verdict application, finding overturn).

---

## Running the full pipeline

`scripts/run_pipeline.py` runs all four stages on one recording + scenario in a single command:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_pipeline.py recording.mp4 scenario.txt --cross-check --out report.json

# scenario on stdin instead of a file:
PYTHONPATH=src python scripts/run_pipeline.py recording.mp4 - --cross-check < scenario.txt
```

| Option | Effect |
|---|---|
| `--cross-check` | Run the Stage 4 cross-check pass (second model on targeted frames) |
| `--out <path>` | Write the audit (or, with `--cross-check`, the full cross-check report) as JSON |

It prints per-stage progress to stderr (including any retry-on-503 backoff) and a readable audit report to stdout: the fidelity score, a PASS/FAIL verdict per step, and every finding with its supporting timestamp.

---

## Demo UI (Streamlit)

```bash
source .venv/bin/activate
streamlit run app.py
```

Upload a recording, paste the expected scenario (free text, numbered steps, or Gherkin), optionally tick **Run cross-check pass (second model)**, and click **Run Audit**. The app runs the full Stage 1 → 4 pipeline with a live status panel, then renders the verdict banner, fidelity score, findings (sorted by severity, each with its video timestamp), cross-check verdicts, and per-step verdicts. A collapsible **Pipeline Debug Output** panel shows the raw JSON from every stage — copy it to diagnose a run.

---

## Parsing a scenario into an expected trace (Stage 1)

This is the Stage 1 operation in isolation: turn scenario text into a structured `ExpectedTrace`. It accepts free text, numbered/bulleted steps, or Gherkin.

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/parse_scenario.py path/to/scenario.txt expected.json

# or pipe the scenario in on stdin
echo "Scenario: ..." | PYTHONPATH=src python scripts/parse_scenario.py -
```

The script:
1. Routes the text by format — clean Gherkin uses deterministic step extraction; everything else is sent whole to the model (so no content is dropped)
2. Makes a single OpenAI call with a strict response schema. Concrete literal values are captured as `input_value`; generic placeholders ("a password") are left empty
3. Reindexes steps sequentially and validates against the `ExpectedTrace` Pydantic schema
4. Writes the result as JSON and prints it to stdout

**Example output** (from a Gherkin login scenario):

```json
{
  "scenario_title": "Login with invalid password shows error message",
  "steps": [
    {
      "index": 1,
      "action": "navigate",
      "target": "login page",
      "input_value": null,
      "assertion": null,
      "preconditions": null,
      "raw_text": "the user is on the login page"
    },
    {
      "index": 5,
      "action": "assert_text",
      "target": "error message",
      "input_value": null,
      "assertion": "Invalid email or password",
      "preconditions": null,
      "raw_text": "the error message \"Invalid email or password\" is displayed"
    }
  ]
}
```

The expected text (`"Invalid email or password"`) goes in `assertion`, not `input_value` — this is what lets Stage 3 catch a recording that never demonstrates the assertion.

The parser rejects empty input and input containing more than one `Scenario:` block (the `ExpectedTrace` schema models a single scenario), raising a `ValueError` in each case.

---

## Extracting a trace from a recording

This is the Stage 2 operation in isolation: decode a recording locally and get back a structured observed trace.

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/extract_trace.py path/to/recording.mp4 output.json
```

The script:
1. Scans the recording with PyAV for the moments the screen changed — no model call
2. Derives action windows around those moments and sends each window's frames to the vision model with a describe-only prompt (no expected scenario)
3. Merges the chunks into a single recording-absolute `ObservedTrace` and validates it against the Pydantic schema
4. Writes the result as JSON and prints it to stdout

**Example output** (`output.json` from a login flow recording):

```json
{
  "source_file": "recording.mp4",
  "clip_offset_ms": 0,
  "clip_end_ms": null,
  "actions": [
    {
      "index": 1,
      "timestamp_ms": 2000,
      "action": "click",
      "target": "Username input field",
      "input_value": null,
      "assertion_evidence": null,
      "confidence": 1.0,
      "screen_description": "Login page with Username and Password fields."
    },
    {
      "index": 6,
      "timestamp_ms": 6000,
      "action": "assert_visible",
      "target": "Error message banner",
      "input_value": null,
      "assertion_evidence": {
        "description": "Red error banner appearing at the top of the content area.",
        "visible_text": "Your password is invalid!"
      },
      "confidence": 1.0,
      "screen_description": "Page reloaded, showing an error indicating an invalid password."
    }
  ]
}
```

Note that `input_value` is `null` for password fields — the model correctly cannot read masked input and reports null rather than guessing.

### Converting a QuickTime recording

QuickTime saves screen recordings as `.mov`. Convert to MP4 before running the script:

```bash
ffmpeg -i ~/Downloads/your-recording.mov -c:v libx264 -c:a aac recording.mp4
```

> macOS filenames from QuickTime contain a Unicode narrow no-break space (` `) before "AM"/"PM". If `ffmpeg` can't find the file, use Python to pass the path: `python -c "import subprocess; subprocess.run(['ffmpeg', '-i', '/path/to/file', ...])"`.

---

## Auditing a trace pair (Stage 3)

This is the Stage 3 operation in isolation: align an expected trace against an observed trace and produce a scored fidelity report. No model is called — the comparison is pure deterministic code, so the same inputs always produce the same verdict.

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_audit.py expected.json observed.json
```

With no arguments it runs on the bundled login fixtures so you can see the output shape immediately:

```bash
PYTHONPATH=src python scripts/run_audit.py
```

The script prints the fidelity score, a PASS/FAIL verdict per expected step, and a list of findings — each carrying the supporting video timestamp:

```
Scenario:       Login with invalid password shows error message
Fidelity score: 80%

Per-step verdicts:
  [PASS] step 1 @1200ms
  [PASS] step 2 @3400ms
  [PASS] step 3 @5100ms
  [PASS] step 4 @6800ms
  [FAIL] step 5 (no evidence)

Findings:
  - missing_action : Expected step 5 (assert_text on 'Error banner') was not observed in the recording.
  - extra_action @8500ms: Observed action 5 (navigate on '/dashboard') was not in the expected scenario.
```

This is the demonstration case: a test that claims to verify an error message, but whose recording quietly logs in successfully instead. The aligner catches that the expected assertion never happened (`missing_action`) and that the recording did something the scenario never described (`extra_action`).

---

## Cross-checking a verdict (Stage 4)

This is the cross-check pass in isolation: re-examine the highest-stakes findings with a second model on targeted frames. It needs the recording itself (to pull frames) plus the expected and observed traces.

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/cross_check.py expected.json observed.json recording.mp4 report.json
```

The script runs the Stage 3 aligner, then for every unconfirmed assertion and every low-confidence observed action it extracts the relevant frames (PyAV), asks the cross-check model a grounded yes/no, and prints each verdict plus the refined fidelity score. A confident "present" overturns a false `assertion_not_observed` and flips the step to pass; genuine findings on the same step (e.g. `wrong_target`) are preserved. The full `CrossCheckReport` (verdicts + refined `AuditResult`) is written to the output path.

---

## Verifying your API key

Run this once to confirm your key, the local decode path, and the vision path all work. A key that works for text can still fail on vision if the configured model isn't vision-capable, so pass a recording to check that too:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/verify_openai.py recording.mp4
```

Expected output:
```
Text OK    — gpt-5.5 replied 'ok'
Decode OK  — 7 change event(s) found in recording.mp4
Frame OK   — decoded frame at 1030ms (68421 bytes JPEG)
Vision OK  — gpt-5.5 replied 'Login page with error'
```

---

## Data models

The core domain models — everything that crosses the Stage 1 → 3 → 4 boundaries — live in `src/reckon/schema.py`. Private, underscore-prefixed LLM/API response submodels (`_LLMTrace`, `_ExtractedActions`, `_AssertionCheck`) live with the feature module that issues the call.

### `ExpectedTrace`
Parsed from the scenario text by Stage 1 (`reckon.parser.parse_scenario`).

| Field | Type | Description |
|---|---|---|
| `scenario_title` | `str` | Title of the test scenario |
| `steps` | `list[ExpectedStep]` | Ordered list of expected steps |

### `ExpectedStep`
| Field | Type | Description |
|---|---|---|
| `index` | `int` | 1-based position in the scenario |
| `action` | `ActionType` | What type of action is expected |
| `target` | `str \| None` | UI element, page, or URL |
| `input_value` | `str \| None` | Text to type, option to select, or value to assert |
| `assertion` | `str \| None` | What must be true after this step |
| `preconditions` | `str \| None` | State required before this step |
| `raw_text` | `str \| None` | The original scenario line this step was parsed from |

### `ObservedTrace`
Produced by Stage 2 from the video.

| Field | Type | Description |
|---|---|---|
| `source_file` | `str` | Filename of the source recording |
| `clip_offset_ms` | `int` | Start offset of this clip within the full recording (ms) |
| `clip_end_ms` | `int \| None` | End offset; `None` means to the end of the recording |
| `actions` | `list[ObservedAction]` | Ordered list of observed actions |

### `ObservedAction`
| Field | Type | Description |
|---|---|---|
| `index` | `int` | 1-based sequential index (always set by code, not the model) |
| `timestamp_ms` | `int` | Milliseconds from clip start |
| `action` | `ActionType` | Observed action type |
| `target` | `str \| None` | Element or page acted upon |
| `input_value` | `str \| None` | Value entered or observed (`null` for masked fields) |
| `assertion_evidence` | `AssertionEvidence \| None` | Visible text that confirms or refutes a test outcome |
| `confidence` | `float` | Model confidence 0–1; low values are routed to the Stage 4 cross-check |
| `screen_description` | `str \| None` | Brief description of screen state at this moment |

### `AssertionEvidence`
Nested inside `ObservedAction`. Populated whenever a visible message appears that could confirm or refute a test outcome.

| Field | Type | Description |
|---|---|---|
| `description` | `str` | What was observed on screen (e.g. `"Error message: Invalid email"`) |
| `visible_text` | `str \| None` | Exact text read from the screen, if legible |

### `ActionType` enum
`click` · `type_text` · `select` · `navigate` · `scroll` · `hover` · `assert_visible` · `assert_text` · `assert_state` · `other`

### `merge_observed_chunks(chunks)`
Merges multiple `ObservedTrace` clips from the same recording into a single trace. Converts each action's `timestamp_ms` from clip-relative to recording-absolute by adding `clip_offset_ms`. Used by the two-pass sampler to reassemble the per-window detail passes.

### `AuditResult`
Produced by Stage 3 (`reckon.aligner.align`). The output of the whole pipeline.

| Field | Type | Description |
|---|---|---|
| `fidelity_score` | `float` | Fraction of expected steps that passed (0.0–1.0). Extra actions are reported as findings but do not lower the score — read `findings` for the full picture |
| `findings` | `list[Finding]` | Every mismatch detected, each with a supporting timestamp |
| `step_verdicts` | `list[StepVerdict]` | One PASS/FAIL verdict per expected step |

### `Finding`
| Field | Type | Description |
|---|---|---|
| `type` | `FindingType` | `missing_action` · `extra_action` · `wrong_target` · `wrong_input_value` · `assertion_not_observed` |
| `expected_step_index` | `int \| None` | Expected step involved (`None` for `extra_action`) |
| `observed_action_index` | `int \| None` | Observed action involved (`None` for `missing_action`) |
| `evidence_timestamp_ms` | `int \| None` | Absolute timestamp supporting the finding (`None` when nothing was observed) |
| `description` | `str` | Human-readable explanation |

### `StepVerdict`
| Field | Type | Description |
|---|---|---|
| `expected_step_index` | `int` | Which expected step this verdict is for |
| `passed` | `bool` | Whether the step was confirmed by the recording |
| `matched_observed_index` | `int \| None` | Observed action this step was matched to |
| `evidence_timestamp_ms` | `int \| None` | Timestamp of the matched action |
| `finding_types` | `list[FindingType]` | Findings raised for this step; empty if passed |

### `align(expected, observed)`
The Stage 3 entry point. Takes an `ExpectedTrace` and an `ObservedTrace`, returns an `AuditResult`. Sequence-matches the two traces with a Needleman–Wunsch dynamic-programming aligner (using RapidFuzz for fuzzy target/value/assertion comparison) and classifies each mismatch. No model calls.

### Cross-check models (Stage 4)

Produced by `reckon.crosscheck.run_cross_check`.

**`CrossCheckVerdict`** — one grounded yes/no judgement.

| Field | Type | Description |
|---|---|---|
| `question` | `str` | The grounded yes/no question put to the model |
| `answer` | `CrossCheckAnswer` | `present` · `absent` · `unclear` |
| `confidence` | `float` | Model confidence 0–1 |
| `rationale` | `str` | One-sentence justification grounded in the frames |
| `frame_timestamps_ms` | `list[int]` | Absolute timestamps of the frames the model was shown |
| `expected_step_index` | `int \| None` | Related expected step, if any |
| `observed_action_index` | `int \| None` | Related observed action, if any |

**`CrossCheckTarget`** — a single thing to re-examine (`kind` is `assertion` or `low_confidence`) with the frame window (`window_start_ms`, `window_end_ms`). **`CrossCheckReport`** — the pass output: `verdicts: list[CrossCheckVerdict]` plus `refined_result: AuditResult`.

### `parse_scenario(scenario_text, *, openai_client=None, model="gpt-5.5", temperature=None)`
The Stage 1 entry point (`reckon.parser`). Takes scenario text (free text, structured steps, or Gherkin) and returns a validated `ExpectedTrace` via a single OpenAI call. `temperature` defaults to `None` (omitted from the request, since GPT-5-family models accept only their default); pass a value to pin it on a model that supports it. Pass `openai_client` to inject a test double. Raises `ValueError` on empty input or input containing more than one `Scenario:` block.

### `extract_observed_trace(openai_client, video_path, source_file, *, model=EXTRACTION_MODEL, max_frames_per_window=8, on_retry=None)`
The Stage 2 entry point (`reckon.sampler`). Runs change detection → action windows → frame selection → one vision call per window → merge, and returns a single recording-absolute `ObservedTrace`. `on_retry` is a callback invoked on each transient-error backoff (used by the UI to show a "model busy" note). Returns an empty trace without making any model call when nothing on screen changed.

### `detect_change_events(video_path, *, sample_fps=4.0, threshold=0.0025, min_gap_ms=700, max_events=60)`
The Stage 2a entry point (`reckon.keyframes`). Scans a recording locally with PyAV and returns ascending millisecond offsets where the screen changed materially. No model, no network. The score is a *changed-pixel fraction*, not a mean difference — a mean is diluted by frame size, so on a high-resolution capture it averages a real but localised change (typing into one field) down into the noise.

### `run_cross_check(expected, observed, audit_result, video_path, *, openai_client=None, ...)`
The Stage 4 entry point (`reckon.crosscheck`). Selects the assertion findings and low-confidence actions to re-examine, verifies each on targeted frames via the cross-check model, and returns a `CrossCheckReport` (the per-target verdicts plus a refined `AuditResult`). Returns the original result unchanged when nothing needs a second opinion.

---

## Key design decisions

**Why the expected scenario is withheld during extraction**
Giving the model the expected outcome before it watches the video causes it to hallucinate that outcome — "object-presence bias". This produces false passes, the exact failure the audit exists to catch. Stage 2 is always describe-only.

**Why timestamps come from code, not the model**
Model-reported timestamps drift and are unreliable as evidence pointers. The extraction model is never asked for a time at all: it names a *frame* ("Frame 3"), and the timestamp is looked up from that frame's decoder presentation time. An out-of-range frame number is clamped in code, `clip_offset_ms`/`clip_end_ms` are always set by the caller, and the absolute timestamp is computed as `action.timestamp_ms + clip_offset_ms` in `merge_observed_chunks`. Every timestamp in a finished audit therefore points at a frame that exists and can be opened.

**Why `verdict` is the headline and `fidelity_score` is not**
`fidelity_score` is the fraction of expected steps that passed — a summary for a human, not a judgement. On the 90-second synthetic flow run repeatedly, the *faithful* recording scored between 0.69 and 0.85 while the deliberately broken variants scored between 0.61 and 0.77. Those ranges overlap, and in one run the faithful recording and the one whose confirmation never rendered **both scored exactly 0.77**. Extraction is not deterministic, and a longer recording gives it more opportunities to describe an action at a different grain than the scenario was written at, which costs steps that were genuinely demonstrated.

So the audit reports a `verdict` — `faithful` or `unfaithful` — derived from findings rather than from step counts. Each finding carries a `severity`:

- **critical** — `assertion_not_observed`, `wrong_input_value`, and `missing_action` on a step carrying an assertion. Across every run of the corpus these appeared only on recordings that genuinely failed their scenario.
- **advisory** — `wrong_target`, `extra_action`, and `missing_action` on a plain action step. These appear on every long run *including faithful ones*, because Stage 2 names elements in its own words and reports actions a scenario leaves implicit.

`verdict` is `unfaithful` when any critical finding survives. That trade is deliberate and not free: a recording that genuinely skips a plain action is reported without being condemned. Blocking on it would condemn faithful recordings far more often, and assertions — the substance of any scenario — stay critical either way. Severity is assigned in `tolerance.py` and carried on every `Finding`, so the UI cannot drift from the policy the verdict is computed with.

**Why Stage 3 uses no model**
The comparison between expected and observed traces is deterministic code (RapidFuzz sequence matching). This makes findings reproducible, auditable, and independent of model variation. A finding is only as trustworthy as the process that produced it.

**Why `instructor` is used throughout**
`instructor` patches the OpenAI client so every model call — Stage 1 parsing, Stage 2 extraction, and the Stage 4 cross-check — returns a validated Pydantic model directly, with automatic retry on schema-invalid output. Now that all three stages speak to the same provider, they share one client factory (`make_openai_client`) and one structured-output mechanism.

**Why the pipeline omits `temperature` by default**
Reproducibility wants `temperature=0`, but the recommended GPT-5-family models accept only their default temperature and reject any explicit value. Stage 1 and the cross-check therefore omit the parameter rather than send `0`. It remains overridable (`parse_scenario(temperature=...)`) for models that support pinning it.

**Why generic value placeholders aren't treated as input values**
"Enter a password" describes a value; it doesn't give one. Capturing "a password" as `input_value` would produce a spurious `wrong_input_value` against the real (masked or different) on-screen value. Stage 1 captures only concrete literals as `input_value`, so the value check is skipped where there is no real value to compare.

**Why alignment policy lives apart from alignment mechanics**
`aligner.py` holds the sequence alignment itself — DP matching, scoring, finding classification — which rarely changes. `tolerance.py` holds the judgements about what counts as a genuine discrepancy versus a wording or observability artifact, which change every time a new recording reveals a way the pipeline can be right in substance and wrong on a technicality. Every rule there exists to prevent a *false finding*, and each carries the condition that stops it producing the opposite error; that asymmetry is the module's whole contract.

**Why a focusing click is satisfied by the typing that follows it**
Clicking into a field to focus it changes almost nothing on screen — a thin ring and a caret, measured on the synthetic corpus at roughly the same magnitude as the mouse pointer's own movement. Such a click is therefore routinely absent from the observed trace even on a faithful recording, and reporting it as `missing_action` penalises the recording for the limits of frame differencing. When the *next* expected step types or selects into the same target **and that step was itself observed**, the click is accepted as evidenced. The guard is what makes this safe: if the field was never filled, nothing is implied and the click still fails, so the leniency can never manufacture a pass for something the recording does not show.

**Why a confirmed assertion does not also report `wrong_target`**
The "target" of an assertion step is whatever noun the scenario used — "the message", "a banner" — not a real element name. Comparing it to the element the model named fails on wording rather than fidelity: `'message'` versus `'Success banner'` for the same confirmed text. Once the asserted text is confirmed, the substantive check has passed and the target comparison is skipped. An assertion that was *not* confirmed still reports the mismatch, where it remains diagnostic.

**Why the cross-check can only overturn toward "pass"**
The Stage 4 cross-check exists to rescue assertions a low-resolution first pass missed on a *faithful* recording — the false-finding mirror of the false pass. It overturns an `assertion_not_observed` only on a confident "present", and never invents a new failure, so a single uncertain second opinion can't silently flip a verdict. Low-confidence verdicts are advisory and don't change the score.

**Why transient API errors are retried**
Stage 2 makes one call per action window, so the odds of at least one hitting a transient `503`/`429` are real. Stages 2 and 4 retry those responses using gateway retry headers or bounded exponential backoff (Stage 2 reports retries through `on_retry`) instead of aborting the run; non-transient errors still fail immediately.

**Why change detection runs locally instead of asking a model**
Finding *when* things happened used to be a whole-recording model call. Doing it with a decoder is free, needs no key, and — the real point — produces timestamps that are exact rather than estimated. It also makes the frame budget purposeful: frames are spent at the moments the screen actually changed, not on a fixed cadence over a mostly-static screen.

**Why Gherkin headers are withheld from the Stage 1 model**
For clean Gherkin, only the `Given/When/Then` step lines are sent to the model; the scenario title is taken from the source. Any input that can't be reduced to step lines without dropping content (prose lines, data tables, doc strings) falls back to sending the whole text, so the parser never silently loses a step.

---

## Build status

| Phase | What | Status |
|---|---|---|
| 0 | Setup, dependencies, API key verification | Complete |
| 1 | Trace schema (`ExpectedTrace`, `ObservedTrace`) | Complete |
| 2 | Stage 2: video → observed trace (local frames + OpenAI vision) | Complete |
| 3 | Stage 3: deterministic aligner | Complete |
| 4 | Stage 1: scenario text → expected trace | Complete |
| 5 | Demo UI (Streamlit) | Complete |
| 6 | Cross-check pass (OpenAI vision on targeted frames) | Complete |
| 7 | Accuracy boosters (two-pass sampling, context caching, audio anchor, retry) | Complete |
| 8 | Evaluation set and success criteria | Not started |
| 9 | Storage and wrap-up | Not started |

The pipeline has been run end to end against a real recording: scenario parse → two-pass extraction → alignment → cross-check, including a live demonstration of the cross-check overturning a false `assertion_not_observed`.
