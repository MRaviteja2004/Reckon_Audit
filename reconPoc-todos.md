# Recording Fidelity Audit — POC Build Checklist

A step-by-step task list to take the Recording Fidelity POC from zero to a working demo.
Work through these top to bottom; check each box as you complete it.

Related design doc: [`reconPoc.md`](./reconPoc.md)

---

## Phase 0 — Setup & Prerequisites

- [x] Confirm answers to the open questions in §11 (scenario format, recording length/resolution/speed, provider constraints, sample recordings, definition of "pass")
- [x] Create the project repo and a Python virtual environment (Python 3.11+)
- [x] Add dependency management (`requirements.txt` or `pyproject.toml`)
- [x] Install core libraries: `fastapi`, `google-genai`, `pydantic`, `instructor`, `rapidfuzz`, `streamlit`
- [x] Install media libraries: `ffmpeg` (system), `av` (PyAV)
- [x] Obtain and securely store the Gemini API key (env var / `.env`, never committed)
- [x] Obtain API keys for the cross-check models (OpenAI GPT-5.5 and/or Anthropic Claude Opus 4.7)
- [x] Verify the Gemini File API works end to end with a trivial test upload

---

## Phase 1 — Define the Trace Schema *(the design comes first)*

- [x] Design the Pydantic model for an **expected step** (action, target, input value, assertion, preconditions)
- [x] Design the Pydantic model for an **observed action** (action, timestamp, screen/page, values entered, assertion-relevant evidence)
- [x] Ensure the expected trace and observed trace share the **same shape** so they can be aligned
- [x] Add per-step `confidence` and `timestamp` fields to support confidence routing later
- [x] Write a couple of hand-authored example traces (one expected, one observed) as fixtures

---

## Phase 2 — Stage 2: Video → Observed Trace *(riskiest part, prove it first)*

- [x] Wire up Gemini 3.1 Pro video ingestion via the `google-genai` SDK on a **single** recording
- [x] Use the File API for upload (not inline data) and reuse the uploaded file across queries
- [x] Get a clean, schema-valid observed trace out of one recording end to end
- [x] Enforce the observed-trace schema with Pydantic + Instructor
- [x] **Withhold the expected scenario** from the model during extraction (describe-only prompt)
- [x] Confirm timestamps are present on each observed action
- [ ] Manually verify the observed trace against the actual video for correctness

---

## Phase 3 — Stage 3: Deterministic Aligner *(zero model calls)*

- [x] Implement sequence-matching alignment between expected and observed traces (RapidFuzz + small custom aligner, or Biopython `PairwiseAligner`)
- [x] Detect **missing action** (expected step with no observed match)
- [x] Detect **extra action** (observed action not in the scenario)
- [x] Detect **wrong target / wrong input value**
- [x] Detect **assertion never observed** (the key false-positive case)
- [x] Compute an overall fidelity score
- [x] Attach the supporting video timestamp to each finding
- [x] Unit-test the aligner against hand-written trace pairs (no model calls needed)

---

## Phase 4 — Stage 1: Scenario → Expected Trace

- [x] Implement a single LLM call with a strict schema to parse scenario text into expected steps
- [x] If scenarios are Gherkin: run the deterministic Gherkin parser first, use the model only to fill semantic gaps
- [x] Validate the expected-trace output against the Pydantic schema
- [x] Test Stage 1 across each input format you'll support (free text / structured steps / Gherkin)

---

## Phase 5 — Demo UI (Streamlit)

- [x] Build an upload control for the video
- [x] Build a text input to paste the expected scenario
- [x] Run the full pipeline (Stage 1 → Stage 2 → Stage 3) on submit
- [x] Render findings with their supporting timestamps
- [x] Display the overall fidelity score and per-step verdicts

---

## Phase 6 — Cross-Check Pass (second model)

- [x] Route assertion checks to the cross-check model (GPT-5.5 or Claude Opus 4.7) on targeted frames
- [x] Extract specific frames with FFmpeg + PyAV where a higher-resolution check is needed
- [x] Route low-confidence steps to the cross-check model instead of publishing directly
- [x] Frame assertion checks as grounded yes/no ("does text matching X appear between these timestamps?")

---

## Phase 7 — Accuracy Boosters

- [x] Implement two-pass sampling: low-frame-rate skeleton pass, then high-frame-rate clipped passes per action window
- [x] Raise media resolution on assertion-bearing segments only
- [x] Use video clipping (start/end offsets) to chunk long recordings instead of one big call
- [x] Enable context caching so the uploaded video isn't re-paid for on each segment query
- [x] Derive evidence timestamps from clip offsets in code (not from model-reported times)
- [x] Keep individual requests under the ~200K-token long-context billing threshold

---

## Phase 8 — Evaluation Set & Success Criteria

- [ ] Assemble 5–10 recordings with known ground-truth traces
- [ ] Include at least one **deliberately broken** recording (test claims an error, video shows success)
- [ ] Include at least one **faithful** recording that should pass cleanly
- [ ] Verify the POC correctly **passes** faithful recordings without inventing problems
- [ ] Verify the POC correctly **flags** the deliberately broken recording
- [ ] Verify each finding carries a correct, human-verifiable timestamp (from the clip, not the model)
- [ ] Verify the same input produces the **same verdict** on repeated runs (reproducibility)

---

## Phase 9 — Storage & Wrap-up

- [ ] Persist traces and findings (SQLite or JSON files — small at POC scale)
- [ ] Write a short README covering setup, run instructions, and the evaluation results
- [ ] Re-confirm fast-moving figures (token costs, frame rates, limits) against current docs at build time
- [ ] Prepare the demo narrative, leading with the deliberately broken recording

---

## Stretch / Later Phases *(out of POC scope — note only)*

- [ ] Cost-optimised self-hosted path: OmniParser V2 + open VLM (e.g. Qwen2.5-VL), frontier reserved for ambiguous cases
- [ ] Calibrate the extractor on the real UIs under test
- [ ] Request higher quota tiers or move to a managed platform (e.g. Vertex AI) for volume
- [ ] Extend to the other six audit dimensions
