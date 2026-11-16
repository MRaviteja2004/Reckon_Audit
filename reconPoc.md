# Recording Fidelity Audit
## Proof-of-Concept Solution Approach
*Validating test recordings against expected scenarios*

---

**Prepared for:** Test Automation Audit Platform initiative  
**Scope:** POC — Recording Fidelity dimension only  
**Version:** 1.0 — Draft for review  
**Date:** 28 May 2026  
**Status:** Facts verified against current model/library state as of May 2026  

---

## Contents

1. [POC Objective](#1-poc-objective)
2. [Core Design Principle](#2-core-design-principle)
3. [Solution Architecture](#3-solution-architecture)
4. [Why This Is the Most Accurate Approach](#4-why-this-is-the-most-accurate-approach)
5. [Technology Stack](#5-technology-stack)
6. [Model Suitability and Scaling](#6-model-suitability-and-scaling)
7. [Build Sequence — Fastest Path to a Demo](#7-build-sequence--fastest-path-to-a-demo)
8. [Accuracy Boosters](#8-accuracy-boosters)
9. [Evaluation and Success Criteria](#9-evaluation-and-success-criteria)
10. [POC-to-Production Gap](#10-poc-to-production-gap)
11. [Open Questions for Confirmation](#11-open-questions-for-confirmation)

---

## 1. POC Objective

This document defines the approach, technology stack, and build plan for a proof-of-concept that audits a test recording against a set of expected scenarios and reports, with high accuracy, whether the recording actually demonstrates what the test claims.

For the POC we assume the following inputs are available:

- A video recording of a test being performed (an MP4 or equivalent), produced by the testing platform. Direct access to that platform is not assumed for the POC; we work from the exported video alone.
- A set of expected behaviours for that recording, expressed as test cases or scenarios in text (free text, structured steps, or Gherkin).

The deliverable is a working tool that takes these two inputs and produces a fidelity verdict per expected step, an overall fidelity score, and a list of findings — each backed by a timestamp in the video as evidence.

**In scope for the POC:** the Recording Fidelity audit only — does the video demonstrate the steps and assertions described in the linked test case?

**Out of scope for the POC:** the other six audit dimensions, production-scale throughput, cost optimisation, and integration with the source testing platform. These are addressed in later phases.

---

## 2. Core Design Principle

The most important design decision is what NOT to do. The obvious approach — hand the video and the expected scenario to a multimodal model and ask "do these match?" — is the least accurate option available. The model performs extraction, alignment, and judgement in a single step; errors compound silently; findings carry no evidence trail; and results are not reproducible. An auditor that works this way has the same trust problem as the tests it is meant to audit.

Instead, the work is decomposed into three stages, each independently verifiable. A model is used only for the narrow, checkable task of describing what it sees; the comparison itself is deterministic code. This is what makes a finding defensible: every result can be traced to a specific observation at a specific timestamp.

---

## 3. Solution Architecture

The pipeline has three stages. The expected scenario and the video are processed into the same structured shape (a "trace"), and the two traces are then compared by deterministic logic.

### Stage 1 — Parse the expected scenario into a structured trace

A single language-model call with a strict schema converts the test case or scenario text into an ordered list of expected steps. Each step records the action, its target, any input value, and the assertion that the step is expected to verify, along with any preconditions.

> **Note:** If the scenarios are in Gherkin, the deterministic Gherkin parser is run first to produce a structured tree, and the model is used only to fill semantic gaps. This reduces reliance on the model where structure already exists.

### Stage 2 — Extract an actual trace from the video

The video is processed into a structured trace of what actually happened: an ordered list of observed actions with timestamps, the screen or page at each step, the values seen entered into fields, and any assertion-relevant evidence observed (an error message appearing, a success confirmation, a field validation state).

> **Critical:** In this stage the model never sees the expected scenario. It is asked only to describe what the recording shows, not to judge it against an expectation. This separation is what keeps the extraction honest and prevents the model from "seeing what it expects to see."

### Stage 3 — Align the two traces and score

The expected trace and the observed trace are aligned by deterministic sequence-matching code — no model is involved. Each category of mismatch falls out mechanically:

- **Missing action** — an expected step has no corresponding observed action.
- **Extra action** — the recording does something not described in the scenario.
- **Wrong target or wrong input value** — the action occurred but on the wrong element or with the wrong data.
- **Assertion never observed** — the most important finding: the expected verification (e.g. an error message) never appears in the video. This is the classic false-positive a recording can hide.

The output is an overall fidelity score plus a list of findings, each carrying the video timestamp that supports it.

---

## 4. Why This Is the Most Accurate Approach

- **Evidence on every finding.** Each finding points to a specific timestamp a human can verify in seconds, e.g. "the test expects an error message, but no matching text appears between 0:12 and 0:18."
- **Bounded model error.** The model's job is limited to describing observations. Hallucination is constrained and checkable; it cannot silently corrupt the verdict because the verdict is computed in code.
- **Partial-credit signal.** "Right action, assertion never validated" is distinguished from "wrong action entirely." Both are useful; neither emerges cleanly from a single-shot judgement.
- **Reproducible.** The same video and scenario produce the same verdict, which is essential for an audit tool that must be defensible.

---

## 5. Technology Stack

The stack below is chosen for the fastest path to an accurate POC. It favours managed frontier models over self-hosted components, because at POC stage accuracy and speed-to-demo matter more than per-unit cost. The cost-optimised alternative is described in Section 9.

| Layer | Recommended Choice | Role |
|---|---|---|
| Language / backend | Python with FastAPI | Orchestration; for a POC, async functions are sufficient — no heavy workflow engine |
| Video understanding | Gemini 3.1 Pro via the google-genai SDK | Stage 2 — native video ingestion with timestamps; the primary accelerator |
| Structured output | Pydantic schemas; Instructor for a cross-model abstraction | Enforces the trace schema in Stages 1 and 2 |
| Cross-check / assertions | GPT-5.5 or Claude Opus 4.7 on targeted frames | Second opinion where assertion precision matters most |
| Frame extraction (targeted) | FFmpeg with PyAV | Pull specific frames only where a higher-resolution check is needed |
| Alignment | RapidFuzz plus a small custom aligner (or Biopython PairwiseAligner) | Stage 3 — deterministic, no model |
| Demo UI | Streamlit | Upload video, paste expectation, view findings with timestamps |
| Storage | SQLite or JSON files | Traces and findings are small at POC scale |

### Why Gemini for the video stage

Modern Gemini models ingest video natively: the file is uploaded directly and sampled at one frame per second by default, with timestamps added each second, and the rate can be raised (up to ten frames per second) for fast action. This removes the entire frame-extraction, scene-detection and frame-stitching pipeline that would otherwise be needed, and gives the model genuine temporal context across the clip. Gemini 3.1 Pro is natively multimodal across text, image, audio and video with a one-million-token context window, which comfortably covers a typical test recording.

The frontier models from OpenAI (GPT-5.5) and Anthropic (Claude Opus 4.7) are used as a cross-check on the specific frames where an assertion must be confirmed, because on published GUI-grounding benchmarks the OpenAI line has held the strongest scores. Using a second model on the highest-stakes question — "did the expected message actually render?" — reduces the chance of a false pass.

---

## 6. Model Suitability and Scaling

Two questions determine whether the recommended model is the right foundation: does it produce accurate results on UI recordings, and does it scale to large videos? Both have affirmative answers, but each comes with a condition that shapes the build.

### Will it produce good results on UI recordings?

Yes, with tuning. Google positions Gemini's UI-understanding capability explicitly at QA testing and UX analysis, and the model has improved on GUI-grounding benchmarks — so screen recordings of UI tests are squarely in scope. The important caveat is that good accuracy is not automatic at default settings; the gap between default and tuned is the gap between a convincing demo and one that quietly produces wrong verdicts.

Accuracy on UI recordings depends on three settings:

- **Frame rate.** At the default one-frame-per-second sampling, fast action sequences lose detail — and UI recordings are mostly fast actions: a click, a quick field entry, a page transition, an error message that flashes briefly. The model supports up to ten frames per second; raising the rate (or slowing the clip) around action moments is necessary to catch transient events such as a momentary error toast, which is exactly the kind of assertion this audit must verify.
- **Media resolution.** Reading an error message or a field value is an OCR-grade task. A higher media-resolution setting improves the model's ability to read fine text and small details, at the cost of more tokens per frame, so resolution should be raised on the segments where assertions are checked.
- **Withholding the expectation during extraction.** Supplying the model with the expected outcome before it watches the video is a known source of hallucination (object-presence bias): told that a test verifies an error, the model is predisposed to "see" that error whether or not it rendered. This produces the worst possible audit outcome — a false pass. It is the direct reason Stage 2 describes what happened without knowing what was expected.

With frame rate raised, resolution raised on assertion-bearing segments, and the expectation withheld during extraction, results are good enough for a credible POC. Left at defaults, output is fluent, plausible, and occasionally wrong — the trap an audit tool must avoid.

### Will it scale to large videos?

For typical recordings of seconds to a few minutes, comfortably and cheaply. For genuinely long recordings, the naive approach of feeding the whole video in one call does not scale, for two independent reasons.

**Context.** The model has roughly a one-million-token context window and can process on the order of one hour of video in a single prompt — but only at default sampling. Video costs roughly 300 tokens per second of footage at default resolution. The multiplier is the problem: raise the frame rate to catch fast UI actions and token usage multiplies accordingly, so a long recording at high frame rate will not fit in a single call. Long video and high frame rate are in direct tension; both cannot be maximised in one shot.

**Cost and rate limits.** At roughly two dollars per million input tokens, a ten-minute clip at default resolution costs on the order of a third of a dollar per pass. But once a request crosses about 200,000 tokens, all tokens are billed at a higher long-context rate, so long single-shot videos become disproportionately expensive. Entry API tiers are also throttled (on the order of hundreds of requests per day), which is fine for a POC but a real constraint at inventory scale.

The fix for scale is the same pattern as the fix for accuracy, so the two do not trade off. Long recordings should be chunked, not gulped: the model supports video clipping by start and end offsets, which lets each segment be processed independently. In practice this means one cheap low-frame-rate pass over the whole recording to find the action skeleton, then high-frame-rate, high-resolution passes clipped to each detected action window. This bounds cost, sidesteps the context ceiling, and improves accuracy on the moments that matter. Because the same uploaded video is queried repeatedly, context caching should be used to avoid re-paying for the video on each segment query.

The deterministic alignment stage scales trivially — it operates on small structured traces, not on the video.

### Reference figures *(verify at build time)*

| Property | Value | Implication for the POC |
|---|---|---|
| Context window | ~1M tokens | Bounds single-shot video length |
| Video token cost | ~300 tokens/sec (default res); ~100/sec (low) | Drives per-recording cost and length limit |
| Single-shot video length | ~1 hour at default sampling | Long clips must be chunked, not sent whole |
| Default frame rate | 1 FPS (raisable to ~10 FPS) | Raise FPS for fast UI actions / transient messages |
| Long-context billing | All tokens at higher rate beyond ~200K | Keep individual requests under the threshold |
| Inline upload limit | ~20 MB | Use the File API for real recordings |

> These figures move between model revisions and should be re-confirmed against current documentation at the point of build. The architecture (chunk, two-pass, withhold the expectation) is the durable part; the exact numbers are not.

### Practical scaling notes

- Use the File API rather than inline video data for real recordings, and reuse the uploaded file across the two-pass segment queries.
- For volume beyond the POC, request higher quota tiers or run on a managed enterprise platform (for example Vertex AI) rather than the default API tier.
- Where unit cost eventually dominates, the self-hosted fallback in Section 10 (a screen-parsing model plus an open vision-language model) becomes the cost-optimisation path, with the frontier model reserved for ambiguous cases.

---

## 7. Build Sequence — Fastest Path to a Demo

The schema is the design, so it is built first. The recommended order is:

1. **Define the trace schema.** Design the Pydantic models for the expected trace and the observed trace. They must share the same shape so they can be aligned.
2. **Get one video to a clean observed trace.** Wire Stage 2 end to end on a single recording before anything else. This is the riskiest part; prove it first.
3. **Write the deterministic aligner.** Build and unit-test Stage 3 against hand-written trace pairs — this requires zero model calls.
4. **Add Stage 1.** Parse the expected scenario text into the expected trace.
5. **Build the Streamlit UI.** Upload video, paste expectation, render findings with timestamps.
6. **Add the cross-check pass.** Route assertion checks and low-confidence steps to the second model.

**Indicative effort:** a believable demo in one to two weeks; a polished POC in three to four weeks, for one capable engineer.

---

## 8. Accuracy Boosters

- **Two-pass sampling.** Run the video at low frame rate first to get the action skeleton, then re-run at higher frame rate clipped to each detected action window. Cheaper than high-rate over the whole clip, and sharper where it matters.
- **Confidence routing.** Ask the model for per-step confidence and timestamps; send low-confidence steps to the cross-check model rather than publishing them directly.
- **Assertions as grounded yes/no.** Treat assertion verification as "does text matching this expected message appear between these timestamps?" rather than a holistic judgement. This is where false passes hide.
- **Derive timestamps from clip offsets, not the model.** Model-reported timestamps are a known weak point and can drift or fall outside the clip. Because long recordings are already processed as clipped segments (Section 6), the reliable evidence timestamp is the segment's own offset, computed in code — not the time the model reports. This makes every finding's evidence pointer trustworthy at no extra cost.
- **A labelled evaluation set with both polarities.** Assemble five to ten recordings with known ground-truth traces. It must include at least one deliberately broken recording (the test claims to verify an error but the video shows success) AND at least one faithful recording that should pass cleanly. The clean recording is essential: a known failure mode is that, when the model is primed to expect a problem, it will invent one even when the recording is correct — a behaviour only detectable by testing a recording that should pass. Without both polarities, accuracy cannot be measured and the POC's claims cannot be defended.

---

## 9. Evaluation and Success Criteria

The POC succeeds if, on the labelled evaluation set, it:

- Correctly passes the faithful recordings that genuinely demonstrate their scenarios — without inventing a problem that is not there. This guards against the known tendency to fabricate a finding when the model is primed to expect one, and is just as important as catching real failures.
- Correctly flags the deliberately broken recording — catching a recording that does not do what its test case claims. This single case is the core demonstration of value.
- Attaches a correct, human-verifiable timestamp to each finding, derived from the processed clip rather than the model's reported time.
- Produces the same verdict on repeated runs of the same input.

The deliberately broken recording is the demonstration to lead with: a passing test whose recording quietly fails its own assertion is exactly the false-positive an independent audit layer exists to catch.

---

## 10. POC-to-Production Gap

Two honest caveats should be stated up front so the POC neither over-promises nor is dismissed unfairly:

- **Cost at scale.** Running a frontier video model on every recording across a full test inventory may be cost-prohibitive. When that point is reached, a self-hosted pipeline becomes the cost-optimisation path: a screen-parsing model such as OmniParser V2 combined with an open vision-language model (for example Qwen2.5-VL), with frontier models reserved for ambiguous cases. The POC validates accuracy; a later phase validates unit economics.
- **Calibration to real UIs.** Published grounding benchmarks are run on generic and professional software, not the specific interfaces under test here. Real-world accuracy will land below the headline numbers until the extractor is calibrated on actual recordings. Setting this expectation early is what turns a POC into a real engagement rather than a demo that is abandoned at the first difficult recording.

---

## 11. Open Questions for Confirmation

1. What form do the expected scenarios take — free text, structured steps, or Gherkin? This determines how much of Stage 1 can be made deterministic.
2. What is the typical length and resolution of a recording, and how fast are the on-screen actions? This drives the frame-rate strategy.
3. Is there any sanctioned constraint on which model provider may be used (data residency, an existing enterprise agreement with one cloud)? The architecture is provider-agnostic, but this decides which model backs the POC.
4. Can a small set of representative recordings — including at least one known-bad case — be provided to build the evaluation set?
5. What does a "pass" mean in this context — who is the authority on whether a recording is judged to faithfully demonstrate its scenario?

---

> **Note on verification:** Model names, capabilities, and benchmark figures referenced in this document were checked against current sources in May 2026. The fast-moving items (specific model versions and benchmark standings) should be re-confirmed at the point of build, as the architecture — not the specific model — is the durable part of this design.
