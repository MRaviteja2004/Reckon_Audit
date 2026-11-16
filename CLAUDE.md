# Reckon — Development Guidelines

## Confirmed project decisions

- **Scenario format:** structured steps and free text. No Gherkin. Stage 1 always uses an LLM call to parse into `ExpectedTrace`.
- **Recording length:** typically over one minute. Recordings are never sent whole. `keyframes.py` finds the moments the screen changed, `sampler.py` turns those into padded windows, and one vision call is made per window. `merge_observed_chunks()` assembles the chunks into a full trace before alignment.
- **Provider:** OpenAI throughout — Stage 1 parsing, Stage 2 extraction, and the cross-check pass. No data residency constraints. Neither OpenAI nor Anthropic accepts video via API, so Stage 2 decodes frames locally with PyAV and sends stills; this is a better fit for screen recordings than native video anyway, since they are static but for discrete state changes.
- **Timestamps are never model-reported.** The extraction model names a *frame*, and the timestamp is looked up in code from that frame's decoder time. Nothing downstream should reintroduce a model-supplied time — the evidence pointer is what the audit is for.
- **Sample recordings:** to be provided by the client. `recording.mp4` (a 21 s Selenium login test) is the only real sample so far; detection thresholds in `keyframes.py` are calibrated against it and should be re-checked when more arrive.

## Python conventions

- Python 3.12+. No `from __future__ import annotations` — not needed here.
- Use `from typing import Optional` only when a value is genuinely absent. Prefer `str | None` syntax for new code.
- All public functions and methods must have type hints. No bare `def f(x)`.
- Prefer `pathlib.Path` over `os.path` for file operations.
- Use `python-dotenv` + `os.environ.get()` for config. Never hardcode keys or paths.
- Imports order: stdlib → third-party → internal. One blank line between groups.

## Pydantic

- All data that crosses a boundary (LLM response, API request/response, file load) must be validated through a Pydantic model. No raw dicts passed around.
- Put all field constraints (`ge`, `le`, `min_length`) in `Field()` — don't validate in application code what the schema can enforce.
- Validators (`@field_validator`, `@model_validator`) are for logic the schema can't express, not for re-implementing what `Field()` already does.
- Keep models in `schema.py`. Do not scatter Pydantic models across feature files.

## FastAPI

- One router per domain area (`routes/video.py`, `routes/audit.py`). Register routers in `main.py` — keep `main.py` thin.
- Use `response_model=` on every endpoint. Never return a raw dict from a route handler.
- Dependency injection (`Depends`) for shared resources (settings, clients). Do not instantiate clients inside route handlers.
- Async all the way: route handlers are `async def`, I/O calls use `await`. No blocking calls inside async handlers.
- HTTP status codes must be explicit. Do not rely on FastAPI defaulting to 200.
- Validation errors (422) are handled automatically by FastAPI/Pydantic — do not catch and re-raise them manually.

## Error handling

- Only catch exceptions you can meaningfully handle. Do not swallow exceptions with bare `except: pass`.
- Raise `ValueError` for bad input caught inside the domain layer. Let FastAPI turn it into an HTTP 422 at the boundary.
- Use `raise SystemExit(message)` only in top-level scripts, never in library code.

## File and module organisation

- One concern per file. `schema.py` holds models, `aligner.py` holds alignment logic, `extractor.py` holds video extraction. Do not mix.
- No file should exceed ~300 lines. If it grows past that, it is doing too many things.
- No circular imports. The dependency direction is: `routes` → `services` → `schema`. Nothing in `schema` imports from `routes` or `services`.
- `__init__.py` re-exports the public API of the package. Internal modules are not part of the public API unless explicitly re-exported.

## Testing

- Tests live in `tests/`. Mirror the `src/reckon/` structure: `tests/test_schema.py` tests `src/reckon/schema.py`.
- Every test function tests exactly one behaviour. Name it `test_<what>_<condition>` (e.g. `test_confidence_rejects_nan`).
- No `import` statements inside test functions — all imports at the top of the file.
- Fixtures (JSON files, sample data) go in `tests/fixtures/`. Use `example.invalid` for email addresses, never real-looking credentials.
- Tests must not make real network or API calls. Mock or skip any test that would hit a live service.
- Run `pytest` before every commit. All tests must be green.

## General engineering

- No premature abstraction. Three similar lines of code is better than a helper that turns out not to generalise.
- No dead code. Delete unused imports, functions, and variables immediately.
- No magic numbers. Name constants at the module level (`LOW_CONFIDENCE_THRESHOLD = 0.7`).
- No comments explaining what the code does — the code should speak for itself. Only comment on *why* something non-obvious is done.
- Secrets (API keys, tokens) go in `.env`, which is gitignored. `.env.example` documents the required keys with placeholder values.
- `if __name__ == "__main__"` guard on every script in `scripts/`.
