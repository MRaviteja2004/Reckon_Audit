"""Stage 1 — parse a scenario text into an ExpectedTrace via a single LLM call.

Supports three input formats:
- Free text (prose description of the scenario)
- Structured steps (numbered or bulleted list)
- Gherkin (Feature/Scenario with Given/When/Then/And/But steps)

For Gherkin the step lines are extracted deterministically and only those lines
are sent to the model — it never sees the Feature/Scenario headers, and the
scenario title is taken from the source rather than the model. The deterministic
path is used only when it is provably lossless: if the input contains any line
that is not a recognised Gherkin construct (prose, a data table, a doc string),
parsing falls back to the free-text path, which sends the whole text to the model
so nothing is dropped.

Assumes a single scenario per call — the ExpectedTrace schema models one
scenario. Inputs containing more than one Scenario block are rejected.
"""

import os
import re

import instructor
from pydantic import BaseModel, Field

from reckon.gateway import make_gateway_client
from reckon.macros.expand import apply_domain_macros
from reckon.macros.textutils import (
    _HUB_LANDING_RE,
    onboarding_milestones_from_text as _onboarding_milestones_from_text,
    strip_step_number_prefix as _strip_step_number_prefix,
)
from reckon.equivalence.citations import bind_expected_intent
from reckon.schema import ActionType, ExpectedStep, ExpectedTrace

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "claude-opus-5-5")

# temperature is omitted by default (None → not sent). Reproducibility (same input →
# same verdict) is a stated POC goal and is best served by temperature=0, but some
# gateway models only accept their default temperature and reject any explicit value.
# Callers targeting a model that supports it can pass temperature=0.0.
DEFAULT_TEMPERATURE: float | None = None

# Matches a Given/When/Then/And/But step line and captures the step text.
_GHERKIN_STEP_RE = re.compile(
    r"^\s*(?:Given|When|Then|And|But)\s+(.+)$",
    re.IGNORECASE,
)
# Feature / Background / Scenario / Scenario Outline header lines.
_GHERKIN_HEADER_RE = re.compile(
    r"^\s*(?:Feature:|Background:|Scenario(?:\s+Outline)?:)",
    re.IGNORECASE,
)
# Counts Scenario / Scenario Outline blocks to reject multi-scenario input.
_SCENARIO_HEADER_RE = re.compile(
    r"^\s*Scenario(?:\s+Outline)?:",
    re.IGNORECASE | re.MULTILINE,
)
_STRUCTURED_STEP_RE = re.compile(
    r"^\s*(?:\d+[.)]|[-*•])\s+.+$"
)

_ASSERTION_RULES = """
- assertion: the canonical value or condition being verified.

  IMPORTANT:
  - When the scenario contains EXACT QUOTED TEXT that must appear, store ONLY
    the text inside the quotes. Remove the surrounding quotes and wrapper
    phrases such as "is displayed", "should be displayed", "should appear",
    or "is visible".
  - This normalization applies ONLY when the assertion text is explicitly
    quoted.
  - For an UNQUOTED assertion, preserve the ENTIRE assertion exactly as
    written. Do NOT remove phrases such as "is displayed", "should appear",
    "is visible", "is enabled", or similar wording.
  - Example:
    Verify that "Invalid username" is displayed.
    -> assertion = Invalid username
  - Example:
    Verify that Invalid username is displayed.
    -> assertion = Invalid username is displayed.
  - Example:
    Verify that the dashboard is visible.
    -> assertion = the dashboard is visible

  Set this for any step that checks an outcome; leave empty for steps that
  only perform an action.
"""
_FREE_TEXT_SYSTEM = f"""\
You convert a test scenario description into an ordered list of structured steps.

Produce one step per distinct user action or assertion. If a single sentence \
describes several actions, split it into separate steps.

Fields:
- action: one of click, type_text, select, navigate, scroll, hover, \
assert_visible, assert_text, assert_state, other.
- target: the UI element label, field name, page, or URL the step acts on.
- input_value: ONLY a concrete, literal value the user actually types into a field \
or selects from a control — the exact characters, such as a quoted string \
("tomsmith") or a specific number. If the step only describes the value generically \
rather than giving it (e.g. "a password", "an invalid username", "some text", "a \
valid email"), leave input_value empty: a description of a value is not a typed \
value. Always leave empty for clicks, navigations, and assertions — never put \
asserted or expected text here.

{_ASSERTION_RULES}

- preconditions: state that must already hold before the step. Leave empty unless \
the scenario states it.
- raw_text: the original sentence or line this step came from.

Rules:
- Do not invent steps, targets, values, or assertions that are not in the input.
- Keep the steps in the order they appear in the input.
- A phrase such as "and wait for ... to complete" constrains the preceding action. \
Keep it in that action's raw_text; do not emit a separate `other` step for the wait.
- Use assert_text when the scenario verifies that specific text, a message, or \
quoted text is displayed.
- Use assert_visible when the scenario only verifies that a UI element is visible \
without checking specific text.

Login decomposition:
- When the scenario only says to log in (e.g. "Login to the driver app with valid \
credentials") without naming sub-actions, emit a single `other` (or one stated \
landing assertion). Do NOT invent password values, field clicks, or Next/Submit \
taps that the text does not mention.
- When the scenario explicitly describes login sub-steps, emit them separately: \
credential entry (`type_text`, with input_value only if a concrete value is given), \
password/login control actions (`click` / `type_text` as written), Next/Submit \
(`click`), and a logged-in or landing-state assertion only when the scenario \
states that outcome.

Onboarding decomposition:
- When the scenario lists required onboarding milestones (for example Security \
Number / SSN, Start Earn, vehicle, then Hub), emit one step per explicitly named \
milestone rather than one coarse `other` for the whole process.
- Include only milestones the scenario text supports. Do not assume every \
onboarding flow needs every milestone.
- A stated Hub / home landing is `assert_visible` / `assert_state` on that screen, \
not part of a bundled `other`.

Structured intent, when you can cite it from the step line:
- operation is one of activate, enter, choose, traverse, dismiss, assert.
- operation_text, kind_text, name, product, instance, and value must be exact \
substrings of that step line. Leave the whole intent empty when you cannot.

Select action versus selected state:
- Imperative wording ("Select …", "Choose …", "Tap … to select …") → `select` \
(or `click` when the scenario says tap/click), with input_value only when a \
concrete option value is given.
- State wording ("… is selected", "… are selected", "already selected") → \
`assert_state` (or `assert_visible` when only visibility of the selection is \
checked). Do not emit `select` for a state claim, and do not treat a preselected \
state as proof that the user performed a selection.
"""

_GHERKIN_SYSTEM = f"""\
You classify pre-extracted Gherkin step lines into structured steps. Each input \
line is one Given/When/Then/And/But step taken from a feature file. Produce \
exactly one output step per input line — never split one line into several steps \
and never merge several lines into one.

Fields:
- action: one of click, type_text, select, navigate, scroll, hover, \
assert_visible, assert_text, assert_state, other.
- target: the UI element, field name, page, or URL the step acts on.
- input_value: ONLY a concrete, literal value typed or selected — the exact \
characters (e.g. a quoted string or a specific identifier). Leave empty when the \
value is only described generically (e.g. "a password", "an invalid username", \
"some text"), and for any step that does not type a value — never put asserted or \
expected text here.
{_ASSERTION_RULES}
- preconditions: state required before the step; leave empty unless stated.
- raw_text: the original step line, unchanged.

Rules:
- Do not invent information that is not present in the line.
- Preserve the input order: one output step per input line.
- Use assert_text when the step verifies that specific text, a message, or quoted \
text is displayed.
- Use assert_visible when the step only verifies that a UI element is visible \
without checking specific text.
- Imperative "Select/Choose …" on one line → `select`. State wording \
("… is selected" / "already selected") on one line → `assert_state`. Do not \
emit `select` for a state-only claim.
- Optional intent: operation, operation_text, kind, kind_text, name, product, \
instance, and value. Every text must be an exact substring of the step line. \
Leave intent empty when you cannot cite it.
"""


class _LLMIntent(BaseModel):
    """Optional slot strings. Python binds them to the original scenario or drops them."""

    operation: str | None = None
    operation_text: str | None = None
    kind: str | None = None
    kind_text: str | None = None
    name: str | None = None
    product: str | None = None
    instance: str | None = None
    value: str | None = None


class _LLMStep(BaseModel):
    """A single step as returned by the LLM, before sequential reindexing."""

    action: ActionType
    target: str | None = Field(None, description="UI element, page, or URL")
    input_value: str | None = Field(
        None,
        description="Text typed or option selected (never asserted/expected text)",
    )
    assertion: str | None = Field(
        None,
        description="What must be true after this step (visible text, state, URL)",
    )
    preconditions: str | None = Field(
        None,
        description="State required before this step executes",
    )
    raw_text: str | None = Field(
        None,
        description="Original scenario sentence or Gherkin line",
    )
    intent: _LLMIntent | None = None


class _LLMTrace(BaseModel):
    """LLM response schema — ordered steps without strict index constraints."""

    scenario_title: str = Field(description="Short descriptive name for this scenario")
    steps: list[_LLMStep]


def make_openai_client() -> instructor.Instructor:
    """Create an instructor-patched client routed through Uber AI Gateway."""
    return instructor.from_openai(make_gateway_client())


def _looks_like_gherkin(text: str) -> bool:
    """Return True when the text contains a Gherkin header or step line.

    This is a loose router only — the safety decision (whether deterministic step
    extraction is lossless) is made separately by _is_lossless_gherkin.
    """
    return any(
        _GHERKIN_STEP_RE.match(line) or _GHERKIN_HEADER_RE.match(line)
        for line in text.splitlines()
    )


def _is_lossless_gherkin(text: str) -> bool:
    """Return True only if every non-blank line is a step, header, tag, or comment.

    When this holds, extracting just the step lines drops nothing meaningful
    (headers carry the title, tags/comments are non-behavioural). If any other
    line is present — prose, a data table row, a doc string — the deterministic
    path would silently lose it, so the caller must fall back to free text.
    """
    saw_step = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _GHERKIN_STEP_RE.match(line):
            saw_step = True
        elif _GHERKIN_HEADER_RE.match(line):
            continue
        elif stripped.startswith(("@", "#")):  # tags, comments
            continue
        else:
            return False
    return saw_step
def _extract_structured_raw_texts(text: str) -> list[str] | None:
    """Return original numbered/bulleted lines when the whole input is structured.

    Returns None when the input is not a lossless structured-step list.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    if not all(_STRUCTURED_STEP_RE.match(line) for line in lines):
        return None

    return lines


def _extract_gherkin_scenario_title(text: str) -> str | None:
    """Extract the Scenario (or Feature) title line from Gherkin text."""
    match = re.search(r"Scenario(?:\s+Outline)?:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    feature_match = re.search(r"Feature:\s*(.+)", text, re.IGNORECASE)
    if feature_match:
        return feature_match.group(1).strip()
    return None


def _extract_gherkin_steps(text: str) -> list[str]:
    """Return the raw step text of each Given/When/Then/And/But line in order."""
    return [
        match.group(1).strip()
        for line in text.splitlines()
        if (match := _GHERKIN_STEP_RE.match(line))
    ]


# Imperative selection verbs — presence means the step is an interaction.
_SELECT_IMPERATIVE_RE = re.compile(
    r"\b(?:select|choose|pick|tap|click)\b",
    re.IGNORECASE,
)
# State claims that an option is already / remains selected.
_SELECTED_STATE_RE = re.compile(
    r"\b(?:is|are|remains?|stays?)\s+selected\b|\balready\s+selected\b",
    re.IGNORECASE,
)


def _reclassify_select_versus_state(step: _LLMStep) -> _LLMStep:
    """Map state-only 'is selected' claims away from interactive ``select``."""
    text = _strip_step_number_prefix(step.raw_text or "")
    if not text:
        return step
    if step.action != ActionType.select:
        return step
    if not _SELECTED_STATE_RE.search(text):
        return step
    if _SELECT_IMPERATIVE_RE.search(text):
        # "Select X so that Y is selected" still describes an interaction.
        return step
    assertion = step.assertion or text
    return step.model_copy(
        update={
            "action": ActionType.assert_state,
            "input_value": None,
            "assertion": assertion,
        }
    )


def _expand_onboarding_other(
    step: _LLMStep,
    *,
    next_step: _LLMStep | None,
) -> list[_LLMStep]:
    """Split a coarse onboarding ``other`` into explicitly listed milestones."""
    if step.action != ActionType.other:
        return [step]
    text = step.raw_text or ""
    milestones = _onboarding_milestones_from_text(text)
    if milestones is None:
        return [step]

    # If the model already emitted one milestone per step (same raw_text on each),
    # do not expand again — that would triplicate Security Number / Start Earn / …
    target_key = (step.target or "").strip().lower()
    if target_key and any(
        target_key == milestone.strip().lower() for milestone in milestones
    ):
        return [step]

    expanded = [
        _LLMStep(
            action=ActionType.other,
            target=milestone,
            input_value=None,
            assertion=None,
            preconditions=step.preconditions,
            raw_text=step.raw_text,
        )
        for milestone in milestones
    ]

    if _HUB_LANDING_RE.search(text):
        next_targets_hub = bool(
            next_step is not None
            and next_step.target
            and re.search(r"\bhub\b", next_step.target, re.IGNORECASE)
        )
        if not next_targets_hub:
            expanded.append(
                _LLMStep(
                    action=ActionType.assert_visible,
                    target="Hub screen",
                    input_value=None,
                    assertion="user should land on Hub screen.",
                    preconditions=step.preconditions,
                    raw_text=step.raw_text,
                )
            )
    return expanded


def _refine_llm_steps(
    steps: list[_LLMStep],
    *,
    expand_onboarding: bool = True,
) -> list[_LLMStep]:
    """Apply deterministic Stage 1 refinements without inventing scenario content.

    When ``expand_onboarding`` is False (macros path), listed onboarding
    milestones are left for ``apply_domain_macros`` so expansion happens once.
    Select-versus-state reclassification always runs.
    """
    refined: list[_LLMStep] = []
    for index, step in enumerate(steps):
        reclassified = _reclassify_select_versus_state(step)
        if not expand_onboarding:
            refined.append(reclassified)
            continue
        next_step = steps[index + 1] if index + 1 < len(steps) else None
        refined.extend(
            _expand_onboarding_other(reclassified, next_step=next_step)
        )
    return refined


def _build_expected_trace(
    llm_trace: _LLMTrace,
    *,
    scenario_text: str = "",
    enable_macros: bool = False,
) -> ExpectedTrace:
    """Convert the raw LLM trace into a validated ExpectedTrace.

    Reindexes steps sequentially starting at 1 so the schema constraint is
    satisfied regardless of what the model emitted.

    Default path (``enable_macros=False``): legacy refine including hard-coded
    onboarding milestone expansion.

    Macros path (``enable_macros=True``): refine keeps select-versus-state and
    dismiss→click only; onboarding milestone expansion is performed exactly once
    by the Domain Macro registry.
    """
    refined = _refine_llm_steps(
        llm_trace.steps,
        expand_onboarding=not enable_macros,
    )
    steps = []
    for i, step in enumerate(refined, start=1):
        action = step.action
        classification_text = _strip_step_number_prefix(step.raw_text or "")
        if (
            action == ActionType.other
            and step.assertion is None
            and re.match(r"^(?:dismiss|close)\b", classification_text, re.IGNORECASE)
        ):
            action = ActionType.click

        steps.append(ExpectedStep(
            index=i,
            action=action,
            target=step.target,
            input_value=step.input_value,
            assertion=step.assertion,
            preconditions=step.preconditions,
            raw_text=step.raw_text,
            action_intent=_intent_for_step(scenario_text, step, i, len(refined)),
        ))

    macros_applied = []
    if enable_macros:
        steps, macros_applied = apply_domain_macros(steps, scenario_text=scenario_text)

    return ExpectedTrace(
        scenario_title=llm_trace.scenario_title,
        steps=steps,
        source_text=scenario_text or None,
        macros_applied=macros_applied,
    )


def _body_spans(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        match = pattern.match(line)
        if match and match.lastindex:
            spans.append((offset + match.start(1), offset + match.end(1)))
        offset += len(line)
    return spans


def _sentence_span(
    scenario_text: str,
    step: _LLMStep,
    index: int,
    count: int,
) -> tuple[int, int] | None:
    """Locate this step in the original input. Return None when the bound is unreliable."""
    if not scenario_text:
        return None
    gherkin = _body_spans(scenario_text, _GHERKIN_STEP_RE)
    numbered = _body_spans(
        scenario_text,
        re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.+)$"),
    )
    if len(gherkin) == count:
        return gherkin[index - 1]
    if len(numbered) == count:
        return numbered[index - 1]
    raw = step.raw_text or ""
    if raw and scenario_text.count(raw) == 1:
        start = scenario_text.find(raw)
        return start, start + len(raw)
    return None


def _intent_for_step(
    scenario_text: str,
    step: _LLMStep,
    index: int,
    count: int,
):
    intent = step.intent
    if (
        intent is None
        or not intent.operation
        or not intent.operation_text
        or not intent.kind
        or not intent.kind_text
    ):
        return None
    span = _sentence_span(scenario_text, step, index, count)
    if span is None:
        return None
    return bind_expected_intent(
        scenario_text,
        sentence_start=span[0],
        sentence_end=span[1],
        operation=intent.operation,
        operation_text=intent.operation_text,
        kind=intent.kind,
        kind_text=intent.kind_text,
        name=intent.name,
        product=intent.product,
        instance=intent.instance,
        value=intent.value,
    )


def _classify_steps(
    client: instructor.Instructor,
    model: str,
    temperature: float | None,
    system: str,
    user_content: str,
) -> _LLMTrace:
    """Make the single Stage 1 LLM call. temperature is sent only when not None, since
    some models reject any explicit temperature and accept only their default."""
    kwargs: dict = {
        "model": model,
        "response_model": _LLMTrace,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return client.chat.completions.create(**kwargs)


def parse_scenario(
    scenario_text: str,
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = OPENAI_MODEL,
    temperature: float | None = DEFAULT_TEMPERATURE,
    enable_macros: bool = False,
) -> ExpectedTrace:
    """Parse a scenario text into a validated ExpectedTrace.

    Accepts free text, structured step lists, and Gherkin. For clean Gherkin the
    step lines are extracted deterministically and only they are sent to the
    model; the scenario title comes from the source. Any input that is not
    losslessly reducible to Gherkin step lines is sent whole to the model so no
    content is dropped.

    A single LLM call is made and its output is validated against the Pydantic
    schema before being returned. Pass openai_client to inject a test double.

    Domain Macros are disabled by default (``enable_macros=False``) so existing
    Stage 1 callers keep the prior ExpectedTrace shape and step content.

    Raises ValueError on empty input or on input containing more than one
    Scenario block (the ExpectedTrace schema models a single scenario).
    """
    if not scenario_text or not scenario_text.strip():
        raise ValueError("scenario_text must not be empty")

    if len(_SCENARIO_HEADER_RE.findall(scenario_text)) > 1:
        raise ValueError(
            "multiple scenarios detected in one input; submit one scenario per call"
        )

    if openai_client is None:
        openai_client = make_openai_client()

    if _looks_like_gherkin(scenario_text) and _is_lossless_gherkin(scenario_text):
        return _parse_gherkin(
            scenario_text,
            openai_client,
            model,
            temperature,
            enable_macros=enable_macros,
        )
    return _parse_free_text(
        scenario_text,
        openai_client,
        model,
        temperature,
        enable_macros=enable_macros,
    )


def _parse_free_text(
    scenario_text: str,
    client: instructor.Instructor,
    model: str,
    temperature: float | None,
    *,
    enable_macros: bool = False,
) -> ExpectedTrace:
    llm_trace = _classify_steps(
        client,
        model,
        temperature,
        _FREE_TEXT_SYSTEM,
        scenario_text,
    )

    structured_raw_texts = _extract_structured_raw_texts(scenario_text)

    if (
        structured_raw_texts is not None
        and len(structured_raw_texts) == len(llm_trace.steps)
    ):
        for step, raw_text in zip(
            llm_trace.steps,
            structured_raw_texts,
        ):
            step.raw_text = raw_text

    return _build_expected_trace(
        llm_trace,
        scenario_text=scenario_text,
        enable_macros=enable_macros,
    )


def _parse_gherkin(
    scenario_text: str,
    client: instructor.Instructor,
    model: str,
    temperature: float | None,
    *,
    enable_macros: bool = False,
) -> ExpectedTrace:
    """Parse clean Gherkin: extract step lines deterministically, classify with LLM.

    The caller guarantees the input is lossless Gherkin (every non-blank line is a
    step, header, tag, or comment) and contains at least one step line.
    """
    title = _extract_gherkin_scenario_title(scenario_text) or "Scenario"
    step_lines = _extract_gherkin_steps(scenario_text)

    user_content = f"Scenario title: {title}\n\nSteps:\n" + "\n".join(step_lines)

    llm_trace = _classify_steps(client, model, temperature, _GHERKIN_SYSTEM, user_content)
    # Prefer the deterministically parsed title over whatever the model emitted.
    llm_trace.scenario_title = title
    return _build_expected_trace(
        llm_trace,
        scenario_text=scenario_text,
        enable_macros=enable_macros,
    )
