"""Validate a model-reported intent and bind it to the scenario, or say why not.

An intent is structured guidance about what a step asks for. It is never evidence that
anything happened: binding one only fills ExpectedStep.action_intent, and a step still
passes only on observed evidence checked elsewhere.

Every citation ends up as an exact slice of the original scenario. Matching the model's
words to that slice ignores whitespace runs, markdown emphasis and quotation marks, and
punctuation at the edges of the cited text. Nothing else is forgiven.
"""

import logging
from dataclasses import dataclass

from reckon.equivalence.citations import bind_expected_intent
from reckon.equivalence.grounding import IGNORED_CITATION_CHARACTERS
from reckon.intent_contract import (
    EXPECTED_INTENT_KINDS,
    EXPECTED_INTENT_OPERATIONS,
    ReportedIntent,
    loggable_token,
    normalise_token,
)
from reckon.schema import (
    ExpectedActionIntent,
    ExpectedStep,
    IntentDiagnostic,
    IntentParseStatus,
    IntentRejectionReason,
    StepSource,
)

logger = logging.getLogger(__name__)

_EDGE_PUNCTUATION = " .,;:!?"
_REQUIRED_SLOTS = ("operation", "operation_text", "kind", "kind_text")
_CITED_SLOTS = ("operation_text", "kind_text", "name", "product", "instance", "value")


@dataclass(frozen=True)
class IntentOutcome:
    intent: ExpectedActionIntent | None
    diagnostic: IntentDiagnostic


def _comparable(text: str) -> tuple[str, list[int], list[int]]:
    """Comparison form of ``text`` with each kept character's source start and end."""
    kept: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for position, char in enumerate(text):
        if char in IGNORED_CITATION_CHARACTERS:
            continue
        if char.isspace():
            if not kept:
                continue
            if kept[-1] == " ":
                ends[-1] = position + 1
                continue
            char = " "
        kept.append(char)
        starts.append(position)
        ends.append(position + 1)
    return "".join(kept), starts, ends


def _locate(
    scenario: str, cited: str, start: int, end: int
) -> tuple[int, int] | IntentRejectionReason:
    """The one source slice inside [start, end) that ``cited`` refers to."""
    wanted = _comparable(cited)[0].strip(_EDGE_PUNCTUATION)
    if not wanted:
        return IntentRejectionReason.invalid_citation
    window, starts, ends = _comparable(scenario[start:end])
    first = window.find(wanted)
    if first < 0:
        return IntentRejectionReason.invalid_citation
    if window.find(wanted, first + 1) >= 0:
        return IntentRejectionReason.ambiguous_citation
    return (
        start + starts[first],
        start + ends[first + len(wanted) - 1],
    )


def _rejected(
    base: dict[str, object], reason: IntentRejectionReason, slot: str | None = None
) -> IntentOutcome:
    diagnostic = IntentDiagnostic(
        **base, status=IntentParseStatus.rejected, reason=reason, slot=slot
    )
    logger.info(
        "intent rejected step=%s reason=%s slot=%s", diagnostic.step_index, reason.value, slot
    )
    return IntentOutcome(None, diagnostic)


def _omitted(step_index: int) -> IntentOutcome:
    return IntentOutcome(
        None, IntentDiagnostic(step_index=step_index, status=IntentParseStatus.omitted)
    )


def _offered(reported: ReportedIntent, slot: str) -> str | None:
    text = getattr(reported, slot)
    return text if text is not None and text.strip() else None


def bind_reported_intent(
    scenario: str,
    span: tuple[int, int] | None,
    reported: ReportedIntent | None,
    *,
    step_index: int,
) -> IntentOutcome:
    """Bind ``reported`` inside the step's sentence span, or return the reason it was refused.

    ``span`` is the step sentence's offsets in ``scenario``; None means it is unknown.
    """
    if reported is None or not any(
        _offered(reported, slot) for slot in (*_REQUIRED_SLOTS, *_CITED_SLOTS)
    ):
        return _omitted(step_index)

    operation = normalise_token(reported.operation)
    kind = normalise_token(reported.kind)
    base: dict[str, object] = {
        "step_index": step_index,
        "offered_operation": loggable_token(reported.operation),
        "offered_kind": loggable_token(reported.kind),
    }
    for slot in _REQUIRED_SLOTS:
        if _offered(reported, slot) is None:
            return _rejected(base, IntentRejectionReason.missing_field, slot)
    if operation not in EXPECTED_INTENT_OPERATIONS:
        return _rejected(base, IntentRejectionReason.unsupported_operation, "operation")
    if kind not in EXPECTED_INTENT_KINDS:
        return _rejected(base, IntentRejectionReason.unsupported_kind, "kind")
    if span is None:
        return _rejected(base, IntentRejectionReason.step_span_unavailable)

    resolved: dict[str, str | None] = {}
    for slot in _CITED_SLOTS:
        cited = _offered(reported, slot)
        if cited is None:
            resolved[slot] = None
            continue
        found = _locate(scenario, cited, span[0], span[1])
        if isinstance(found, IntentRejectionReason):
            return _rejected(base, found, slot)
        resolved[slot] = scenario[found[0]:found[1]]

    intent = bind_expected_intent(
        scenario,
        sentence_start=span[0],
        sentence_end=span[1],
        operation=operation,
        operation_text=resolved["operation_text"],
        kind=kind,
        kind_text=resolved["kind_text"],
        name=resolved["name"],
        product=resolved["product"],
        instance=resolved["instance"],
        value=resolved["value"],
    )
    if intent is None:
        return _rejected(base, IntentRejectionReason.binding_failed)
    return IntentOutcome(
        intent, IntentDiagnostic(**base, status=IntentParseStatus.bound)
    )


def diagnostics_for_final_steps(
    pending: list[tuple[str | None, IntentDiagnostic]], steps: list[ExpectedStep]
) -> list[IntentDiagnostic]:
    """Re-key diagnostics onto final step indexes after domain macros renumber the steps.

    Each scenario step takes the first unused diagnostic recorded for the same sentence.
    Steps a macro synthesised carry none.
    """
    unused = list(pending)
    diagnostics: list[IntentDiagnostic] = []
    for step in steps:
        if step.source != StepSource.scenario:
            continue
        for position, (raw_text, diagnostic) in enumerate(unused):
            if raw_text == step.raw_text:
                diagnostics.append(diagnostic.model_copy(update={"step_index": step.index}))
                del unused[position]
                break
    return diagnostics
