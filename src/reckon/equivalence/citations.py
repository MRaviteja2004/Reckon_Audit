"""Bind model-supplied slot strings to exact slices of the original scenario.

Citations are not invented. A slot that does not occur exactly once inside the
step sentence leaves the whole intent unset.
"""

from reckon.equivalence.grounding import locate_unique
from reckon.schema import (
    CitationSurface,
    ElementKind,
    ExpectedActionIntent,
    ExpectedObject,
    IntentOperation,
    ObjectKind,
    ObservedElement,
    ObservedInteraction,
    SlotCitation,
)


def _operation(value: str | None) -> IntentOperation | None:
    if value is None:
        return None
    try:
        return IntentOperation(value)
    except ValueError:
        return None


def _object_kind(value: str | None) -> ObjectKind | None:
    if value is None:
        return None
    try:
        return ObjectKind(value)
    except ValueError:
        return None


def _element_kind(value: str | None) -> ElementKind | None:
    if value is None:
        return None
    try:
        return ElementKind(value)
    except ValueError:
        return None


def bind_expected_intent(
    scenario: str,
    *,
    sentence_start: int,
    sentence_end: int,
    operation: str,
    operation_text: str,
    kind: str,
    kind_text: str,
    name: str | None = None,
    product: str | None = None,
    instance: str | None = None,
    value: str | None = None,
) -> ExpectedActionIntent | None:
    """Return an intent whose every string slot is an exact unique slice, or None."""
    parsed_operation = _operation(operation)
    parsed_kind = _object_kind(kind)
    if (
        parsed_operation is None
        or parsed_operation == IntentOperation.unspecified
        or parsed_kind is None
        or parsed_kind == ObjectKind.unspecified
        or sentence_end <= sentence_start
        or sentence_end > len(scenario)
    ):
        return None
    operation_citation = locate_unique(scenario, operation_text, sentence_start, sentence_end)
    kind_citation = locate_unique(scenario, kind_text, sentence_start, sentence_end)
    if operation_citation is None or kind_citation is None:
        return None
    slots: dict[str, SlotCitation | None] = {}
    for field, text in (
        ("name", name),
        ("product", product),
        ("instance", instance),
        ("value", value),
    ):
        if text is None:
            slots[field] = None
            continue
        citation = locate_unique(scenario, text, sentence_start, sentence_end)
        if citation is None:
            return None
        slots[field] = citation
    return ExpectedActionIntent(
        schema_version="1",
        operation=parsed_operation,
        operation_citation=operation_citation,
        object=ExpectedObject(
            kind=parsed_kind,
            kind_citation=kind_citation,
            name=None if slots["name"] is None else slots["name"].text,
            name_citation=slots["name"],
            product=None if slots["product"] is None else slots["product"].text,
            product_citation=slots["product"],
            instance=None if slots["instance"] is None else slots["instance"].text,
            instance_citation=slots["instance"],
            value=None if slots["value"] is None else slots["value"].text,
            value_citation=slots["value"],
        ),
        sentence_start=sentence_start,
        sentence_end=sentence_end,
    )


def bind_observed_interaction(
    *,
    surface_text: str,
    surface: str,
    operation: str,
    kind: str,
    kind_text: str,
    cited_frame_index: int,
    name: str | None = None,
    product: str | None = None,
    instance: str | None = None,
    value: str | None = None,
    operation_text: str | None = None,
) -> ObservedInteraction | None:
    """Bind observed slots into one surface. ``screen_description`` is not accepted."""
    if surface not in {"target", "input_value", "visible_text"}:
        return None
    parsed_operation = _operation(operation)
    parsed_kind = _element_kind(kind)
    if (
        parsed_operation is None
        or parsed_operation == IntentOperation.unspecified
        or parsed_kind is None
        or parsed_kind == ElementKind.unspecified
        or not surface_text
    ):
        return None
    kind_citation = locate_unique(surface_text, kind_text, 0, len(surface_text))
    if kind_citation is None:
        return None
    operation_citation = None
    if operation_text is not None:
        operation_citation = locate_unique(surface_text, operation_text, 0, len(surface_text))
        if operation_citation is None:
            return None
    bound: dict[str, SlotCitation | None] = {}
    for field, text in (
        ("name", name),
        ("product", product),
        ("instance", instance),
        ("value", value),
    ):
        if text is None:
            bound[field] = None
            continue
        citation = locate_unique(surface_text, text, 0, len(surface_text))
        if citation is None:
            return None
        bound[field] = citation
    return ObservedInteraction(
        schema_version="1",
        operation=parsed_operation,
        operation_citation=operation_citation,
        element=ObservedElement(
            kind=parsed_kind,
            kind_citation=kind_citation,
            name=None if bound["name"] is None else bound["name"].text,
            name_citation=bound["name"],
            product=None if bound["product"] is None else bound["product"].text,
            product_citation=bound["product"],
            instance=None if bound["instance"] is None else bound["instance"].text,
            instance_citation=bound["instance"],
            value=None if bound["value"] is None else bound["value"].text,
            value_citation=bound["value"],
        ),
        citation_surface=CitationSurface(surface),
        cited_frame_index=cited_frame_index,
    )
