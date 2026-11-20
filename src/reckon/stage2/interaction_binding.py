"""Validate a model-reported interaction and bind it to one observation, or say why not.

An interaction is a structured description of something the model saw done to an
element. It is not proof by itself: the equivalence engine still checks every citation
against the observation and decides compatibility with the expected step.

Every citation ends up as an exact slice of the cited field. Matching the model's words
to that slice ignores runs of whitespace and nothing else.
"""

import logging
from dataclasses import dataclass

from reckon.equivalence.citations import bind_observed_interaction
from reckon.schema import (
    ActionType,
    InteractionParseStatus,
    InteractionRejectionReason,
    ObservedInteraction,
    ObservedInteractionDiagnostic,
)
from reckon.stage2.interaction_contract import (
    CITATION_SURFACES,
    INTERACTION_CONTRACT_VERSION,
    OBSERVED_KINDS,
    OBSERVED_OPERATIONS,
    OPERATIONS_FOR_ACTION,
    ReportedInteraction,
    loggable_token,
    normalise_token,
)

logger = logging.getLogger(__name__)

_REQUIRED_SLOTS = {
    "operation": "interaction_operation",
    "kind": "interaction_kind",
    "kind_text": "interaction_kind_text",
    "surface": "interaction_surface",
}
_CITED_SLOTS = {
    "operation_text": "interaction_operation_text",
    "kind_text": "interaction_kind_text",
    "name": "interaction_name",
    "product": "interaction_product",
    "instance": "interaction_instance",
    "value": "interaction_value",
}


@dataclass(frozen=True)
class InteractionOutcome:
    interaction: ObservedInteraction | None
    diagnostic: ObservedInteractionDiagnostic


def _offered(reported: ReportedInteraction, field: str) -> str | None:
    text = getattr(reported, field)
    return text if text is not None and text.strip() else None


def _comparable(text: str) -> tuple[str, list[int], list[int]]:
    """``text`` with whitespace runs collapsed, plus each kept character's source span."""
    kept: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for position, char in enumerate(text):
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


def _locate(surface: str, cited: str) -> str | InteractionRejectionReason:
    """The one slice of ``surface`` that ``cited`` refers to."""
    wanted = _comparable(cited)[0].strip()
    window, starts, ends = _comparable(surface)
    first = window.find(wanted)
    if first < 0:
        return InteractionRejectionReason.invalid_citation
    if window.find(wanted, first + 1) >= 0:
        return InteractionRejectionReason.ambiguous_citation
    return surface[starts[first]:ends[first + len(wanted) - 1]]


def _diagnostic(
    base: dict[str, str | None],
    status: InteractionParseStatus,
    reason: InteractionRejectionReason | None = None,
    slot: str | None = None,
) -> ObservedInteractionDiagnostic:
    return ObservedInteractionDiagnostic(
        contract_version=INTERACTION_CONTRACT_VERSION,
        status=status,
        reason=reason,
        slot=slot,
        **base,
    )


def _rejected(
    base: dict[str, str | None], reason: InteractionRejectionReason, slot: str | None = None
) -> InteractionOutcome:
    logger.info("interaction rejected reason=%s slot=%s", reason.value, slot)
    return InteractionOutcome(
        None, _diagnostic(base, InteractionParseStatus.rejected, reason, slot)
    )


def bind_reported_interaction(
    reported: ReportedInteraction,
    *,
    action: ActionType,
    surfaces: dict[str, str | None],
    frame_index: int,
) -> InteractionOutcome:
    """Bind ``reported`` to the observation whose fields are ``surfaces``.

    ``surfaces`` maps each citation surface name to that observation's text for it.
    """
    if not any(
        _offered(reported, field) for field in (*_REQUIRED_SLOTS.values(), *_CITED_SLOTS.values())
    ):
        return InteractionOutcome(
            None, _diagnostic({}, InteractionParseStatus.omitted)
        )

    operation = normalise_token(reported.interaction_operation)
    kind = normalise_token(reported.interaction_kind)
    surface = normalise_token(reported.interaction_surface)
    base = {
        "offered_operation": loggable_token(reported.interaction_operation),
        "offered_kind": loggable_token(reported.interaction_kind),
        "offered_surface": loggable_token(reported.interaction_surface),
    }
    for slot, field in _REQUIRED_SLOTS.items():
        if _offered(reported, field) is None:
            return _rejected(base, InteractionRejectionReason.missing_field, slot)
    if operation not in OBSERVED_OPERATIONS:
        return _rejected(base, InteractionRejectionReason.unsupported_operation, "operation")
    if kind not in OBSERVED_KINDS:
        return _rejected(base, InteractionRejectionReason.unsupported_kind, "kind")
    if surface not in CITATION_SURFACES:
        return _rejected(base, InteractionRejectionReason.unsupported_surface, "surface")
    if operation not in {op.value for op in OPERATIONS_FOR_ACTION.get(action, ())}:
        return _rejected(
            base, InteractionRejectionReason.operation_contradicts_action, "operation"
        )
    surface_text = surfaces.get(surface)
    if surface_text is None or not surface_text.strip():
        return _rejected(base, InteractionRejectionReason.surface_unavailable, "surface")

    resolved: dict[str, str | None] = {}
    for slot, field in _CITED_SLOTS.items():
        cited = _offered(reported, field)
        if cited is None:
            resolved[slot] = None
            continue
        found = _locate(surface_text, cited)
        if isinstance(found, InteractionRejectionReason):
            return _rejected(base, found, slot)
        resolved[slot] = found

    interaction = bind_observed_interaction(
        surface_text=surface_text,
        surface=surface,
        operation=operation,
        operation_text=resolved["operation_text"],
        kind=kind,
        kind_text=resolved["kind_text"],
        cited_frame_index=frame_index,
        name=resolved["name"],
        product=resolved["product"],
        instance=resolved["instance"],
        value=resolved["value"],
    )
    if interaction is None:
        return _rejected(base, InteractionRejectionReason.binding_failed)
    return InteractionOutcome(interaction, _diagnostic(base, InteractionParseStatus.bound))
