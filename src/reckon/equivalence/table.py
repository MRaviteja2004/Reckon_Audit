"""Versioned type-realization rows.

Rows relate an expected operation and object kind to an observed operation and
element kind. They contain no application names, verbs, or scenario ids.
Shipped rows are immutable. A new or edited row requires a new ``TABLE_VERSION``
and its own positive and negative tests.
"""

from dataclasses import dataclass

from reckon.schema import ElementKind, IntentOperation, ObjectKind

TABLE_VERSION = "1"
APPLICATION_ICON_RULE_ID = "application_icon"


@dataclass(frozen=True)
class TypeRealization:
    """One declarative realization. ``None`` on a same-kind row means 'equal to the other side'."""

    rule_id: str
    expected_operation: IntentOperation | None
    expected_kind: ObjectKind | None
    observed_operation: IntentOperation | None
    observed_kind: ElementKind | None
    same_kind: bool = False


TYPE_REALIZATIONS: tuple[TypeRealization, ...] = (
    TypeRealization(
        rule_id="same_kind",
        expected_operation=None,
        expected_kind=None,
        observed_operation=None,
        observed_kind=None,
        same_kind=True,
    ),
    TypeRealization(
        rule_id=APPLICATION_ICON_RULE_ID,
        expected_operation=IntentOperation.traverse,
        expected_kind=ObjectKind.application,
        observed_operation=IntentOperation.activate,
        observed_kind=ElementKind.icon,
    ),
    TypeRealization(
        rule_id="control_button",
        expected_operation=IntentOperation.activate,
        expected_kind=ObjectKind.control,
        observed_operation=IntentOperation.activate,
        observed_kind=ElementKind.button,
    ),
    TypeRealization(
        rule_id="control_link",
        expected_operation=IntentOperation.activate,
        expected_kind=ObjectKind.control,
        observed_operation=IntentOperation.activate,
        observed_kind=ElementKind.link,
    ),
)


def _validate_table(rows: tuple[TypeRealization, ...]) -> None:
    if not TABLE_VERSION:
        raise RuntimeError("type realization table requires a version")
    seen: set[str] = set()
    operations = set(IntentOperation)
    object_kinds = set(ObjectKind)
    element_kinds = set(ElementKind)
    for row in rows:
        if row.rule_id in seen:
            raise RuntimeError(f"duplicate type realization row {row.rule_id}")
        seen.add(row.rule_id)
        if row.expected_operation is not None and row.expected_operation not in operations:
            raise RuntimeError(f"{row.rule_id} has an invalid expected operation")
        if row.observed_operation is not None and row.observed_operation not in operations:
            raise RuntimeError(f"{row.rule_id} has an invalid observed operation")
        if row.expected_kind is not None and row.expected_kind not in object_kinds:
            raise RuntimeError(f"{row.rule_id} has an invalid expected kind")
        if row.observed_kind is not None and row.observed_kind not in element_kinds:
            raise RuntimeError(f"{row.rule_id} has an invalid observed kind")
        if row.same_kind:
            continue
        if None in (
            row.expected_operation,
            row.expected_kind,
            row.observed_operation,
            row.observed_kind,
        ):
            raise RuntimeError(f"{row.rule_id} is missing a tuple cell")


_validate_table(TYPE_REALIZATIONS)


def matching_rule(
    expected_operation: IntentOperation,
    expected_kind: ObjectKind,
    observed_operation: IntentOperation,
    observed_kind: ElementKind,
) -> TypeRealization | None:
    """Return the first row that realizes this pair, or None."""
    if (
        expected_operation == IntentOperation.assert_
        or observed_operation == IntentOperation.assert_
    ):
        return None
    for row in TYPE_REALIZATIONS:
        if row.same_kind:
            if (
                expected_operation == observed_operation
                and expected_operation != IntentOperation.unspecified
                and expected_kind.value == observed_kind.value
                and expected_kind != ObjectKind.unspecified
            ):
                return row
            continue
        if (
            row.expected_operation == expected_operation
            and row.expected_kind == expected_kind
            and row.observed_operation == observed_operation
            and row.observed_kind == observed_kind
        ):
            return row
    return None
