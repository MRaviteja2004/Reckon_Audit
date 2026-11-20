import pytest

from reckon.equivalence.grounding import slices_adjacent
from reckon.schema import SlotCitation


def _slice(source: str, text: str) -> SlotCitation:
    start = source.index(text)
    return SlotCitation(start=start, end=start + len(text), text=text)


@pytest.mark.parametrize(
    "source",
    [
        "Click the Sign in button.",
        "Click the **Sign in** button.",
        'Click the "Sign in" button.',
        "Click the `Sign in`   button.",
        "Click the \u201cSign in\u201d button.",
    ],
)
def test_slices_adjacent_ignores_whitespace_and_markup_between_slices(source: str):
    assert slices_adjacent(source, _slice(source, "Sign in"), _slice(source, "button"))


def test_slices_adjacent_accepts_either_order():
    source = "Click the **Sign in** button."

    assert slices_adjacent(source, _slice(source, "button"), _slice(source, "Sign in"))


def test_slices_adjacent_rejects_a_word_between_the_slices():
    source = "Open the application used by drivers."

    assert not slices_adjacent(
        source, _slice(source, "drivers"), _slice(source, "application")
    )


def test_slices_adjacent_rejects_punctuation_that_separates_clauses():
    source = "Click Sign in, button."

    assert not slices_adjacent(source, _slice(source, "Sign in"), _slice(source, "button"))


def test_slices_adjacent_rejects_overlapping_slices():
    source = "Click the Sign in button."

    assert not slices_adjacent(source, _slice(source, "Sign in"), _slice(source, "in button"))
