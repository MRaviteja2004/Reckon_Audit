"""Offset checks for structured slots. No semantic reading of the source text."""

from reckon.schema import SlotCitation

# Markdown emphasis and quotation marks carry no meaning in a scenario sentence, so they
# neither change what a citation refers to nor break adjacency between two citations.
IGNORED_CITATION_CHARACTERS = frozenset("*`\"'\u201c\u201d\u2018\u2019")


def citation_matches(source: str, citation: SlotCitation) -> bool:
    """True when the citation is an exact exclusive-end slice of ``source``."""
    if citation.start < 0 or citation.end > len(source) or citation.end <= citation.start:
        return False
    return source[citation.start:citation.end] == citation.text


def citation_inside(citation: SlotCitation, start: int, end: int) -> bool:
    return start <= citation.start and citation.end <= end


def slices_adjacent(source: str, left: SlotCitation, right: SlotCitation) -> bool:
    """True when only whitespace and markup separate the slices, in either order."""
    if left.end <= right.start:
        gap = source[left.end:right.start]
    elif right.end <= left.start:
        gap = source[right.end:left.start]
    else:
        return False
    return all(char.isspace() or char in IGNORED_CITATION_CHARACTERS for char in gap)


def locate_unique(source: str, text: str, start: int, end: int) -> SlotCitation | None:
    """Bind ``text`` to the only exact occurrence inside ``source[start:end]``."""
    if not text or start < 0 or end > len(source) or end <= start:
        return None
    window = source[start:end]
    found: list[int] = []
    cursor = 0
    while True:
        at = window.find(text, cursor)
        if at < 0:
            break
        found.append(at)
        cursor = at + 1
        if len(found) > 1:
            return None
    if len(found) != 1:
        return None
    absolute = start + found[0]
    return SlotCitation(start=absolute, end=absolute + len(text), text=text)
