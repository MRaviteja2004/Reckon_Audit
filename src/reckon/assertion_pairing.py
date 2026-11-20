"""Which observations an assertion step may be paired with.

Sequence alignment scores a compatible action type and an absent target as a
pair, so an assertion step could claim any nearby assertion observation. That
observation then looked consumed to the step it actually proves. An assertion is
paired only with an observation that carries evidence for it, or that is
unmistakably about the same thing.
"""

from reckon.page_context import opening_navigation_target
from reckon.schema import ExpectedStep, ObservedAction
from reckon.tolerance import (
    ASSERTION_ACTIONS,
    assertion_satisfied,
    assertion_target_corroborated,
)

# No earlier navigation is excluded when asking whether a URL can name the screen:
# alignment has not yet decided which pair precedes this one. The audited verdict
# still applies the real floor in aligner._classify_pair.
NO_NAVIGATION_FLOOR_MS = 0


def assertion_pair_grounded(
    exp: ExpectedStep,
    obs: ObservedAction,
    *,
    observed_actions: list[ObservedAction] | None,
) -> bool:
    """False when an assertion step and an assertion observation have nothing in common.

    Evidence is judged by the same validators that decide a pass. A distinct target
    that the observation itself carries keeps the pair, so a screen that shows the
    wrong text is still reported against the step it belongs to.
    """
    if (
        exp.assertion is None
        or exp.action not in ASSERTION_ACTIONS
        or obs.action not in ASSERTION_ACTIONS
    ):
        return True
    context_target = opening_navigation_target(
        observed_actions or [], obs, floor_ms=NO_NAVIGATION_FLOOR_MS
    )
    if assertion_satisfied(
        exp.assertion,
        obs,
        expected_target=exp.target,
        context_target=context_target,
    ):
        return True
    return assertion_target_corroborated(exp.target, obs)
