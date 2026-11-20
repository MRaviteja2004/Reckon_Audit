"""Which accepted Stage 3 proposals stand once observation ownership is applied.

A proposal can be accepted by its own validator and still lose: it would pass a step
on an observation another passed step owns. Whether a step is "already recovered"
has to be decided after that judgement. Deciding before it leaves a step with
neither the proposal that lost nor the independently valid one that was set aside.
"""

from reckon.ownership import as_owned_conflict, split_conflicting_claims
from reckon.schema import (
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    ExpectedTrace,
)


def _base_committed_steps(deterministic: AuditResult) -> set[int]:
    return {
        decision.expected_step_index
        for decision in deterministic.equivalence_decisions
        if decision.status == "committed" or decision.agent_status == "committed"
    }


def settle_proposals(
    deterministic: AuditResult,
    expected: ExpectedTrace,
    proposals: list[AssistProposalRecord],
) -> tuple[list[AssistProposalRecord], list[AssistProposalRecord]]:
    """Apply ownership to accepted proposals, in order.

    Returns every proposal, with those that lost an observation rejected, and the
    accepted proposals that stand. A step the base already committed is left alone.
    """
    committed = _base_committed_steps(deterministic)
    accepted = [
        proposal
        for proposal in proposals
        if proposal.status == AssistProposalStatus.accepted
        and proposal.expected_step_index not in committed
    ]
    standing, conflicting = split_conflicting_claims(deterministic, accepted, expected)
    lost = {id(proposal) for proposal in conflicting}
    settled = [
        as_owned_conflict(proposal) if id(proposal) in lost else proposal
        for proposal in proposals
    ]
    return settled, standing


def steps_with_standing_proposal(
    deterministic: AuditResult,
    expected: ExpectedTrace,
    proposals: list[AssistProposalRecord],
) -> set[int]:
    """Steps already recovered: an accepted proposal survives ownership, or the base committed."""
    _, standing = settle_proposals(deterministic, expected, proposals)
    committed = _base_committed_steps(deterministic)
    return {proposal.expected_step_index for proposal in standing} | {
        proposal.expected_step_index
        for proposal in proposals
        if proposal.status == AssistProposalStatus.accepted
        and proposal.expected_step_index in committed
    }
