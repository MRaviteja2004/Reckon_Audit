"""Deterministic Domain Macro expansion and provenance recording.

The LLM never invents macro members or override decisions. Expansion only uses
scenario / raw_text cues and the versioned registry.
"""

import re

from reckon.macros.registry import (
    AUTH_LOGIN_COARSE,
    DRIVER_ONBOARDING_MILESTONES,
    DomainMacro,
    MacroMember,
)
from reckon.macros.textutils import onboarding_milestones_from_text
from reckon.schema import (
    ActionType,
    ExpectedStep,
    MacroApplicationRecord,
    MacroMemberDecision,
    StepSource,
)


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _scenario_blob(steps: list[ExpectedStep], scenario_text: str) -> str:
    parts = [scenario_text or ""]
    for step in steps:
        if step.raw_text:
            parts.append(step.raw_text)
    return "\n".join(parts)


def _explicit_login_detail(text: str, macro: DomainMacro) -> bool:
    return any(pattern.search(text) for pattern in macro.block_expansion_patterns)


def _member_negated(member_id: str, text: str, macro: DomainMacro) -> bool:
    patterns = macro.negation_patterns.get(member_id, ())
    return any(pattern.search(text) for pattern in patterns)


def _step_covers_member(step: ExpectedStep, member: MacroMember) -> bool:
    """True when an existing step already expresses the member's intent."""
    target = _norm(step.target)
    member_target = _norm(member.target)
    if not target or not member_target:
        return False
    if target == member_target or member_target in target or target in member_target:
        return True
    if member.member_id == "hub_landing" and "hub" in target:
        return step.action in {
            ActionType.assert_visible,
            ActionType.assert_text,
            ActionType.assert_state,
        }
    return False


def _annotate(
    step: ExpectedStep,
    *,
    macro: DomainMacro,
    member: MacroMember,
) -> ExpectedStep:
    return step.model_copy(
        update={
            "source": StepSource.macro,
            "macro_id": macro.macro_id,
            "macro_version": macro.version,
            "obligation": member.obligation,
        }
    )


def _new_member_step(
    *,
    macro: DomainMacro,
    member: MacroMember,
    raw_text: str | None,
) -> ExpectedStep:
    return ExpectedStep(
        index=1,
        action=member.action,
        target=member.target,
        assertion=member.assertion,
        raw_text=raw_text,
        source=StepSource.macro,
        macro_id=macro.macro_id,
        macro_version=macro.version,
        obligation=member.obligation,
    )


def _is_login_step(step: ExpectedStep) -> bool:
    if step.action != ActionType.other:
        return False
    if step.target and re.search(r"login|log\s*in", step.target, re.I):
        return True
    if step.raw_text and re.search(r"\blog(?:\s*in|in)\b", step.raw_text, re.I):
        return True
    return False


def _is_coarse_onboarding_bundle(
    step: ExpectedStep,
    milestones: list[str],
) -> bool:
    """True for a single bundled onboarding other, not a per-milestone step."""
    if step.action != ActionType.other or not step.target:
        return False
    if "onboarding" not in step.target.lower():
        return False
    return not any(_norm(step.target) == m.strip().lower() for m in milestones)


def _apply_login_coarse(
    steps: list[ExpectedStep],
    scenario_text: str,
) -> tuple[list[ExpectedStep], MacroApplicationRecord | None]:
    macro = AUTH_LOGIN_COARSE
    blob = _scenario_blob(steps, scenario_text)
    if not any(pattern.search(blob) for pattern in macro.trigger_patterns):
        return steps, None

    decisions: list[MacroMemberDecision] = []
    out = list(steps)
    login_indices = [i for i, step in enumerate(out) if _is_login_step(step)]

    if not login_indices:
        decisions.append(
            MacroMemberDecision(
                member_id="coarse_login",
                decision="not_expanded_insufficient_detail",
                detail=(
                    "Login cue present but no login step to annotate; "
                    "sub-steps not invented"
                ),
            )
        )
        return out, MacroApplicationRecord(
            macro_id=macro.macro_id,
            version=macro.version,
            domain=macro.domain,
            source_text=scenario_text or None,
            decisions=decisions,
        )

    idx = login_indices[0]
    step = out[idx]
    source_text = step.raw_text or scenario_text
    # Credential detail must come from the login step itself or sibling
    # username/password/email type_text steps — not from an unrelated later
    # "Tap Next" (e.g. referral / onboarding continue).
    credential_siblings = [
        s
        for s in out
        if s.action == ActionType.type_text
        and (
            (s.target and re.search(r"password|email|username|credential", s.target, re.I))
            or (s.raw_text and re.search(r"\b(?:password|email|username|credential)\b", s.raw_text, re.I))
        )
    ]
    login_local_blob = "\n".join(
        part
        for part in (
            step.raw_text or "",
            *(s.raw_text or "" for s in credential_siblings),
        )
        if part
    )
    has_explicit_credentials = bool(credential_siblings) or _explicit_login_detail(
        login_local_blob, macro
    )
    if has_explicit_credentials:
        decisions.append(
            MacroMemberDecision(
                member_id="coarse_login",
                decision="skipped_explicit_cover",
                detail=(
                    "Scenario names credential actions; "
                    "coarse macro not applied"
                ),
            )
        )
    else:
        member = macro.members[0]
        out[idx] = _annotate(step, macro=macro, member=member)
        decisions.append(
            MacroMemberDecision(
                member_id="coarse_login",
                decision="annotated_existing",
                detail="Coarse login retained as a single other step",
            )
        )

    return out, MacroApplicationRecord(
        macro_id=macro.macro_id,
        version=macro.version,
        domain=macro.domain,
        source_text=source_text,
        decisions=decisions,
    )


def _find_onboarding_source(
    steps: list[ExpectedStep],
    scenario_text: str,
) -> tuple[str | None, ExpectedStep | None]:
    for step in steps:
        text = step.raw_text or ""
        if onboarding_milestones_from_text(text):
            return text, step
    for step in steps:
        if step.target and re.search(r"onboarding", step.target, re.I):
            return step.raw_text or scenario_text, step
    if onboarding_milestones_from_text(scenario_text):
        return scenario_text, None
    return scenario_text or None, None


def _member_listed(member: MacroMember, milestones: list[str], source_text: str) -> bool:
    if member.member_id == "hub_landing":
        return bool(member.cue_pattern and member.cue_pattern.search(source_text))
    if member.cue_pattern is None:
        return False
    for name in milestones:
        if member.cue_pattern.search(name):
            return True
    return False


def _apply_onboarding_milestones(
    steps: list[ExpectedStep],
    scenario_text: str,
) -> tuple[list[ExpectedStep], MacroApplicationRecord | None]:
    macro = DRIVER_ONBOARDING_MILESTONES
    blob = _scenario_blob(steps, scenario_text)
    if not any(pattern.search(blob) for pattern in macro.trigger_patterns):
        return steps, None

    source_text, _source_step = _find_onboarding_source(steps, scenario_text)
    milestones = onboarding_milestones_from_text(source_text or "")
    if milestones is None:
        return steps, MacroApplicationRecord(
            macro_id=macro.macro_id,
            version=macro.version,
            domain=macro.domain,
            source_text=source_text,
            decisions=[
                MacroMemberDecision(
                    member_id="*",
                    decision="not_expanded_insufficient_detail",
                    detail="Onboarding cue without an explicit milestone list",
                )
            ],
        )

    decisions: list[MacroMemberDecision] = []
    candidates: list[MacroMember] = []
    suppressed: list[MacroMember] = []
    for member in macro.members:
        if member.requires_cue and not _member_listed(member, milestones, source_text or ""):
            decisions.append(
                MacroMemberDecision(
                    member_id=member.member_id,
                    decision="not_expanded_insufficient_detail",
                    detail=f"Milestone '{member.target}' not listed in scenario",
                )
            )
            continue
        if _member_negated(member.member_id, source_text or "", macro):
            decisions.append(
                MacroMemberDecision(
                    member_id=member.member_id,
                    decision="suppressed_by_negation",
                    detail=f"Scenario negates '{member.target}'",
                )
            )
            suppressed.append(member)
            continue
        candidates.append(member)

    working = list(steps)
    coarse_indices = [
        i
        for i, step in enumerate(working)
        if _is_coarse_onboarding_bundle(step, milestones)
    ]
    insert_at = coarse_indices[0] if coarse_indices else None
    if insert_at is not None:
        working.pop(insert_at)

    # Explicit negation wins over a listed / previously expanded milestone.
    if suppressed:
        working = [
            step
            for step in working
            if not any(_step_covers_member(step, member) for member in suppressed)
        ]

    annotated = list(working)
    still_pending: list[MacroMember] = []
    for member in candidates:
        covered_at = next(
            (
                i
                for i, step in enumerate(annotated)
                if _step_covers_member(step, member)
            ),
            None,
        )
        if covered_at is None:
            still_pending.append(member)
            continue

        prior = annotated[covered_at]
        # Explicit Hub (or other) assertion already present: keep it, annotate
        # for provenance, and do not emit a duplicate member step.
        if member.member_id == "hub_landing":
            annotated[covered_at] = _annotate(prior, macro=macro, member=member)
            decisions.append(
                MacroMemberDecision(
                    member_id=member.member_id,
                    decision="skipped_explicit_cover",
                    detail="Explicit Hub step retained; macro did not duplicate",
                )
            )
        else:
            annotated[covered_at] = _annotate(prior, macro=macro, member=member)
            decisions.append(
                MacroMemberDecision(
                    member_id=member.member_id,
                    decision="annotated_existing",
                    detail=f"Annotated existing milestone '{member.target}'",
                )
            )

    insert_steps = [
        _new_member_step(macro=macro, member=member, raw_text=source_text)
        for member in still_pending
    ]
    for member in still_pending:
        decisions.append(
            MacroMemberDecision(
                member_id=member.member_id,
                decision="emitted",
                detail=f"Emitted milestone '{member.target}'",
            )
        )

    if insert_steps:
        if insert_at is not None:
            annotated[insert_at:insert_at] = insert_steps
        else:
            annotated.extend(insert_steps)

    reindexed = [
        step.model_copy(update={"index": i})
        for i, step in enumerate(annotated, start=1)
    ]
    return reindexed, MacroApplicationRecord(
        macro_id=macro.macro_id,
        version=macro.version,
        domain=macro.domain,
        source_text=source_text,
        decisions=decisions,
    )


def apply_domain_macros(
    steps: list[ExpectedStep],
    scenario_text: str = "",
) -> tuple[list[ExpectedStep], list[MacroApplicationRecord]]:
    """Apply registered Domain Macros deterministically; return steps + provenance.

    Safe to call repeatedly: outcomes depend only on steps + scenario_text +
    registry versions. Does not invent input values or login sub-interactions.
    """
    records: list[MacroApplicationRecord] = []
    current = [
        step.model_copy(update={"index": i})
        for i, step in enumerate(steps, start=1)
    ]

    current, login_record = _apply_login_coarse(current, scenario_text)
    if login_record is not None:
        records.append(login_record)

    current, onboarding_record = _apply_onboarding_milestones(current, scenario_text)
    if onboarding_record is not None:
        records.append(onboarding_record)

    current = [
        step.model_copy(update={"index": i})
        for i, step in enumerate(current, start=1)
    ]
    return current, records
