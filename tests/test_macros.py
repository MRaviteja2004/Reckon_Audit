"""Domain Macro registry — identification, expansion, provenance, determinism."""

from reckon.macros import apply_domain_macros, get_macro, list_macros
from reckon.macros.registry import AUTH_LOGIN_COARSE, DRIVER_ONBOARDING_MILESTONES
from reckon.parser import (
    _LLMStep,
    _LLMTrace,
    _build_expected_trace,
    _expand_onboarding_other,
    _refine_llm_steps,
)
from reckon.schema import (
    ActionType,
    ExpectedStep,
    ExpectedTrace,
    StepObligation,
    StepSource,
)


def _step(
    index: int,
    action: ActionType,
    target: str | None,
    *,
    raw_text: str | None = None,
    assertion: str | None = None,
    input_value: str | None = None,
) -> ExpectedStep:
    return ExpectedStep(
        index=index,
        action=action,
        target=target,
        assertion=assertion,
        input_value=input_value,
        raw_text=raw_text,
    )


def _norm_target(step: ExpectedStep) -> str:
    return (step.target or "").strip().lower()


def _austin_style_llm_trace() -> _LLMTrace:
    return _LLMTrace(
        scenario_title="Austin onboarding",
        steps=[
            _LLMStep(
                action=ActionType.other,
                target="Driver app login",
                raw_text="1. Login to Driver app with valid credential.",
            ),
            _LLMStep(
                action=ActionType.other,
                target="Onboarding process (Security Number, Start Earn, vehicle)",
                raw_text=(
                    "2. After the Onboarding process like- Security Number, "
                    "Start Earn, vehicle user should land on Hub screen."
                ),
            ),
            _LLMStep(
                action=ActionType.assert_visible,
                target="Hub screen",
                assertion="user should land on Hub screen.",
                preconditions=(
                    "Onboarding process (Security Number, Start Earn, vehicle) "
                    "completed"
                ),
                raw_text=(
                    "2. After the Onboarding process like- Security Number, "
                    "Start Earn, vehicle user should land on Hub screen."
                ),
            ),
        ],
    )


def _austin_scenario_text() -> str:
    return (
        "1. Login to Driver app with valid credential.\n"
        "2. After the Onboarding process like- Security Number, "
        "Start Earn, vehicle user should land on Hub screen."
    )


def test_macro_registry_ids_and_versions():
    login = get_macro("auth.login.coarse")
    onboarding = get_macro("driver.onboarding.milestones")
    assert login.version == "1.0.0"
    assert login.domain == "auth"
    assert onboarding.version == "1.0.0"
    assert onboarding.domain == "driver_onboarding"
    assert {macro.macro_id for macro in list_macros()} == {
        AUTH_LOGIN_COARSE.macro_id,
        DRIVER_ONBOARDING_MILESTONES.macro_id,
    }


def test_macros_off_preserves_legacy_onboarding_refine():
    llm_trace = _austin_style_llm_trace()
    legacy = _build_expected_trace(llm_trace, enable_macros=False)
    assert legacy.macros_applied == []
    assert [step.target for step in legacy.steps] == [
        "Driver app login",
        "Security Number",
        "Start Earn",
        "vehicle",
        "Hub screen",
    ]
    assert all(step.macro_id is None for step in legacy.steps)
    assert all(step.source == StepSource.scenario for step in legacy.steps)


def test_macros_on_skips_legacy_onboarding_refine_inside_refine_llm_steps():
    llm_trace = _austin_style_llm_trace()
    refined = _refine_llm_steps(llm_trace.steps, expand_onboarding=False)
    assert len(refined) == 3
    assert refined[1].target == (
        "Onboarding process (Security Number, Start Earn, vehicle)"
    )
    # Contrast: legacy refine still expands when asked.
    legacy_refined = _refine_llm_steps(llm_trace.steps, expand_onboarding=True)
    assert [step.target for step in legacy_refined] == [
        "Driver app login",
        "Security Number",
        "Start Earn",
        "vehicle",
        "Hub screen",
    ]


def test_macros_on_expands_coarse_onboarding_without_legacy_helper():
    llm_trace = _austin_style_llm_trace()
    # Prove expand helper is not needed: only macros create milestones.
    enabled = _build_expected_trace(
        llm_trace,
        scenario_text=_austin_scenario_text(),
        enable_macros=True,
    )
    assert [step.target for step in enabled.steps] == [
        "Driver app login",
        "Security Number",
        "Start Earn",
        "vehicle",
        "Hub screen",
    ]
    onboarding = next(
        r for r in enabled.macros_applied if r.macro_id == "driver.onboarding.milestones"
    )
    by_member = {d.member_id: d.decision for d in onboarding.decisions}
    assert by_member["security_number"] == "emitted"
    assert by_member["start_earn"] == "emitted"
    assert by_member["vehicle"] == "emitted"
    assert by_member["hub_landing"] == "skipped_explicit_cover"
    assert enabled.steps[1].macro_id == "driver.onboarding.milestones"
    assert enabled.steps[1].source == StepSource.macro


def test_macros_on_does_not_duplicate_milestone_expansion():
    llm_trace = _austin_style_llm_trace()
    enabled = _build_expected_trace(
        llm_trace,
        scenario_text=_austin_scenario_text(),
        enable_macros=True,
    )
    targets = [step.target for step in enabled.steps]
    assert targets.count("Security Number") == 1
    assert targets.count("Start Earn") == 1
    assert targets.count("vehicle") == 1
    assert targets.count("Hub screen") == 1
    # Applying macros again on already-expanded steps only annotates.
    again_steps, again_records = apply_domain_macros(
        enabled.steps,
        scenario_text=_austin_scenario_text(),
    )
    assert [step.target for step in again_steps] == targets
    onboarding = next(
        r for r in again_records if r.macro_id == "driver.onboarding.milestones"
    )
    assert {d.decision for d in onboarding.decisions} <= {
        "annotated_existing",
        "skipped_explicit_cover",
    }
    assert "emitted" not in {d.decision for d in onboarding.decisions}


def test_bare_login_remains_coarse_other():
    steps = [
        _step(
            1,
            ActionType.other,
            "Driver app login",
            raw_text="1. Login to Driver app with valid credential.",
        )
    ]
    out, records = apply_domain_macros(steps, scenario_text=steps[0].raw_text or "")
    assert len(out) == 1
    assert out[0].action == ActionType.other
    assert out[0].target == "Driver app login"
    assert out[0].input_value is None
    assert out[0].source == StepSource.macro
    assert out[0].macro_id == "auth.login.coarse"
    assert out[0].obligation == StepObligation.mandatory
    assert records[0].decisions[0].decision == "annotated_existing"
    assert not any(
        step.action in {ActionType.type_text, ActionType.click} for step in out
    )


def test_explicit_login_details_remain_intact():
    scenario = (
        "1. Enter email user@example.invalid\n"
        "2. Enter password secret\n"
        "3. Tap Next"
    )
    steps = [
        _step(
            1,
            ActionType.type_text,
            "Email",
            input_value="user@example.invalid",
            raw_text="1. Enter email user@example.invalid",
        ),
        _step(
            2,
            ActionType.type_text,
            "Password",
            input_value="secret",
            raw_text="2. Enter password secret",
        ),
        _step(3, ActionType.click, "Next", raw_text="3. Tap Next"),
        _step(
            4,
            ActionType.other,
            "Driver app login",
            raw_text="Login to Driver app with valid credential.",
        ),
    ]
    out, records = apply_domain_macros(steps, scenario_text=scenario)
    assert [step.action for step in out] == [
        ActionType.type_text,
        ActionType.type_text,
        ActionType.click,
        ActionType.other,
    ]
    assert out[1].input_value == "secret"
    login_record = next(r for r in records if r.macro_id == "auth.login.coarse")
    assert login_record.decisions[0].decision == "skipped_explicit_cover"
    assert out[3].source == StepSource.scenario
    assert out[3].macro_id is None


def test_onboarding_milestone_expansion_emits_new_steps():
    raw = (
        "2. After the Onboarding process like- Security Number, Start Earn, "
        "vehicle user should land on Hub screen."
    )
    steps = [
        _step(
            1,
            ActionType.other,
            "Onboarding process (Security Number, Start Earn, vehicle)",
            raw_text=raw,
        )
    ]
    out, records = apply_domain_macros(steps, scenario_text=raw)
    assert [step.target for step in out] == [
        "Security Number",
        "Start Earn",
        "vehicle",
        "Hub screen",
    ]
    assert out[-1].action == ActionType.assert_visible
    assert all(step.source == StepSource.macro for step in out)
    assert {d.member_id for d in records[0].decisions if d.decision == "emitted"} == {
        "security_number",
        "start_earn",
        "vehicle",
        "hub_landing",
    }


def test_duplicate_hub_assertion_prevention():
    raw = (
        "After the Onboarding process like- Security Number, Start Earn, "
        "vehicle user should land on Hub screen."
    )
    steps = [
        _step(1, ActionType.other, "Security Number", raw_text=raw),
        _step(2, ActionType.other, "Start Earn", raw_text=raw),
        _step(3, ActionType.other, "vehicle", raw_text=raw),
        _step(
            4,
            ActionType.assert_visible,
            "Hub screen",
            assertion="user should land on Hub screen.",
            raw_text=raw,
        ),
    ]
    out, records = apply_domain_macros(steps, scenario_text=raw)
    hub_steps = [step for step in out if step.target and "hub" in step.target.lower()]
    assert len(hub_steps) == 1
    by_member = {d.member_id: d.decision for d in records[0].decisions}
    assert by_member["hub_landing"] == "skipped_explicit_cover"
    assert by_member["security_number"] == "annotated_existing"
    assert by_member["start_earn"] == "annotated_existing"
    assert by_member["vehicle"] == "annotated_existing"


def test_explicit_vehicle_negation_suppresses_vehicle():
    raw = (
        "After the Onboarding process like- Security Number, Start Earn, "
        "vehicle user should land on Hub screen without vehicle onboarding."
    )
    # Coarse bundle path: macros must expand and omit vehicle.
    coarse = [
        _step(
            1,
            ActionType.other,
            "Onboarding process (Security Number, Start Earn, vehicle)",
            raw_text=raw,
        )
    ]
    out, records = apply_domain_macros(coarse, scenario_text=raw)
    assert not any(_norm_target(step) == "vehicle" for step in out)
    assert [step.target for step in out] == [
        "Security Number",
        "Start Earn",
        "Hub screen",
    ]
    assert any(d.decision == "suppressed_by_negation" for d in records[0].decisions)


def test_explicit_values_preserved():
    steps = [
        _step(
            1,
            ActionType.select,
            "Location",
            input_value="Austin",
            raw_text="2. Set a Location to Austin.",
        ),
        _step(
            2,
            ActionType.type_text,
            "Referral code",
            input_value="1",
            raw_text="3. Enter Referral code as 1",
        ),
    ]
    out, _records = apply_domain_macros(
        steps,
        scenario_text="Login then Set a Location to Austin. Enter Referral code as 1",
    )
    assert out[0].input_value == "Austin"
    assert out[1].input_value == "1"


def test_no_invented_passwords_or_ssns():
    raw = (
        "After the Onboarding process like- Security Number, Start Earn, "
        "vehicle user should land on Hub screen."
    )
    steps = [
        _step(
            1,
            ActionType.other,
            "Driver app login",
            raw_text="1. Login to Driver app with valid credential.",
        ),
        _step(
            2,
            ActionType.other,
            "Onboarding process (Security Number, Start Earn, vehicle)",
            raw_text=raw,
        ),
    ]
    out, _records = apply_domain_macros(
        steps,
        scenario_text=steps[0].raw_text + "\n" + raw,
    )
    for step in out:
        assert step.input_value is None
        assert step.action != ActionType.type_text


def test_select_versus_state_identical_on_both_paths():
    llm_trace = _LLMTrace(
        scenario_title="Preselected",
        steps=[
            _LLMStep(
                action=ActionType.select,
                target="Rides and Delivery with car",
                input_value="Rides and Delivery with car",
                raw_text='Verify that "Rides and Delivery with car" is selected.',
            ),
            _LLMStep(
                action=ActionType.select,
                target="Rides and Delivery with car",
                input_value="Rides and Delivery with car",
                raw_text='Choose "Rides and Delivery with car" and tap Continue.',
            ),
        ],
    )
    default_trace = _build_expected_trace(llm_trace, enable_macros=False)
    macro_trace = _build_expected_trace(llm_trace, enable_macros=True)
    assert default_trace.steps[0].action == ActionType.assert_state
    assert macro_trace.steps[0].action == ActionType.assert_state
    assert default_trace.steps[1].action == ActionType.select
    assert macro_trace.steps[1].action == ActionType.select
    assert [
        (s.action, s.target, s.input_value, s.assertion) for s in default_trace.steps
    ] == [
        (s.action, s.target, s.input_value, s.assertion) for s in macro_trace.steps
    ]


def test_provenance_distinguishes_emitted_from_annotated():
    raw = (
        "After the Onboarding process like- Security Number, Start Earn, "
        "vehicle user should land on Hub screen."
    )
    emitted_out, emitted_records = apply_domain_macros(
        [
            _step(
                1,
                ActionType.other,
                "Onboarding process (Security Number, Start Earn, vehicle)",
                raw_text=raw,
            )
        ],
        scenario_text=raw,
    )
    assert all(d.decision == "emitted" for d in emitted_records[0].decisions)

    annotated_out, annotated_records = apply_domain_macros(
        emitted_out,
        scenario_text=raw,
    )
    assert [s.target for s in annotated_out] == [s.target for s in emitted_out]
    decisions = {d.member_id: d.decision for d in annotated_records[0].decisions}
    assert decisions["security_number"] == "annotated_existing"
    assert decisions["hub_landing"] == "skipped_explicit_cover"
    assert "emitted" not in decisions.values()


def test_old_expected_trace_fixtures_without_metadata_still_load():
    payload = {
        "scenario_title": "Legacy fixture",
        "steps": [
            {
                "index": 1,
                "action": "click",
                "target": "Submit",
                "input_value": None,
                "assertion": None,
                "preconditions": None,
                "raw_text": "Click Submit",
            }
        ],
    }
    trace = ExpectedTrace.model_validate(payload)
    assert trace.macros_applied == []
    assert trace.steps[0].source == StepSource.scenario
    assert trace.steps[0].macro_id is None
    assert trace.steps[0].obligation is None


def test_macro_expansion_is_deterministic_across_repeated_runs():
    raw = _austin_scenario_text()
    llm_trace = _austin_style_llm_trace()
    first = _build_expected_trace(llm_trace, scenario_text=raw, enable_macros=True)
    second = _build_expected_trace(llm_trace, scenario_text=raw, enable_macros=True)
    assert first.model_dump() == second.model_dump()


def test_legacy_expand_helper_still_available_for_macros_off_path():
    step = _LLMStep(
        action=ActionType.other,
        target="Onboarding process (Security Number, Start Earn, vehicle)",
        raw_text=(
            "After the Onboarding process like- Security Number, Start Earn, "
            "vehicle user should land on Hub screen."
        ),
    )
    expanded = _expand_onboarding_other(step, next_step=None)
    assert [item.target for item in expanded] == [
        "Security Number",
        "Start Earn",
        "vehicle",
        "Hub screen",
    ]
