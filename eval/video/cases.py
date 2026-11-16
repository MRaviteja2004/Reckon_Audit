"""The synthetic evaluation corpus.

Each case is a timeline of page states with the action that produced each one, so the
ground truth is exact rather than annotated. The corpus is chosen to probe the places
this pipeline is most likely to be wrong, not to flatter it:

  - changes far smaller than a page repaint (a checkbox, one typed character)
  - events shorter than the sampling interval (a toast that comes and goes)
  - continuous motion with no semantics (a spinner) that must not become events
  - long idle stretches that must not produce events at all
  - scrolling, which discrete frames represent worst
  - a recording that genuinely does not demonstrate its scenario, which the audit must
    catch rather than pass
"""

from eval.video.render import Field, PageState

_PASSWORD = "SuperSecret!"


class Step:
    """One moment in a case timeline, with the action that produced it."""

    def __init__(
        self,
        at_ms: int,
        state: PageState,
        *,
        action: str | None = None,
        target: str | None = None,
        value: str | None = None,
        assertion: str | None = None,
        cursor: tuple[int, int] = (960, 620),
    ) -> None:
        self.at_ms = at_ms
        self.state = state
        self.action = action
        self.target = target
        self.value = value
        self.assertion = assertion
        self.cursor = cursor


class Case:
    """A synthetic recording plus everything needed to score a run against it."""

    def __init__(
        self,
        name: str,
        duration_ms: int,
        steps: list[Step],
        *,
        scenario: str | None = None,
        expect_finding: str | tuple[str, ...] | None = None,
        note: str = "",
    ) -> None:
        self.name = name
        self.duration_ms = duration_ms
        self.steps = steps
        self.scenario = scenario
        # The finding type(s) that would prove this recording's defect was caught; any
        # one of them suffices. None means the recording is faithful and no defect
        # finding may appear at all.
        self.expect_finding = (
            (expect_finding,) if isinstance(expect_finding, str) else expect_finding
        )
        self.note = note

    @property
    def change_times_ms(self) -> list[int]:
        """Times at which the rendered page changes — every step after the first."""
        return [step.at_ms for step in self.steps[1:]]

    @property
    def actions(self) -> list[Step]:
        return [step for step in self.steps if step.action is not None]


def _login(username: str = "", password_len: int = 0, focus: str | None = None,
           country: str = "", **kwargs) -> PageState:
    """A login page with the two standard fields, for the short probe cases."""
    fields = [
        Field("Username", username, focused=focus == "username"),
        Field("Password", "x" * password_len, masked=True, focused=focus == "password"),
    ]
    if country:
        fields.append(Field("Country", country))
    return PageState(
        url="app.example/login",
        heading="Login Page",
        fields=fields,
        button="Login",
        checkbox=kwargs.pop("checkbox", False),
        **kwargs,
    )


def _login_flow_steps(final_banner: str | None) -> list[Step]:
    """The shared happy-path login timeline; the final banner is what varies."""
    steps = [
        Step(0, _login(), cursor=(1200, 900)),
        Step(1400, _login(focus="username"), action="click", target="Username field", cursor=(700, 340)),
        Step(2100, _login(username="tom", focus="username"), action="type_text",
             target="Username field", value="tom", cursor=(700, 340)),
        Step(2900, _login(username="tomsmith", focus="username"), action="type_text",
             target="Username field", value="tomsmith", cursor=(700, 340)),
        Step(3900, _login(username="tomsmith", focus="password"), action="click",
             target="Password field", cursor=(700, 450)),
        Step(4700, _login(username="tomsmith", password_len=12, focus="password"),
             action="type_text", target="Password field", value=_PASSWORD, cursor=(700, 450)),
        Step(5800, _login(username="tomsmith", password_len=12), action="click",
             target="Login button", cursor=(620, 700)),
    ]
    if final_banner:
        steps.append(Step(7000, _login(banner=final_banner, banner_kind="success"),
                          action="assert_visible", target="success banner",
                          assertion=final_banner, cursor=(620, 700)))
    else:
        steps.append(Step(7000, _login(), cursor=(620, 700)))
    return steps


_LOGIN_SCENARIO = """\
1. Click the Username field and enter "tomsmith"
2. Click the Password field and enter the password
3. Click the Login button
4. The message "Welcome back, tomsmith!" should be visible
"""


def build_corpus() -> list[Case]:
    """Return every evaluation case."""
    return [
        Case(
            "login_faithful", 9000,
            _login_flow_steps("Welcome back, tomsmith!"),
            scenario=_LOGIN_SCENARIO,
            note="Happy path; the recording does demonstrate the scenario.",
        ),
        Case(
            "login_missing_assertion", 9000,
            _login_flow_steps(None),
            scenario=_LOGIN_SCENARIO,
            expect_finding=("assertion_not_observed", "missing_action"),
            note="ADVERSARIAL: identical actions, but the success message never renders.",
        ),
        Case(
            "transient_toast", 9000,
            [
                Step(0, _login(username="tomsmith")),
                Step(2000, _login(username="tomsmith", toast="Draft saved"),
                     action="assert_visible", target="toast", assertion="Draft saved"),
                Step(2450, _login(username="tomsmith")),
                Step(5000, _login(username="tomsmith", checkbox=True),
                     action="click", target="Remember me checkbox"),
            ],
            note="The toast is visible for 450 ms — near the 250 ms sampling interval.",
        ),
        Case(
            "modal_confirm", 10_000,
            [
                Step(0, _login(username="tomsmith")),
                Step(1800, _login(username="tomsmith", modal="Delete this account?"),
                     action="click", target="Delete button"),
                Step(4200, _login(username="tomsmith", modal="Delete this account?", toast="Deleting…"),
                     action="click", target="Confirm button"),
                Step(6000, _login(username="tomsmith", banner="Account deleted"),
                     action="assert_visible", target="banner", assertion="Account deleted"),
            ],
            note="Large overlay appearing and clearing.",
        ),
        Case(
            "dropdown_select", 10_000,
            [
                Step(0, _login(username="tomsmith")),
                Step(1900, _login(username="tomsmith", dropdown=["United Kingdom", "Germany", "Japan"]),
                     action="click", target="Country dropdown"),
                Step(4100, _login(username="tomsmith", country="Germany"),
                     action="select", target="Country dropdown", value="Germany"),
                Step(6500, _login(username="tomsmith", country="Germany", checkbox=True),
                     action="click", target="Remember me checkbox"),
            ],
            note="Option list opening, then a selected value.",
        ),
        Case(
            "scroll_table", 11_000,
            [
                Step(0, PageState(url="app.example/orders", heading="Orders", rows=14)),
                Step(2000, PageState(url="app.example/orders", heading="Orders", rows=14, scroll_y=260),
                     action="scroll", target="orders table"),
                Step(4000, PageState(url="app.example/orders", heading="Orders", rows=14, scroll_y=560),
                     action="scroll", target="orders table"),
                Step(6500, PageState(url="app.example/orders", heading="Orders", rows=14, scroll_y=860),
                     action="scroll", target="orders table"),
            ],
            note="Scrolling is the weakest case for discrete frames — tested honestly.",
        ),
        Case(
            "rapid_typing", 8000,
            [
                Step(0, _login(focus="username")),
                Step(1500, _login(username="t", focus="username"), action="type_text",
                     target="Username field", value="t"),
                Step(1750, _login(username="to", focus="username")),
                Step(2000, _login(username="tom", focus="username")),
                Step(2250, _login(username="toms", focus="username")),
                Step(2500, _login(username="tomsm", focus="username")),
                Step(2750, _login(username="tomsmith", focus="username")),
                Step(5500, _login(username="tomsmith", banner="Username taken"),
                     action="assert_visible", target="banner", assertion="Username taken"),
            ],
            note="Keystroke burst that debouncing should collapse, not report 6 times.",
        ),
        Case(
            "long_static", 14_000,
            [
                Step(0, _login(username="tomsmith", password_len=12)),
                Step(11_000, _login(username="tomsmith", password_len=12, banner="Session expired"),
                     action="assert_visible", target="banner", assertion="Session expired"),
            ],
            note="Eleven idle seconds must produce no events at all.",
        ),
        Case(
            "spinner_noise", 10_000,
            [
                Step(0, _login(username="tomsmith", spinner_phase=0)),
                Step(6000, _login(username="tomsmith", banner="Loaded 42 records", banner_kind="success"),
                     action="assert_visible", target="banner", assertion="Loaded 42 records"),
            ],
            note="A spinner animates throughout; only the banner is a real event.",
        ),
        Case(
            "checkbox_toggle", 9000,
            [
                Step(0, _login(username="tomsmith")),
                Step(2500, _login(username="tomsmith", checkbox=True),
                     action="click", target="Remember me checkbox"),
                Step(5500, _login(username="tomsmith", checkbox=False),
                     action="click", target="Remember me checkbox"),
            ],
            note="A 26x26 px change on a 1920x1080 frame — the smallest real event.",
        ),
    ]
