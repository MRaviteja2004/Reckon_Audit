"""A long, multi-page recording — the shape of a real client capture.

The short probe cases isolate one hazard each. This is the opposite test: a hundred
seconds across five pages with thirty-odd actions, where the audit has to keep its place
through navigation, scrolling, a modal, and two forms rather than reason about a single
event in isolation.

Four variants share one scenario, which is what makes the corpus discriminating:

  faithful          every step happens as written
  no_confirmation   the refund is issued but the confirmation never renders — the
                    classic false pass, and the reason this tool exists
  wrong_amount      everything happens, but a different figure is typed into the refund
                    box: £8.90 rather than £89.00
  opposite_outcome  the banner renders and reads "Refund failed" — the flow completed
                    and reported the opposite of what the scenario claims

The last two are the hardest and the most realistic. A recording that is wholly wrong is
easy to catch. A recording that is wrong in one field, or in one word, is what ships.
"""

from eval.video.cases import Case, Step
from eval.video.render import Field, PageState

_NAV = ["Dashboard", "Orders", "Customers", "Settings"]
_EMAIL = "admin@shop.test"
_ORDER = "4211"
_CORRECT_AMOUNT = "89.00"
_WRONG_AMOUNT = "8.90"


def _login_page(email: str = "", password_len: int = 0, focus: str | None = None,
                banner: str | None = None) -> PageState:
    return PageState(
        url="shop.test/sign-in", heading="Sign in", banner=banner, banner_kind="success",
        fields=[
            Field("Email", email, focused=focus == "email"),
            Field("Password", "x" * password_len, masked=True, focused=focus == "password"),
        ],
        button="Sign in",
    )


def _dashboard(banner: str | None = None, toast: str | None = None) -> PageState:
    return PageState(
        url="shop.test/dashboard", heading="Dashboard", nav=_NAV, active_nav="Dashboard",
        banner=banner, banner_kind="success", toast=toast,
        fields=[Field("Search", "")], button="Go",
    )


def _orders(scroll_y: int = 0, query: str = "", focus: bool = False,
            rows: int = 16, toast: str | None = None) -> PageState:
    return PageState(
        url="shop.test/orders", heading="Orders", nav=_NAV, active_nav="Orders",
        rows=rows, scroll_y=scroll_y, toast=toast,
    ) if not query and not focus else PageState(
        url="shop.test/orders", heading="Orders", nav=_NAV, active_nav="Orders",
        fields=[Field("Filter orders", query, focused=focus)], button="Search", toast=toast,
    )


def _order_detail(amount: str = "", focus: bool = False, modal: str | None = None,
                  banner: str | None = None, banner_kind: str = "success",
                  toast: str | None = None) -> PageState:
    return PageState(
        url=f"shop.test/orders/{_ORDER}", heading=f"Order #{_ORDER}", nav=_NAV,
        active_nav="Orders", banner=banner, banner_kind=banner_kind, modal=modal,
        modal_body="The customer will be refunded immediately.", toast=toast,
        fields=[
            Field("Customer", "tomsmith@example.invalid"),
            Field("Order total", f"£{_CORRECT_AMOUNT}"),
            Field("Refund amount", amount, focused=focus),
        ],
        button="Issue refund",
    )


SCENARIO = """\
1. Enter "admin@shop.test" in the Email field
2. Enter the password
3. Click the Sign in button
4. The Dashboard should be visible
5. Click the Orders tab
6. Scroll down the orders list
7. Enter "4211" in the Filter orders field
8. Click the Search button
9. Open order #4211
10. Enter "89.00" in the Refund amount field
11. Click the Issue refund button
12. Click the Confirm button in the dialog
13. A confirmation reading "Refund issued" should be visible
"""


def _steps(*, confirmation: bool, amount: str, outcome: str = "Refund issued") -> list[Step]:
    """The full journey; the two variants differ only in `confirmation` and `amount`."""
    return [
        Step(0, _login_page(), cursor=(1200, 900)),
        Step(1800, _login_page(focus="email"), action="click", target="Email field",
             cursor=(700, 330)),
        Step(3200, _login_page(email="admin@", focus="email"), cursor=(700, 330)),
        Step(4600, _login_page(email=_EMAIL, focus="email"), action="type_text",
             target="Email field", value=_EMAIL, cursor=(700, 330)),
        Step(6400, _login_page(email=_EMAIL, focus="password"), action="click",
             target="Password field", cursor=(700, 440)),
        Step(8000, _login_page(email=_EMAIL, password_len=11, focus="password"),
             action="type_text", target="Password field", value="hunter2hunt",
             cursor=(700, 440)),
        Step(10_200, _login_page(email=_EMAIL, password_len=11), action="click",
             target="Sign in button", cursor=(620, 700)),
        Step(12_400, _dashboard(banner="Signed in as admin@shop.test"),
             action="assert_visible", target="Dashboard", assertion="Dashboard",
             cursor=(620, 700)),

        Step(16_000, _dashboard(), cursor=(900, 200)),
        Step(18_500, _orders(), action="click", target="Orders tab", cursor=(760, 130)),
        Step(22_000, _orders(scroll_y=300), action="scroll", target="orders list",
             cursor=(960, 600)),
        Step(25_500, _orders(scroll_y=640), action="scroll", target="orders list",
             cursor=(960, 600)),
        Step(29_000, _orders(scroll_y=980), cursor=(960, 600)),

        Step(33_000, _orders(focus=True), action="click", target="Filter orders field",
             cursor=(700, 330)),
        Step(35_500, _orders(query="42", focus=True), cursor=(700, 330)),
        Step(37_500, _orders(query=_ORDER, focus=True), action="type_text",
             target="Filter orders field", value=_ORDER, cursor=(700, 330)),
        Step(40_000, _orders(query=_ORDER, rows=3), action="click", target="Search button",
             cursor=(620, 480)),
        Step(44_000, _orders(rows=3), cursor=(700, 300)),

        Step(48_000, _order_detail(), action="click", target=f"order #{_ORDER} row",
             cursor=(700, 300)),
        Step(52_000, _order_detail(focus=True), action="click",
             target="Refund amount field", cursor=(700, 560)),
        Step(55_000, _order_detail(amount="89", focus=True), cursor=(700, 560)),
        Step(57_500, _order_detail(amount=amount, focus=True), action="type_text",
             target="Refund amount field", value=amount, cursor=(700, 560)),

        Step(62_000, _order_detail(amount=amount, modal="Issue refund?"),
             action="click", target="Issue refund button", cursor=(620, 700)),
        Step(66_500, _order_detail(amount=amount, modal="Issue refund?", toast="Processing…"),
             action="click", target="Confirm button", cursor=(760, 620)),
        Step(70_000, _order_detail(amount=amount, toast="Processing…"), cursor=(760, 620)),

        Step(74_000, (
            _order_detail(amount=amount, banner=outcome,
                          banner_kind="success" if outcome == "Refund issued" else "error")
            if confirmation else _order_detail(amount=amount)
        ), **({
            "action": "assert_visible", "target": "confirmation banner",
            "assertion": outcome,
        } if confirmation else {}), cursor=(760, 620)),

        Step(80_000, _orders(rows=3, toast="Refund recorded" if confirmation else None),
             action="click", target="Orders tab", cursor=(760, 130)),
        Step(85_000, _orders(rows=3), cursor=(960, 500)),
    ]


def build_long_cases() -> list[Case]:
    """The three long-flow variants."""
    return [
        Case(
            "long_refund_faithful", 90_000,
            _steps(confirmation=True, amount=_CORRECT_AMOUNT),
            scenario=SCENARIO,
            note="90s, 5 pages, 22 actions; the recording matches the scenario.",
        ),
        Case(
            "long_refund_no_confirmation", 90_000,
            _steps(confirmation=False, amount=_CORRECT_AMOUNT),
            scenario=SCENARIO,
            expect_finding=("assertion_not_observed", "missing_action"),
            note="ADVERSARIAL: refund issued but 'Refund issued' never renders.",
        ),
        Case(
            "long_refund_wrong_amount", 90_000,
            _steps(confirmation=True, amount=_WRONG_AMOUNT),
            scenario=SCENARIO,
            expect_finding="wrong_input_value",
            note="ADVERSARIAL: one field wrong — £8.90 refunded, not £89.00.",
        ),
        Case(
            "long_refund_opposite_outcome", 90_000,
            _steps(confirmation=True, amount=_CORRECT_AMOUNT, outcome="Refund failed"),
            scenario=SCENARIO,
            expect_finding="assertion_not_observed",
            note="ADVERSARIAL: banner reads 'Refund failed' — the opposite outcome.",
        ),
    ]
