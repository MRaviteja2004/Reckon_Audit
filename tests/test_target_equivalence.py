import pytest

from reckon.schema import ActionType, ExpectedStep, ObservedAction
from reckon.target_equivalence import target_exactly_equivalent


def _pair(
    action: ActionType, expected: str, observed: str, observed_action: ActionType | None = None
) -> tuple[ExpectedStep, ObservedAction]:
    return (
        ExpectedStep(index=1, action=action, target=expected),
        ObservedAction(
            index=1,
            timestamp_ms=1000,
            action=observed_action or action,
            target=observed,
            confidence=0.9,
        ),
    )


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        ("username field", "Username or email address"),
        ("email address input", "Username or email address"),
        ("phone", "Email/Phone"),
    ],
)
def test_target_exactly_equivalent_accepts_one_of_the_alternative_labels(
    expected: str, observed: str
):
    assert target_exactly_equivalent(*_pair(ActionType.type_text, expected, observed))


def test_target_exactly_equivalent_accepts_alternative_label_for_select():
    assert target_exactly_equivalent(*_pair(ActionType.select, "country", "Country or region"))


def test_target_exactly_equivalent_rejects_label_that_is_not_listed_as_an_alternative():
    assert not target_exactly_equivalent(
        *_pair(ActionType.type_text, "password field", "Username or email address")
    )


def test_target_exactly_equivalent_rejects_partial_alternative_overlap():
    assert not target_exactly_equivalent(
        *_pair(ActionType.type_text, "email field", "Username or email address")
    )


def test_target_exactly_equivalent_rejects_label_without_alternatives():
    assert not target_exactly_equivalent(
        *_pair(ActionType.type_text, "username field", "Username address")
    )


def test_target_exactly_equivalent_rejects_alternative_labels_on_click_targets():
    assert not target_exactly_equivalent(
        *_pair(ActionType.click, "Save", "Save or Cancel")
    )


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        ("browser search bar", "Address bar"),
        ("the address bar", "URL bar"),
        ("browser address bar", "Omnibox"),
        ("location bar", "address bar"),
    ],
)
def test_target_exactly_equivalent_accepts_browser_chrome_synonyms(expected: str, observed: str):
    assert target_exactly_equivalent(*_pair(ActionType.type_text, expected, observed))


def test_target_exactly_equivalent_rejects_bare_search_bar_as_browser_chrome():
    assert not target_exactly_equivalent(*_pair(ActionType.type_text, "search bar", "Address bar"))


def test_target_exactly_equivalent_rejects_in_page_search_against_browser_chrome():
    assert not target_exactly_equivalent(
        *_pair(ActionType.type_text, "browser search bar", "Search products bar")
    )


def test_target_exactly_equivalent_rejects_browser_chrome_across_action_types():
    assert not target_exactly_equivalent(
        *_pair(ActionType.type_text, "browser search bar", "Address bar", ActionType.click)
    )


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        ("Uber website", "uber.com/in/en/rider-home/"),
        ("Github site", "https://github.com"),
        ("Uber homepage", "m.uber.com/go/home"),
        ("Stack Overflow site", "www.stackoverflow.com/questions"),
    ],
)
def test_target_exactly_equivalent_accepts_site_navigation_to_brand_domain(
    expected: str, observed: str
):
    assert target_exactly_equivalent(*_pair(ActionType.navigate, expected, observed))


@pytest.mark.parametrize(
    "observed",
    [
        "uber.com.example.invalid",
        "uber-eats.com",
        "ubereats.com",
        "user@uber.example.invalid",
        "uber",
    ],
)
def test_target_exactly_equivalent_rejects_navigation_to_other_domains(observed: str):
    assert not target_exactly_equivalent(*_pair(ActionType.navigate, "Uber website", observed))


def test_target_exactly_equivalent_rejects_named_page_navigation():
    assert not target_exactly_equivalent(
        *_pair(ActionType.navigate, "Uber login page", "uber.com/login")
    )


def test_target_exactly_equivalent_rejects_site_noun_without_brand():
    assert not target_exactly_equivalent(*_pair(ActionType.navigate, "website", "uber.com"))


def test_target_exactly_equivalent_rejects_brand_without_site_noun():
    assert not target_exactly_equivalent(*_pair(ActionType.navigate, "Uber", "uber.com"))
