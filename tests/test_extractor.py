from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from PIL import Image

from reckon.extractor import (
    EXTRACTION_SYSTEM,
    _call_with_retry,
    _ExtractedAction,
    _ExtractedActions,
    _resolve_frame_index,
    extract_window,
)
from reckon.schema import (
    ActionType,
    InteractionParseStatus,
    InteractionRejectionReason,
    ObservedTrace,
)

_VIDEO = Path("recording.mp4")
_SOURCE = "recording.mp4"


def _image() -> Image.Image:
    return Image.new("RGB", (8, 6), color=(10, 20, 30))


def _samples(timestamps_ms: list[int]) -> list[tuple[int, Image.Image]]:
    return [(ts, _image()) for ts in timestamps_ms]


def _action(frame_index: int = 1, **overrides) -> _ExtractedAction:
    fields = {
        "frame_index": frame_index,
        "action": ActionType.click,
        "target": "Submit",
        "input_value": None,
        "assertion_evidence": None,
        "confidence": 0.9,
        "screen_description": None,
    }
    fields.update(overrides)
    return _ExtractedAction(**fields)


def _client(actions: list[_ExtractedAction]) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _ExtractedActions(actions=actions)
    return client


def _api_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


# ---------------------------------------------------------------------------
# Extraction prompt — the separation that keeps the audit honest
# ---------------------------------------------------------------------------

def test_extraction_prompt_does_not_mention_expected_scenario():
    lowered = EXTRACTION_SYSTEM.lower()
    assert "expected" not in lowered or "any expectation" in lowered
    assert "scenario" not in lowered


def test_extraction_prompt_instructs_describe_only():
    assert "only what is clearly visible" in EXTRACTION_SYSTEM.lower()


def test_extraction_prompt_requires_frame_index():
    assert "frame_index" in EXTRACTION_SYSTEM


def test_extraction_prompt_forbids_guessing_a_time():
    assert "never guess a time" in EXTRACTION_SYSTEM.lower()


def test_extraction_prompt_instructs_assertion_evidence():
    assert "assertion_evidence" in EXTRACTION_SYSTEM


# ---------------------------------------------------------------------------
# _resolve_frame_index — the code-side bound on a model-reported position
# ---------------------------------------------------------------------------

def test_resolve_frame_index_within_range_unchanged():
    assert _resolve_frame_index(2, 5) == 2


def test_resolve_frame_index_above_range_clamped_to_last():
    assert _resolve_frame_index(9, 5) == 5


def test_resolve_frame_index_below_range_clamped_to_first():
    assert _resolve_frame_index(0, 5) == 1


# ---------------------------------------------------------------------------
# extract_window — evidence anchoring
# ---------------------------------------------------------------------------

def test_extract_window_returns_observed_trace():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0, 500])):
        trace = extract_window(client, _VIDEO, _SOURCE, [0, 500])
    assert isinstance(trace, ObservedTrace)
    assert len(trace.actions) == 1


def test_extract_window_timestamp_comes_from_frame_not_model():
    client = _client([_action(frame_index=2)])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([1000, 2500, 4000])):
        trace = extract_window(client, _VIDEO, _SOURCE, [1000, 2500, 4000])
    assert trace.actions[0].timestamp_ms == 2500


def test_extract_window_timestamps_are_clip_relative():
    client = _client([_action(frame_index=2)])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([1000, 2500])):
        trace = extract_window(
            client, _VIDEO, _SOURCE, [1000, 2500], clip_offset_ms=1000, clip_end_ms=3000
        )
    assert trace.actions[0].timestamp_ms == 1500


def test_extract_window_out_of_range_frame_index_clamps_to_last_frame():
    client = _client([_action(frame_index=99)])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0, 500, 900])):
        trace = extract_window(client, _VIDEO, _SOURCE, [0, 500, 900])
    assert trace.actions[0].timestamp_ms == 900


def test_extract_window_source_file_comes_from_code_not_model():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0])):
        trace = extract_window(client, _VIDEO, "authoritative.mp4", [0])
    assert trace.source_file == "authoritative.mp4"


def test_extract_window_clip_bounds_come_from_code_not_model():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([2000])):
        trace = extract_window(
            client, _VIDEO, _SOURCE, [2000], clip_offset_ms=2000, clip_end_ms=5000
        )
    assert trace.clip_offset_ms == 2000
    assert trace.clip_end_ms == 5000


def test_extract_window_reindexes_actions_sequentially():
    client = _client([_action(frame_index=1), _action(frame_index=2), _action(frame_index=1)])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0, 500])):
        trace = extract_window(client, _VIDEO, _SOURCE, [0, 500])
    assert [a.index for a in trace.actions] == [1, 2, 3]


def test_extract_window_preserves_model_action_fields():
    client = _client([_action(action=ActionType.type_text, target="Email", input_value="a@b.c")])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0])):
        trace = extract_window(client, _VIDEO, _SOURCE, [0])
    action = trace.actions[0]
    assert action.action == ActionType.type_text
    assert action.target == "Email"
    assert action.input_value == "a@b.c"


def test_extract_window_no_actions_returns_empty_trace():
    client = _client([])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0])):
        trace = extract_window(client, _VIDEO, _SOURCE, [0])
    assert trace.actions == []


# ---------------------------------------------------------------------------
# extract_window — request construction
# ---------------------------------------------------------------------------

def test_extract_window_labels_each_frame_in_order():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0, 500, 900])):
        extract_window(client, _VIDEO, _SOURCE, [0, 500, 900])
    content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    labels = [part["text"] for part in content if part["type"] == "text"]
    assert "Frame 1:" in labels and "Frame 2:" in labels and "Frame 3:" in labels


def test_extract_window_sends_one_image_per_frame():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0, 500, 900])):
        extract_window(client, _VIDEO, _SOURCE, [0, 500, 900])
    content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 3


def test_extract_window_requests_high_detail_frames():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0])):
        extract_window(client, _VIDEO, _SOURCE, [0])
    content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    image = next(part for part in content if part["type"] == "image_url")
    assert image["image_url"]["detail"] == "high"


def test_extract_window_sends_frames_as_jpeg_data_urls():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0])):
        extract_window(client, _VIDEO, _SOURCE, [0])
    content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    image = next(part for part in content if part["type"] == "image_url")
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_extract_window_omits_expected_scenario_from_system_prompt():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0])):
        extract_window(client, _VIDEO, _SOURCE, [0])
    system = client.chat.completions.create.call_args.kwargs["messages"][0]
    assert system["content"] == EXTRACTION_SYSTEM


def test_extract_window_does_not_send_temperature():
    client = _client([_action()])
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0])):
        extract_window(client, _VIDEO, _SOURCE, [0])
    assert "temperature" not in client.chat.completions.create.call_args.kwargs


# ---------------------------------------------------------------------------
# extract_window — input validation
# ---------------------------------------------------------------------------

def test_extract_window_rejects_empty_frame_list():
    with pytest.raises(ValueError, match="must not be empty"):
        extract_window(_client([]), _VIDEO, _SOURCE, [])


def test_extract_window_rejects_clip_end_not_after_offset():
    with pytest.raises(ValueError, match="must be greater than"):
        extract_window(
            _client([]), _VIDEO, _SOURCE, [1000], clip_offset_ms=2000, clip_end_ms=2000
        )


def test_extract_window_rejects_unknown_detail_level():
    with pytest.raises(ValueError, match="detail must be one of"):
        extract_window(_client([]), _VIDEO, _SOURCE, [0], detail="ultra")


def test_extract_window_rejects_frame_before_clip_offset():
    with pytest.raises(ValueError, match="at or after clip_offset_ms"):
        extract_window(_client([]), _VIDEO, _SOURCE, [500], clip_offset_ms=1000)


def test_extract_window_rejects_frame_after_clip_end():
    with pytest.raises(ValueError, match="at or before clip_end_ms"):
        extract_window(_client([]), _VIDEO, _SOURCE, [9000], clip_end_ms=5000)


# ---------------------------------------------------------------------------
# _call_with_retry
# ---------------------------------------------------------------------------

def test_call_with_retry_returns_on_first_success():
    client = _client([_action()])
    result = _call_with_retry(client, "m", [])
    assert isinstance(result, _ExtractedActions)
    assert client.chat.completions.create.call_count == 1


def test_call_with_retry_retries_then_succeeds_on_503():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _api_error(503),
        _ExtractedActions(actions=[]),
    ]
    with patch("reckon.retry.time.sleep"):
        result = _call_with_retry(client, "m", [])
    assert isinstance(result, _ExtractedActions)
    assert client.chat.completions.create.call_count == 2


def test_call_with_retry_retries_on_429():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _api_error(429),
        _ExtractedActions(actions=[]),
    ]
    with patch("reckon.retry.time.sleep"):
        _call_with_retry(client, "m", [])
    assert client.chat.completions.create.call_count == 2


def test_call_with_retry_does_not_retry_client_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = _api_error(400)
    with pytest.raises(openai.APIStatusError):
        _call_with_retry(client, "m", [])
    assert client.chat.completions.create.call_count == 1


def test_call_with_retry_reraises_after_exhausting_attempts():
    client = MagicMock()
    client.chat.completions.create.side_effect = _api_error(503)
    with patch("reckon.retry.time.sleep"), pytest.raises(openai.APIStatusError):
        _call_with_retry(client, "m", [])
    assert client.chat.completions.create.call_count == 4


def test_call_with_retry_invokes_on_retry_callback():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _api_error(503),
        _ExtractedActions(actions=[]),
    ]
    seen: list[tuple[int, int, float]] = []
    with patch("reckon.retry.time.sleep"):
        _call_with_retry(client, "m", [], on_retry=lambda a, c, d: seen.append((a, c, d)))
    assert seen == [(1, 503, 12.0)]


def test_call_with_retry_on_retry_not_called_on_success():
    client = _client([_action()])
    seen: list[tuple[int, int, float]] = []
    _call_with_retry(client, "m", [], on_retry=lambda a, c, d: seen.append((a, c, d)))
    assert seen == []


def test_extract_window_survives_transient_503():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _api_error(503),
        _ExtractedActions(actions=[_action()]),
    ]
    with (
        patch("reckon.extractor.extract_frames_at", return_value=_samples([0])),
        patch("reckon.retry.time.sleep"),
    ):
        trace = extract_window(client, _VIDEO, _SOURCE, [0])
    assert len(trace.actions) == 1


# ---------------------------------------------------------------------------
# Interaction binding and diagnostics on extracted observations
# ---------------------------------------------------------------------------

def _interaction_fields(**overrides: str | None) -> dict[str, str | None]:
    fields = {
        "interaction_operation": "activate",
        "interaction_kind": "button",
        "interaction_kind_text": "button",
        "interaction_name": "Sign in",
        "interaction_surface": "target",
    }
    fields.update(overrides)
    return fields


def _extract(actions: list[_ExtractedAction]) -> ObservedTrace:
    with patch("reckon.extractor.extract_frames_at", return_value=_samples([0, 500])):
        return extract_window(_client(actions), _VIDEO, _SOURCE, [0, 500])


def test_extract_window_attaches_a_bound_interaction_to_its_observation():
    trace = _extract([_action(frame_index=2, target="Sign in button", **_interaction_fields())])

    observed = trace.actions[0]
    assert (observed.interaction.element.name, observed.interaction.cited_frame_index) == (
        "Sign in", 2,
    )
    assert observed.interaction_diagnostic.status == InteractionParseStatus.bound


def test_extract_window_records_an_omitted_interaction():
    trace = _extract([_action(target="Sign in")])

    observed = trace.actions[0]
    assert (observed.interaction, observed.interaction_diagnostic.status) == (
        None, InteractionParseStatus.omitted,
    )


def test_extract_window_records_why_an_interaction_was_rejected():
    trace = _extract([_action(target="Sign in", **_interaction_fields())])

    observed = trace.actions[0]
    assert observed.interaction is None
    assert observed.interaction_diagnostic.reason == InteractionRejectionReason.invalid_citation


def test_extract_window_keeps_the_observation_when_its_interaction_is_rejected():
    trace = _extract([_action(target="Sign in", **_interaction_fields(interaction_kind="textbox"))])

    assert (trace.actions[0].action, trace.actions[0].target) == (ActionType.click, "Sign in")


def test_extract_window_gives_every_observation_a_diagnostic():
    trace = _extract([
        _action(target="Sign in button", **_interaction_fields()),
        _action(target="Next"),
        _action(target="Sign in", **_interaction_fields()),
    ])

    assert [a.interaction_diagnostic.status for a in trace.actions] == [
        InteractionParseStatus.bound,
        InteractionParseStatus.omitted,
        InteractionParseStatus.rejected,
    ]


def test_extract_window_never_cites_screen_description():
    trace = _extract([
        _action(
            target="Next",
            screen_description="Sign in button",
            **_interaction_fields(interaction_surface="screen_description"),
        )
    ])

    assert trace.actions[0].interaction_diagnostic.reason == (
        InteractionRejectionReason.unsupported_surface
    )
