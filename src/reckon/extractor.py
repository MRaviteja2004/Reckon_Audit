"""Stage 2 extraction — observed actions read from recording frames by a vision model.

The model is shown a labelled sequence of full-resolution stills pulled at the moments
keyframes.py found interesting, and asked what the user did. It never sees the video
and never reports a time.

That last point is the design's centre of gravity. Asking a model "when did this
happen?" invites drift, and a drifting timestamp poisons the evidence pointer the whole
audit rests on. Here the model instead names the *frame* an action is visible in, and
the timestamp is looked up in code from the frame's decoder time. A model that
misidentifies a frame produces a wrong-but-real timestamp pointing at a frame a human
can open and check; it cannot invent a moment that does not exist.

The expected scenario is deliberately withheld — supplying the expectation before
extraction causes object-presence bias and manufactures false passes, the exact failure
this audit exists to catch.
"""

import base64
import logging
import os
from pathlib import Path

import instructor
from pydantic import BaseModel, Field

from reckon.frames import extract_frames_at, frame_to_jpeg_bytes
from reckon.latency import (
    LatencyMetrics,
    WindowLatency,
    elapsed_ms,
    monotonic_ms,
    resolve_latency,
)
from reckon.parser import make_openai_client
from reckon.retry import RetryCallback, call_with_retry
from reckon.schema import ActionType, AssertionEvidence, ObservedAction, ObservedTrace
from reckon.stage2.interaction_binding import InteractionOutcome, bind_reported_interaction
from reckon.stage2.interaction_contract import ReportedInteraction, interaction_prompt_section

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "claude-opus-5-5")

# Frames are sent at "high" detail: the audit turns on reading error banners, toast
# text, and field values, which low detail renders illegible.
EXTRACTION_DETAIL = "high"
_DETAIL_LEVELS = frozenset({"low", "high", "auto"})

# Longest side of a transmitted frame. Above this the vision API downscales anyway,
# so the extra pixels cost upload time and buy nothing.
MAX_FRAME_DIMENSION = 1536

# temperature is omitted (never sent): determinism is the goal, but some gateway models
# accept only their default temperature and reject any explicit value.

EXTRACTION_SYSTEM = """\
You are reading frames from a UI test recording. The frames are stills captured at the \
moments the screen changed, in chronological order, each labelled "Frame N".

List every user action you can see, in the order it occurs.

Describe only what is clearly visible. Do not invent actions, and do not judge the \
recording against any expectation — the comparison is handled separately.

Rules:
- Set frame_index to the label number of the frame where the action is most clearly \
visible. Never guess a time; the frame number is the only position you report.
- Report an action once. If it stays visible across several frames, use the frame \
where it is clearest.
- Set assertion_evidence whenever a message, toast, dialog, or on-screen text appears \
that could confirm or refute a test outcome: error banners, success notices, field \
validation text. Set visible_text to the exact on-screen text if legible.
- Record such a message as its own entry with action=assert_visible (or assert_text), \
in addition to whatever click preceded it. A click and the outcome it produced are two \
separate observations, and folding the outcome into the click loses it.
- Do not open with a navigate action invented to describe the first frame. Report a \
navigate only when you can see the page actually change to a new one.
- Set confidence to reflect how certain you are the action occurred as described.
""" + interaction_prompt_section()


class _ExtractedAction(ReportedInteraction):
    """One action as reported by the model, positioned by frame rather than by time."""

    frame_index: int = Field(
        ge=1,
        description="Label number of the frame where this action is most clearly visible",
    )
    action: ActionType
    target: str | None = Field(None, description="UI element, page, or URL acted upon")
    input_value: str | None = Field(None, description="Text entered or value observed")
    assertion_evidence: AssertionEvidence | None = Field(
        None,
        description="Assertion-relevant evidence visible at this step",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence this action occurred (0–1)")
    screen_description: str | None = Field(
        None,
        description="Brief description of the screen state in this frame",
    )


class _ExtractedActions(BaseModel):
    """Response schema for the extraction call — only the model-provided portion of a trace."""

    actions: list[_ExtractedAction]


def make_client() -> instructor.Instructor:
    """Create the instructor-patched OpenAI client used for frame extraction."""
    return make_openai_client()


def _frame_content(frame_number: int, jpeg_bytes: bytes, detail: str) -> list[dict]:
    """Build the label-then-image content pair for one frame.

    The text label precedes its image so the model can bind "Frame N" to the picture
    that follows; without it, frame_index would be a guess at ordinal position.
    """
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return [
        {"type": "text", "text": f"Frame {frame_number}:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": detail},
        },
    ]


def _call_with_retry(
    openai_client: instructor.Instructor,
    model: str,
    content: list[dict],
    on_retry: RetryCallback | None = None,
) -> _ExtractedActions:
    """Call the vision model with shared rate-limit and transient-error retries."""
    return call_with_retry(
        lambda: openai_client.chat.completions.create(
            model=model,
            response_model=_ExtractedActions,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": content},
            ],
        ),
        on_retry=on_retry,
    )


def _resolve_frame_index(frame_index: int, frame_count: int) -> int:
    """Clamp a model-reported frame label into the range of frames actually sent.

    The frame list is built in code, so it is the authoritative bound. A model that
    names Frame 9 when eight were shown is corrected here rather than being allowed to
    produce an out-of-range evidence pointer.
    """
    return max(1, min(frame_index, frame_count))


def _bound_interaction(reported: _ExtractedAction, frame_index: int) -> InteractionOutcome:
    evidence = reported.assertion_evidence
    return bind_reported_interaction(
        reported,
        action=reported.action,
        surfaces={
            "target": reported.target,
            "input_value": reported.input_value,
            "visible_text": None if evidence is None else evidence.visible_text,
        },
        frame_index=frame_index,
    )


def extract_window(
    openai_client: instructor.Instructor,
    video_path: Path,
    source_file: str,
    frame_timestamps_ms: list[int],
    *,
    clip_offset_ms: int = 0,
    clip_end_ms: int | None = None,
    model: str = EXTRACTION_MODEL,
    detail: str = EXTRACTION_DETAIL,
    on_retry: RetryCallback | None = None,
    latency: LatencyMetrics | None = None,
    window_index: int = 0,
) -> ObservedTrace:
    """Extract an observed trace from one window of a recording, given its frame times.

    frame_timestamps_ms are absolute offsets into the recording and must fall inside
    the clip bounds; they are the only positions the resulting actions can be anchored
    to. Returned timestamps are clip-relative, matching what merge_observed_chunks
    expects to convert back to absolute time.

    When latency is provided (argument or active latency_scope), records this window's
    wall time, frame count, and JPEG payload size (bytes before base64). Never logs
    image or prompt contents.
    """
    latency = resolve_latency(latency)
    if clip_end_ms is not None and clip_end_ms <= clip_offset_ms:
        raise ValueError(
            f"clip_end_ms ({clip_end_ms}) must be greater than clip_offset_ms ({clip_offset_ms})"
        )
    if not frame_timestamps_ms:
        raise ValueError("frame_timestamps_ms must not be empty")
    if detail not in _DETAIL_LEVELS:
        raise ValueError(f"detail must be one of {sorted(_DETAIL_LEVELS)}, got {detail!r}")
    if any(ts < clip_offset_ms for ts in frame_timestamps_ms):
        raise ValueError(
            f"frame timestamps must be at or after clip_offset_ms ({clip_offset_ms})"
        )
    if clip_end_ms is not None and any(ts > clip_end_ms for ts in frame_timestamps_ms):
        raise ValueError(f"frame timestamps must be at or before clip_end_ms ({clip_end_ms})")

    window_started = monotonic_ms()
    samples = extract_frames_at(video_path, frame_timestamps_ms)

    content: list[dict] = [
        {"type": "text", "text": "Frames follow, in chronological order."}
    ]
    image_payload_bytes = 0
    for frame_number, (_, image) in enumerate(samples, start=1):
        jpeg_bytes = frame_to_jpeg_bytes(image, max_dimension=MAX_FRAME_DIMENSION)
        image_payload_bytes += len(jpeg_bytes)
        content.extend(_frame_content(frame_number, jpeg_bytes, detail))

    extracted = _call_with_retry(openai_client, model, content, on_retry)

    sample_timestamps = [ts for ts, _ in samples]
    actions = []
    for index, reported in enumerate(extracted.actions, start=1):
        frame_index = _resolve_frame_index(reported.frame_index, len(sample_timestamps))
        bound = _bound_interaction(reported, frame_index)
        actions.append(ObservedAction(
            index=index,
            timestamp_ms=sample_timestamps[frame_index - 1] - clip_offset_ms,
            action=reported.action,
            target=reported.target,
            input_value=reported.input_value,
            assertion_evidence=reported.assertion_evidence,
            confidence=reported.confidence,
            screen_description=reported.screen_description,
            interaction=bound.interaction,
            interaction_diagnostic=bound.diagnostic,
        ))

    if latency is not None:
        frame_count = len(samples)
        window = WindowLatency(
            index=window_index,
            duration_ms=elapsed_ms(window_started),
            frame_count=frame_count,
            image_payload_bytes=image_payload_bytes,
        )
        latency.windows.append(window)
        latency.frame_count += frame_count
        logger.info(
            "stage2b.window index=%d duration_ms=%.1f frame_count=%d "
            "image_payload_bytes=%d",
            window.index,
            window.duration_ms,
            window.frame_count,
            window.image_payload_bytes,
        )

    return ObservedTrace(
        source_file=source_file,
        clip_offset_ms=clip_offset_ms,
        clip_end_ms=clip_end_ms,
        actions=actions,
    )
