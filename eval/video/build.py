"""Render a case timeline into an encoded recording plus its ground truth.

Frames are encoded with the same H.264 path a real screen capture would take, so the
corpus carries genuine compression artefacts rather than being pixel-perfect. Together
with the always-moving cursor, that gives detection a real noise floor to discriminate
against instead of a synthetic one it would beat trivially.
"""

import json
from fractions import Fraction
from pathlib import Path

import av

from eval.video.cases import Case, Step
from eval.video.render import HEIGHT, WIDTH, draw_cursor, render

FPS = 20
_MS_PER_S = 1000

# Frames per spinner increment. The spinner exists to generate motion with no meaning;
# advancing it every frame keeps that motion continuous.
_SPINNER_STRIDE = 1


def _step_at(steps: list[Step], time_ms: int) -> Step:
    """Return the step in force at time_ms (the last one that has started)."""
    current = steps[0]
    for step in steps:
        if step.at_ms <= time_ms:
            current = step
        else:
            break
    return current


def _cursor_at(steps: list[Step], time_ms: int) -> tuple[int, int]:
    """Interpolate the pointer between the surrounding steps.

    A real pointer is always drifting; holding it still between events would leave the
    corpus with a zero noise floor and make detection look better than it is.
    """
    previous = steps[0]
    following = None
    for step in steps:
        if step.at_ms <= time_ms:
            previous = step
        else:
            following = step
            break
    if following is None:
        return previous.cursor
    span = following.at_ms - previous.at_ms
    if span <= 0:
        return following.cursor
    ratio = (time_ms - previous.at_ms) / span
    return (
        int(previous.cursor[0] + (following.cursor[0] - previous.cursor[0]) * ratio),
        int(previous.cursor[1] + (following.cursor[1] - previous.cursor[1]) * ratio),
    )


def _render_timeline(case: Case, frame_index: int) -> "Image":
    time_ms = int(frame_index * _MS_PER_S / FPS)
    step = _step_at(case.steps, time_ms)
    state = step.state
    if state.spinner_phase is not None:
        state.spinner_phase = frame_index // _SPINNER_STRIDE
    frame = render(state).copy()
    x, y = _cursor_at(case.steps, time_ms)
    draw_cursor(frame, x, y)
    return frame


def build_case(case: Case, out_dir: Path) -> tuple[Path, Path]:
    """Encode one case to <name>.mp4 and write <name>.truth.json beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{case.name}.mp4"
    truth_path = out_dir / f"{case.name}.truth.json"

    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, FPS)

    total_frames = int(case.duration_ms * FPS / _MS_PER_S)
    for frame_index in range(total_frames):
        image = _render_timeline(case, frame_index)
        video_frame = av.VideoFrame.from_image(image)
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    truth = {
        "name": case.name,
        "note": case.note,
        "duration_ms": case.duration_ms,
        "change_times_ms": case.change_times_ms,
        "actions": [
            {
                "at_ms": step.at_ms,
                "action": step.action,
                "target": step.target,
                "value": step.value,
                "assertion": step.assertion,
            }
            for step in case.actions
        ],
        "scenario": case.scenario,
        "expect_finding": list(case.expect_finding) if case.expect_finding else None,
    }
    truth_path.write_text(json.dumps(truth, indent=2))
    return video_path, truth_path
