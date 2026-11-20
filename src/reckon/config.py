"""Runtime configuration loaded from the environment."""

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


STAGE2_EVENT_SELECTION_POLICY = os.getenv("STAGE2_EVENT_SELECTION_POLICY", "legacy")

# Opt-in denser frame sampling on static choice-control screens.
# "legacy" keeps select_window_frames unchanged; "choice_control" enables adaptive density.
STAGE2_ADAPTIVE_SAMPLING_POLICY = os.getenv(
    "STAGE2_ADAPTIVE_SAMPLING_POLICY", "legacy"
)
STAGE2_ADAPTIVE_SAMPLE_INTERVAL_MS = _env_int(
    "STAGE2_ADAPTIVE_SAMPLE_INTERVAL_MS", 1000
)
STAGE2_ADAPTIVE_MAX_EXTRA_FRAMES = _env_int(
    "STAGE2_ADAPTIVE_MAX_EXTRA_FRAMES", 8
)
STAGE2_ADAPTIVE_MAX_FRAMES_PER_WINDOW = _env_int(
    "STAGE2_ADAPTIVE_MAX_FRAMES_PER_WINDOW", 16
)
STAGE2_ADAPTIVE_INTERACTION_WINDOW_MS = _env_int(
    "STAGE2_ADAPTIVE_INTERACTION_WINDOW_MS", 45_000
)

# Opt-in frame sampling of long quiet stretches that change detection leaves unsampled.
# "legacy" keeps change-detected windows only; "gap_coverage" adds one midpoint frame
# per gap longer than STAGE2_GAP_MIN_MS, up to STAGE2_GAP_MAX_EXTRA_FRAMES per recording.
STAGE2_GAP_SAMPLING_POLICY = os.getenv("STAGE2_GAP_SAMPLING_POLICY", "legacy")
STAGE2_GAP_MIN_MS = _env_int("STAGE2_GAP_MIN_MS", 8000)
STAGE2_GAP_MAX_EXTRA_FRAMES = _env_int("STAGE2_GAP_MAX_EXTRA_FRAMES", 8)
