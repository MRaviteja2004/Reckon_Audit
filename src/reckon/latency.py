"""Wall-clock latency metrics for pipeline diagnostics.

Collectors are optional: pass ``latency=`` explicitly, or install an active collector
via ``latency_scope`` so callers (e.g. Streamlit ``app.py``) need not pass a keyword
that older cached module objects may not accept. Logged fields are counts and
durations only — never tokens, prompts, images, or authorization material.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from reckon.retry import RetryCallback

logger = logging.getLogger("reckon.latency")


@dataclass
class WindowLatency:
    """Timing for one Stage 2b vision window."""

    index: int
    duration_ms: float
    frame_count: int
    image_payload_bytes: int


@dataclass
class LatencyMetrics:
    """Accumulated pipeline timing and volume counters."""

    stage1_ms: float | None = None
    stage2a_ms: float | None = None
    stage2b_ms: float | None = None
    stage3_ms: float | None = None
    stage4_semantic_ms: float | None = None
    stage4_visual_ms: float | None = None
    total_ms: float | None = None
    window_count: int = 0
    frame_count: int = 0
    retry_count: int = 0
    retry_sleep_s: float = 0.0
    stage4_semantic_candidate_count: int = 0
    stage4_visual_target_count: int = 0
    windows: list[WindowLatency] = field(default_factory=list)

    def record_retry(self, _attempt: int, _status_code: int, delay_s: float) -> None:
        """Count a transient-error backoff without altering retry policy."""
        self.retry_count += 1
        self.retry_sleep_s += delay_s

    def summary_fields(self) -> dict[str, float | int | None]:
        """Flat numeric fields safe to log or display."""
        return {
            "total_ms": self.total_ms,
            "stage1_ms": self.stage1_ms,
            "stage2a_ms": self.stage2a_ms,
            "stage2b_ms": self.stage2b_ms,
            "stage3_ms": self.stage3_ms,
            "stage4_semantic_ms": self.stage4_semantic_ms,
            "stage4_visual_ms": self.stage4_visual_ms,
            "window_count": self.window_count,
            "frame_count": self.frame_count,
            "retry_count": self.retry_count,
            "retry_sleep_s": round(self.retry_sleep_s, 3),
            "stage4_semantic_candidate_count": self.stage4_semantic_candidate_count,
            "stage4_visual_target_count": self.stage4_visual_target_count,
        }


_active_latency: ContextVar[LatencyMetrics | None] = ContextVar(
    "reckon_active_latency", default=None
)


def monotonic_ms() -> float:
    """Return a monotonic clock reading in milliseconds."""
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    """Milliseconds elapsed since a monotonic_ms() reading."""
    return monotonic_ms() - start_ms


def active_latency() -> LatencyMetrics | None:
    """Return the LatencyMetrics installed by the nearest latency_scope, if any."""
    return _active_latency.get()


def resolve_latency(explicit: LatencyMetrics | None) -> LatencyMetrics | None:
    """Prefer an explicit collector; otherwise use the active scope collector."""
    return explicit if explicit is not None else active_latency()


@contextmanager
def latency_scope(metrics: LatencyMetrics | None) -> Iterator[LatencyMetrics | None]:
    """Install metrics for the duration of the block without requiring callee kwargs."""
    if metrics is None:
        yield None
        return
    token = _active_latency.set(metrics)
    try:
        yield metrics
    finally:
        _active_latency.reset(token)


def bind_retry_callback(
    metrics: LatencyMetrics | None,
    on_retry: RetryCallback | None,
) -> RetryCallback | None:
    """Return an on_retry callback that records metrics then forwards.

    When metrics is None the original callback is returned unchanged so identity
    and behaviour stay identical for callers that do not opt into instrumentation.
    """
    if metrics is None:
        return on_retry

    def _wrapped(attempt: int, status_code: int, delay_s: float) -> None:
        metrics.record_retry(attempt, status_code, delay_s)
        if on_retry is not None:
            on_retry(attempt, status_code, delay_s)

    return _wrapped


def log_metrics(metrics: LatencyMetrics, *, where: str = "pipeline") -> None:
    """Emit a single INFO line of numeric latency fields."""
    fields = metrics.summary_fields()
    parts = [f"{key}={value}" for key, value in fields.items()]
    logger.info("latency[%s] %s", where, " ".join(parts))
    for window in metrics.windows:
        logger.info(
            "latency[%s].window index=%d duration_ms=%.1f frame_count=%d "
            "image_payload_bytes=%d",
            where,
            window.index,
            window.duration_ms,
            window.frame_count,
            window.image_payload_bytes,
        )
