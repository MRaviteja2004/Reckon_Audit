"""Frame manifest helpers for Stage 2C timestamp materialization."""

from reckon.schema import FrameManifest, FrameManifestEntry


def build_frame_manifest(timestamps_ms: list[int]) -> FrameManifest:
    """Build F1..Fn labels for absolute recording timestamps (decoder order)."""
    if not timestamps_ms:
        raise ValueError("timestamps_ms must not be empty")
    if any(ts < 0 for ts in timestamps_ms):
        raise ValueError("timestamps_ms must be non-negative")
    return FrameManifest(
        frames=[
            FrameManifestEntry(frame_id=f"F{index}", timestamp_ms=timestamp_ms)
            for index, timestamp_ms in enumerate(timestamps_ms, start=1)
        ]
    )


def normalize_frame_id(frame_id: str) -> str:
    return frame_id.strip().upper()


def materialize_timestamp_range(
    manifest: FrameManifest,
    frame_ids: list[str],
) -> tuple[list[str], int, int] | None:
    """Resolve frame IDs to (normalized_ids, start_ms, end_ms).

    Returns None when any frame ID is missing from the manifest.
    """
    if not frame_ids:
        return None
    resolved: list[tuple[str, int]] = []
    for raw in frame_ids:
        fid = normalize_frame_id(raw)
        ts = manifest.timestamp_for(fid)
        if ts is None:
            return None
        resolved.append((fid, ts))
    times = [ts for _, ts in resolved]
    return [fid for fid, _ in resolved], min(times), max(times)
