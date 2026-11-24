"""Turn per-window trigger signals into time segments worth analysing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:
    start_s: float
    end_s: float
    sources: set[str] = field(default_factory=set)  # "motion", "sound"
    motion_peak: float = 0.0  # peak changed-fraction / trigger threshold
    loudness_peak_db: float = 0.0  # peak dB above the room's noise floor

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def mid_s(self) -> float:
        return (self.start_s + self.end_s) / 2


def blockwise_baseline(values, times, block_s: float = 300.0, q: float = 50.0) -> np.ndarray:
    """A slowly varying baseline: the q-th percentile of each block, interpolated.

    Recomputing it every few minutes copes with a fan switching on or the room
    getting lighter towards morning.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.size == 0:
        return values.copy()
    block = np.floor((times - times[0]) / block_s).astype(int)
    centers, levels = [], []
    for b in np.unique(block):
        sel = block == b
        centers.append(times[sel].mean())
        levels.append(np.nanpercentile(values[sel], q))
    if len(centers) == 1:
        return np.full_like(values, levels[0])
    return np.interp(times, centers, levels)


def mask_to_intervals(times, mask, window_s: float) -> list[tuple[float, float]]:
    """Runs of True samples -> (start, end) intervals; each sample covers window_s."""
    intervals: list[tuple[float, float]] = []
    start = end = None
    for t, m in zip(times, mask):
        if m:
            if start is None:
                start = float(t)
            end = float(t) + window_s
        elif start is not None:
            intervals.append((start, end))
            start = None
    if start is not None:
        intervals.append((start, end))
    return intervals


def merge_segments(
    segments: list[Segment],
    *,
    gap_s: float,
    pad_s: float,
    duration_s: float,
    min_trigger_s: float,
    max_segment_s: float,
) -> list[Segment]:
    """Drop blips, pad, merge nearby segments, then split very long ones."""
    kept = [s for s in segments if s.duration_s >= min_trigger_s - 1e-9]
    padded = [
        Segment(max(0.0, s.start_s - pad_s), min(duration_s, s.end_s + pad_s),
                set(s.sources), s.motion_peak, s.loudness_peak_db)
        for s in kept
    ]
    padded.sort(key=lambda s: s.start_s)

    merged: list[Segment] = []
    for s in padded:
        if merged and s.start_s - merged[-1].end_s <= gap_s:
            cur = merged[-1]
            cur.end_s = max(cur.end_s, s.end_s)
            cur.sources |= s.sources
            cur.motion_peak = max(cur.motion_peak, s.motion_peak)
            cur.loudness_peak_db = max(cur.loudness_peak_db, s.loudness_peak_db)
        else:
            merged.append(s)

    out: list[Segment] = []
    for s in merged:
        if s.duration_s <= max_segment_s:
            out.append(s)
            continue
        # Long restless periods are split so each clip stays reviewable. The
        # trigger peaks are kept on every piece because we no longer know where
        # inside the period they happened.
        t = s.start_s
        while t < s.end_s - 1e-9:
            end = min(s.end_s, t + max_segment_s)
            out.append(Segment(t, end, set(s.sources), s.motion_peak, s.loudness_peak_db))
            t = end
    return out
