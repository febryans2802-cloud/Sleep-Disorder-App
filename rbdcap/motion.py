"""Whole-night motion trigger: how much of the picture changed between frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import TriggerConfig
from .segments import Segment, blockwise_baseline, mask_to_intervals


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def video_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0 or fps > 240:
        return 30.0  # some containers report nonsense; 30 is the usual phone default
    return float(fps)


def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    return cap


@dataclass
class MotionTrace:
    times: np.ndarray  # start of each sample window, seconds
    fraction: np.ndarray  # fraction of pixels that changed
    window_s: float
    info: VideoInfo


def compute_motion(
    video_path: str | Path,
    cfg: TriggerConfig,
    progress: Callable[[float], None] | None = None,
) -> MotionTrace:
    """Decode the whole night once at a low sample rate.

    Frames between samples are only grabbed, not decoded into images, which keeps
    an 8-hour recording to minutes rather than hours.
    """
    cap = open_video(video_path)
    fps = video_fps(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(fps / cfg.motion_sample_fps))
    times: list[float] = []
    fraction: list[float] = []
    prev = None
    prev_t = 0.0
    idx = 0
    width = height = 0
    try:
        while True:
            if idx % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_h = max(1, round(height * cfg.motion_width / width))
            small = cv2.resize(grey, (cfg.motion_width, small_h), interpolation=cv2.INTER_AREA)
            # Blurring suppresses the speckle noise phone sensors produce in dim light.
            small = cv2.GaussianBlur(small, (5, 5), 0)
            t = idx / fps
            if prev is not None:
                diff = cv2.absdiff(small, prev)
                times.append(prev_t)
                fraction.append(float((diff > cfg.pixel_diff_threshold).mean()))
            prev, prev_t = small, t
            idx += 1
            if progress and total and idx % (step * 500) == 0:
                progress(idx / total)
    finally:
        cap.release()
    info = VideoInfo(fps=fps, frame_count=idx, width=width, height=height)
    return MotionTrace(np.array(times), np.array(fraction), step / fps, info)


def motion_segments(trace: MotionTrace, cfg: TriggerConfig) -> list[Segment]:
    if trace.fraction.size == 0:
        return []
    baseline = blockwise_baseline(trace.fraction, trace.times, cfg.baseline_block_s, q=50)
    threshold = np.maximum(cfg.motion_min_fraction, cfg.motion_k * baseline)
    ratio = trace.fraction / threshold
    segments = []
    for start, end in mask_to_intervals(trace.times, ratio > 1.0, trace.window_s):
        sel = (trace.times >= start) & (trace.times < end)
        segments.append(Segment(start, end, {"motion"}, motion_peak=float(ratio[sel].max())))
    return segments
