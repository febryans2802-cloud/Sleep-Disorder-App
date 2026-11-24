"""Body pose inside each segment, and movement features computed from it."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import PoseConfig

# MediaPipe pose landmark indices (33-point model).
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28
LIMBS = (L_WRIST, R_WRIST, L_ANKLE, R_ANKLE)
N_LANDMARKS = 33


@dataclass
class PoseFeatures:
    detected_fraction: float  # share of sampled frames with the sleeper found
    peak_limb_speed: float  # 95th percentile of the fastest limb, torso lengths/s
    fast_fraction: float  # share of steps with a limb above PoseConfig.fast_speed
    limb_vs_trunk: float  # limb speed relative to trunk speed (high = flailing, low = rolling over)
    arm_reach: float  # furthest wrist-to-shoulder distance, in torso lengths


def _hip_center(p: np.ndarray) -> np.ndarray:
    return (p[L_HIP, :2] + p[R_HIP, :2]) / 2


def _shoulder_center(p: np.ndarray) -> np.ndarray:
    return (p[L_SHOULDER, :2] + p[R_SHOULDER, :2]) / 2


def smooth_track(xy: np.ndarray, frames: int) -> np.ndarray:
    """Centred, NaN-aware moving median over time for a (T, L, 2) array."""
    if frames <= 1 or len(xy) < frames:
        return xy
    half = frames // 2
    padded = np.concatenate([np.repeat(xy[:1], half, axis=0), xy, np.repeat(xy[-1:], half, axis=0)])
    stack = np.stack([padded[i: i + len(xy)] for i in range(frames)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows stay NaN
        return np.nanmedian(stack, axis=0)


def pair_speeds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-step displacement of a left/right landmark pair, (T-1, 2).

    The pose model sometimes swaps left and right (common for a person lying
    down or seen from above), which would look like a fast kick. Each step uses
    whichever labelling moves the pair less.
    """
    same = np.stack([np.linalg.norm(a[1:] - a[:-1], axis=1), np.linalg.norm(b[1:] - b[:-1], axis=1)], axis=1)
    swap = np.stack([np.linalg.norm(b[1:] - a[:-1], axis=1), np.linalg.norm(a[1:] - b[:-1], axis=1)], axis=1)
    use_swap = swap.sum(axis=1) < same.sum(axis=1)  # NaN sums compare False -> keep labels
    return np.where(use_swap[:, None], swap, same)


def select_subject(
    poses: list[np.ndarray],
    subject: str,
    image_width: int,
    prev_center: np.ndarray | None,
) -> np.ndarray | None:
    """Pick the sleeper out of up to two detected people.

    subject="left"/"right" picks the person on that side of the image (the most
    reliable choice when a bed partner is in frame). "auto" follows whoever is
    closest to the previous frame's sleeper, starting with the largest person.
    """
    if not poses:
        return None
    if subject in ("left", "right"):
        xs = [_hip_center(p)[0] for p in poses]
        i = int(np.argmin(xs)) if subject == "left" else int(np.argmax(xs))
        return poses[i]
    if prev_center is not None:
        d = [np.linalg.norm(_hip_center(p) - prev_center) for p in poses]
        if min(d) < 0.25 * image_width:
            return poses[int(np.argmin(d))]

    def area(p: np.ndarray) -> float:
        return float(np.ptp(p[:, 0]) * np.ptp(p[:, 1]))

    return max(poses, key=area)


def pose_features(times: np.ndarray, track: np.ndarray, cfg: PoseConfig) -> PoseFeatures | None:
    """Movement features from a (T, 33, 3) array of [x_px, y_px, visibility].

    Frames where the sleeper was not found are NaN. Speeds are divided by the
    sleeper's torso length so they don't depend on how far the phone stands
    from the bed.
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 3:
        return None
    found = ~np.isnan(track[:, 0, 0])
    detected_fraction = float(found.mean())
    if found.sum() < 3:
        return None

    torso_each = np.linalg.norm(
        np.array([_shoulder_center(p) - _hip_center(p) for p in track]), axis=1
    )
    torso = float(np.nanmedian(torso_each))
    if not np.isfinite(torso) or torso < 5:  # pixels; too small to trust
        return None

    xy = track[:, :, :2].copy()
    xy[track[:, :, 2] < cfg.min_visibility] = np.nan
    xy = smooth_track(xy, cfg.smooth_frames)
    dt = np.diff(times)[:, None]

    limb_step = np.concatenate(
        [pair_speeds(xy[:, L_WRIST], xy[:, R_WRIST]), pair_speeds(xy[:, L_ANKLE], xy[:, R_ANKLE])], axis=1
    )
    limb_speed = limb_step / dt / torso  # (T-1, 4)
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    hip_step = np.linalg.norm(np.diff(hip, axis=0), axis=1) / torso
    trunk_speed = hip_step / dt[:, 0]
    # A trunk that jumps more than a torso length between samples is the tracker
    # switching people or losing the body, not real movement: drop those steps.
    glitch = hip_step > cfg.max_trunk_jump
    limb_speed[glitch] = np.nan
    trunk_speed[glitch] = np.nan

    with np.errstate(all="ignore"):
        fastest = np.nanmax(np.where(np.isnan(limb_speed), -np.inf, limb_speed), axis=1)
    fastest[~np.isfinite(fastest)] = np.nan
    valid = ~np.isnan(fastest)
    if valid.sum() < 2:
        return None

    peak = float(np.nanpercentile(fastest, 95))
    fast_fraction = float((fastest[valid] > cfg.fast_speed).mean())
    trunk_peak = float(np.nanpercentile(trunk_speed, 95)) if np.any(~np.isnan(trunk_speed)) else 0.0
    limb_vs_trunk = peak / (trunk_peak + 0.25)  # +0.25 keeps a still trunk from exploding the ratio

    reach = []
    for wrist, shoulder in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        reach.append(np.linalg.norm(xy[:, wrist] - xy[:, shoulder], axis=1) / torso)
    reach_all = np.concatenate(reach)
    arm_reach = float(np.nanmax(reach_all)) if np.any(~np.isnan(reach_all)) else 0.0

    return PoseFeatures(detected_fraction, peak, fast_fraction, limb_vs_trunk, arm_reach)


class PoseTracker:
    """MediaPipe Pose Landmarker in video mode, run only inside segments."""

    def __init__(self, model_path: str | Path, cfg: PoseConfig):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.cfg = cfg
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=cfg.num_poses,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1
        self._prev_center: np.ndarray | None = None

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def track(self, cap: cv2.VideoCapture, fps: float, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, track) for one segment. Segments must be given in time order."""
        step = max(1, round(fps / self.cfg.sample_fps))
        idx = int(start_s * fps)
        last = int(end_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        times, rows = [], []
        while idx <= last:
            if (idx - int(start_s * fps)) % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            idx += 1
            # Video mode needs strictly increasing timestamps across the whole night.
            ts = max(self._last_ts + 1, int(t * 1000))
            self._last_ts = ts
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, ts)
            poses = [
                np.array([[lm.x * w, lm.y * h, lm.visibility if lm.visibility is not None else 1.0] for lm in person])
                for person in result.pose_landmarks
            ]
            chosen = select_subject(poses, self.cfg.subject, w, self._prev_center)
            if chosen is None:
                rows.append(np.full((N_LANDMARKS, 3), np.nan))
            else:
                self._prev_center = _hip_center(chosen)
                rows.append(chosen)
            times.append(t)
        if not rows:
            return np.array([]), np.empty((0, N_LANDMARKS, 3))
        return np.array(times), np.stack(rows)
