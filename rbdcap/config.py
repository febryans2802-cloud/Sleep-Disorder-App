"""All tunable thresholds in one place.

Every number here is a starting guess, not a validated value. Tune them on your
own test nights and record what you changed and why.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TriggerConfig:
    """Cheap whole-night checks that decide which moments get analysed."""

    # Motion: frame differencing on a small grey image.
    motion_sample_fps: float = 5.0
    motion_width: int = 160  # pixels; the frame is shrunk to this width
    pixel_diff_threshold: int = 12  # grey levels; above this a pixel counts as changed
    motion_min_fraction: float = 0.01  # changed-pixel fraction that always counts as noise
    motion_k: float = 4.0  # trigger when changed fraction > k x local baseline

    # Sound: loudness relative to the room's noise floor.
    loudness_window_s: float = 0.5
    loudness_db_above_floor: float = 15.0
    loudness_min_dbfs: float = -50.0  # ignore anything quieter than this

    # Turning trigger moments into segments.
    baseline_block_s: float = 300.0  # baselines are re-estimated every 5 minutes
    min_trigger_s: float = 0.4  # drop single-frame blips (sensor noise in dim light)
    merge_gap_s: float = 5.0
    pad_s: float = 3.0  # context added on each side before analysis
    max_segment_s: float = 120.0


@dataclass
class PoseConfig:
    sample_fps: float = 10.0
    num_poses: int = 2  # track the bed partner too, so they can be ignored
    subject: str = "auto"  # "auto", "left" or "right" side of the image
    min_visibility: float = 0.5
    fast_speed: float = 2.0  # torso lengths per second counts as a fast limb movement
    smooth_frames: int = 3  # median filter against landmark jitter (1 = off)
    max_trunk_jump: float = 1.0  # torso lengths in one step; more means a tracking glitch


@dataclass
class ScoringConfig:
    """Hand-set rule weights. Score = sum of parts, clipped to 0..1."""

    limb_speed_full: float = 4.0  # torso lengths/s that earns full movement credit
    w_movement: float = 0.35
    w_limb_vs_trunk: float = 0.10
    limb_vs_trunk_min: float = 2.0  # limbs moving much faster than the trunk
    w_vocal: float = 0.30
    w_impact: float = 0.05
    w_late_night: float = 0.15
    late_night_from: float = 0.5  # fraction of the night; REM sleep dominates later
    w_snore_cough_penalty: float = 0.10
    motion_only_factor: float = 0.5  # movement credit when no person was detected


@dataclass
class ReportConfig:
    top_clips: int = 30
    clip_before_s: float = 10.0
    clip_after_s: float = 5.0
    clip_height: int = 480


@dataclass
class Config:
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> dict:
        return asdict(self)
