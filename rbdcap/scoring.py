"""Transparent rule-based score for each segment.

The score ranks clips for human review. It is not a probability of RBD and must
never be presented as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio import SoundFeatures
from .config import ScoringConfig
from .pose import PoseFeatures
from .segments import Segment


@dataclass
class Episode:
    segment: Segment
    pose: PoseFeatures | None
    sound: SoundFeatures | None
    night_fraction: float | None  # 0 = lights out, 1 = wake-up
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_episode(
    segment: Segment,
    pose: PoseFeatures | None,
    sound: SoundFeatures | None,
    night_fraction: float | None,
    cfg: ScoringConfig,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    # Movement
    if pose is not None:
        part = cfg.w_movement * _clip01(pose.peak_limb_speed / cfg.limb_speed_full)
        score += part
        if part > 0.05:
            reasons.append(f"fast limb movement ({pose.peak_limb_speed:.1f} torso/s)")
        if pose.limb_vs_trunk >= cfg.limb_vs_trunk_min:
            score += cfg.w_limb_vs_trunk
            reasons.append(f"limbs moved {pose.limb_vs_trunk:.1f}x faster than trunk")
    elif "motion" in segment.sources:
        # Nobody detected (too dark, or covered by a blanket): give partial credit
        # for raw motion so the clip still gets reviewed.
        part = cfg.w_movement * cfg.motion_only_factor * _clip01(segment.motion_peak / 3.0)
        score += part
        reasons.append("movement seen, but no body detected")

    # Sound
    if sound is not None:
        g = sound.groups
        vocal = max(g.get("shout", 0.0), 0.7 * g.get("speech", 0.0),
                    0.6 * g.get("groan", 0.0), 0.5 * g.get("laugh", 0.0))
        if vocal > 0.1:
            score += cfg.w_vocal * vocal
            best = max(("shout", "speech", "groan", "laugh"), key=lambda k: g.get(k, 0.0))
            label = sound.group_labels.get(best, best)
            reasons.append(f"vocalisation: {label} (p={g.get(best, 0.0):.2f})")
        if g.get("impact", 0.0) > 0.2:
            score += cfg.w_impact * g["impact"]
            reasons.append(f"impact sound (p={g['impact']:.2f})")
        snore_cough = max(g.get("snore", 0.0), g.get("cough", 0.0))
        if snore_cough > 0.3 and snore_cough > vocal:
            score -= cfg.w_snore_cough_penalty * snore_cough
            reasons.append("mostly snoring/coughing (possible apnea arousal)")

    # Timing: REM sleep, and so RBD, is concentrated in the second half of the night.
    if night_fraction is not None and night_fraction >= cfg.late_night_from:
        score += cfg.w_late_night
        reasons.append("second half of the night")

    return _clip01(score), reasons
