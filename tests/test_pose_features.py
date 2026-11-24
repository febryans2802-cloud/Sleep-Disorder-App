import numpy as np

from rbdcap.config import PoseConfig
from rbdcap.pose import L_ANKLE, R_ANKLE, L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST, pose_features, select_subject


def body(cx=300.0, cy=200.0):
    """A lying-down body: 33 landmarks, torso of 100 px, everything visible."""
    p = np.zeros((33, 3))
    p[:, 0], p[:, 1], p[:, 2] = cx, cy, 1.0
    p[L_SHOULDER, :2] = (cx - 10, cy - 50)
    p[R_SHOULDER, :2] = (cx + 10, cy - 50)
    p[L_HIP, :2] = (cx - 10, cy + 50)
    p[R_HIP, :2] = (cx + 10, cy + 50)
    p[L_WRIST, :2] = (cx - 30, cy)
    p[R_WRIST, :2] = (cx + 30, cy)
    return p


def test_still_sleeper_has_low_speed():
    times = np.arange(20) / 10
    track = np.stack([body() for _ in times])
    f = pose_features(times, track, PoseConfig())
    assert f is not None
    assert f.peak_limb_speed < 0.1 and f.fast_fraction == 0.0


def test_punching_scores_fast_and_limb_dominant():
    times = np.arange(20) / 10
    frames = []
    for i in range(len(times)):
        p = body()
        p[R_WRIST, 0] += 150 if (i // 3) % 2 else 0  # jabs 1.5 torso lengths, held 0.3 s
        frames.append(p)
    f = pose_features(times, np.stack(frames), PoseConfig())
    assert f.peak_limb_speed > 10
    assert f.fast_fraction > 0.2
    assert f.limb_vs_trunk > 2
    assert f.arm_reach > 1.5


def test_rolling_over_moves_trunk_with_limbs():
    times = np.arange(20) / 10
    track = np.stack([body(cx=300 + 30 * i) for i in range(len(times))])
    f = pose_features(times, track, PoseConfig())
    assert f.limb_vs_trunk < 2  # whole body moves together


def test_missing_frames_and_no_person():
    times = np.arange(10) / 10
    track = np.full((10, 33, 3), np.nan)
    assert pose_features(times, track, PoseConfig()) is None
    track[:5] = body()
    f = pose_features(times, track, PoseConfig())
    assert f is not None and f.detected_fraction == 0.5


def test_select_subject_by_side_and_continuity():
    left, right = body(cx=100), body(cx=500)
    assert select_subject([right, left], "left", 640, None) is left
    assert select_subject([left, right], "right", 640, None) is right
    assert select_subject([left, right], "auto", 640, np.array([505.0, 250.0])) is right
    assert select_subject([], "auto", 640, None) is None


def test_jitter_is_smoothed_and_identity_jumps_ignored():
    rng = np.random.default_rng(0)
    times = np.arange(30) / 10
    frames = []
    for i in range(len(times)):
        p = body()
        p[:, :2] += rng.normal(0, 4, (33, 2))  # landmark jitter of a few pixels
        if i == 15:
            p = body(cx=600)  # one frame locked onto someone else
        frames.append(p)
    f = pose_features(times, np.stack(frames), PoseConfig())
    assert f.peak_limb_speed < 2.0


def test_left_right_label_swap_is_not_a_kick():
    times = np.arange(20) / 10
    frames = []
    for i in range(len(times)):
        p = body()
        p[L_ANKLE, :2], p[R_ANKLE, :2] = (250, 300), (350, 300)
        if i % 4 == 2:
            p[[L_ANKLE, R_ANKLE]] = p[[R_ANKLE, L_ANKLE]]  # model swaps the labels
        frames.append(p)
    f = pose_features(times, np.stack(frames), PoseConfig(smooth_frames=1))
    assert f.peak_limb_speed < 0.1
