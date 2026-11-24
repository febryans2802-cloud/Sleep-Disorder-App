from rbdcap.audio import SOUND_GROUPS, SoundFeatures, summarise_scores
from rbdcap.config import ScoringConfig
from rbdcap.pose import PoseFeatures
from rbdcap.scoring import score_episode
from rbdcap.segments import Segment


def sound(**groups):
    g = {k: 0.0 for k in SOUND_GROUPS}
    g.update(groups)
    return SoundFeatures(groups=g, group_labels={k: k.title() for k in g})


def test_enactment_scores_higher_than_rolling_over():
    cfg = ScoringConfig()
    seg = Segment(0, 10, {"motion", "sound"})
    enact, reasons = score_episode(seg, PoseFeatures(1.0, 5.0, 0.4, 4.0, 1.6), sound(shout=0.8), 0.8, cfg)
    roll, _ = score_episode(seg, PoseFeatures(1.0, 1.0, 0.0, 1.1, 0.6), sound(), 0.2, cfg)
    assert enact > 0.8 > 0.2 > roll
    assert any("Shout" in r for r in reasons)
    assert "second half of the night" in reasons


def test_snoring_is_penalised_and_score_is_clipped():
    cfg = ScoringConfig()
    seg = Segment(0, 10, {"sound"})
    s_snore, reasons = score_episode(seg, None, sound(snore=0.9), 0.1, cfg)
    assert s_snore == 0.0
    assert any("snoring" in r for r in reasons)


def test_motion_only_gets_partial_credit():
    cfg = ScoringConfig()
    s, reasons = score_episode(Segment(0, 5, {"motion"}, motion_peak=6.0), None, None, None, cfg)
    assert 0 < s <= cfg.w_movement * cfg.motion_only_factor + 1e-9
    assert "no body detected" in reasons[0]


def test_summarise_scores_takes_max_per_group():
    f = summarise_scores([{"Speech": 0.3, "Silence": 0.9}, {"Yell": 0.6, "Snoring": 0.1}])
    assert f.groups["speech"] == 0.3 and f.groups["shout"] == 0.6
    assert f.group_labels["shout"] == "Yell"
    assert f.top_label == "Yell"  # "Silence" is not reported as the top sound
