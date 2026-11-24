import numpy as np

from rbdcap.segments import Segment, blockwise_baseline, mask_to_intervals, merge_segments


def test_mask_to_intervals():
    times = np.arange(10) * 0.5
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1], bool)
    assert mask_to_intervals(times, mask, 0.5) == [(0.5, 1.5), (2.5, 3.0), (4.0, 5.0)]


def test_baseline_follows_level_changes():
    times = np.arange(0, 1200, 1.0)
    values = np.where(times < 600, 1.0, 5.0)
    base = blockwise_baseline(values, times, block_s=300, q=50)
    assert base[0] == 1.0 and base[-1] == 5.0


def test_merge_drops_blips_pads_merges_and_splits():
    segs = [
        Segment(10.0, 10.2, {"motion"}),          # blip: dropped
        Segment(20.0, 21.0, {"motion"}, motion_peak=2.0),
        Segment(25.0, 26.0, {"sound"}, loudness_peak_db=20.0),  # within gap after padding: merged
        Segment(100.0, 400.0, {"motion"}),        # long: split
    ]
    out = merge_segments(segs, gap_s=5, pad_s=3, duration_s=1000, min_trigger_s=0.4, max_segment_s=120)
    first = out[0]
    assert (first.start_s, first.end_s) == (17.0, 29.0)
    assert first.sources == {"motion", "sound"}
    assert first.motion_peak == 2.0 and first.loudness_peak_db == 20.0
    long_parts = out[1:]
    assert len(long_parts) == 3
    assert all(p.duration_s <= 120 for p in long_parts)
    assert long_parts[0].start_s == 97.0 and long_parts[-1].end_s == 403.0
