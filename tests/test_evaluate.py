from rbdcap.evaluate import Interval, evaluate, load_ground_truth
from rbdcap.timeline import parse_clock


def test_recall_and_false_alarms():
    truth = [Interval(100, 110), Interval(500, 505), Interval(800, 810, "normal")]
    dets = [
        Interval(95, 120, score=0.9),   # finds episode 1
        Interval(520, 530, score=0.3),  # finds episode 2 within 15 s tolerance, low score
        Interval(805, 812, score=0.5),  # false alarm on deliberate normal movement
        Interval(2000, 2005, score=0.8),  # false alarm elsewhere
    ]
    rows = {r["threshold"]: r for r in evaluate(dets, truth, hours=1.0, thresholds=(0.2, 0.6))}
    assert rows[0.2]["episodes_found"] == 2 and rows[0.2]["recall"] == 1.0
    assert rows[0.2]["false_alarms"] == 2 and rows[0.2]["false_alarms_on_normal_movement"] == 1
    assert rows[0.6]["episodes_found"] == 1 and rows[0.6]["false_alarms"] == 1


def test_load_ground_truth_clock_and_seconds(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("start,end,label\n00:10,00:10:20,episode\n3600,,normal\n")
    gt = load_ground_truth(p, parse_clock("23:50"))
    assert (gt[0].start_s, gt[0].end_s, gt[0].label) == (1200, 1220, "episode")
    assert (gt[1].start_s, gt[1].end_s, gt[1].label) == (3600, 3610, "normal")
