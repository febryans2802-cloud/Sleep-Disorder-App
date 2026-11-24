import csv
import json

from conftest import make_night_video, needs_ffmpeg
from rbdcap.cli import main


@needs_ffmpeg
def test_end_to_end_without_models(tmp_path):
    video = make_night_video(tmp_path / "night.mp4", seconds=90, fps=10,
                             move_times=[(20, 23), (60, 62)], loud_times=[(60.5, 62)])
    out = tmp_path / "res"
    assert main(["analyze", str(video), "--out", str(out), "--start-time", "23:00",
                 "--models", str(tmp_path / "none"), "--top", "5"]) == 0
    rows = list(csv.DictReader(open(out / "episodes.csv")))
    assert len(rows) == 2
    # The segment with movement and a loud sound in the second half ranks first.
    assert float(rows[0]["start_s"]) > 50 and rows[0]["triggers"] == "motion+sound"
    assert rows[0]["start_clock"].startswith("23:00:5")
    assert (out / rows[0]["clip"]).exists()
    assert "Night review" in (out / "review.html").read_text()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["has_audio"] and not (out / "audio.wav").exists()

    gt = tmp_path / "gt.csv"
    gt.write_text("start,end,label\n23:00:20,23:00:23,normal\n23:01:00,23:01:02,episode\n")
    assert main(["evaluate", str(out / "episodes.csv"), str(gt)]) == 0
    ev = list(csv.DictReader(open(out / "evaluation.csv")))
    assert ev[0]["recall"] == "1.0"
