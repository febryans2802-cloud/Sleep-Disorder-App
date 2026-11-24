import numpy as np

from conftest import make_night_video, needs_ffmpeg, write_wav
from rbdcap.audio import compute_loudness, loudness_segments
from rbdcap.config import TriggerConfig
from rbdcap.motion import compute_motion, motion_segments


def test_loudness_trigger_finds_burst(tmp_path):
    sr = 16000
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.003, 60 * sr)
    x[30 * sr: 32 * sr] += 0.5 * np.sin(2 * np.pi * 300 * np.arange(2 * sr) / sr)
    wav = tmp_path / "a.wav"
    write_wav(wav, x, sr)
    segs = loudness_segments(compute_loudness(wav, 0.5, chunk_s=7), TriggerConfig())
    assert len(segs) == 1
    assert abs(segs[0].start_s - 30) < 0.6 and abs(segs[0].end_s - 32) < 0.6
    assert segs[0].loudness_peak_db > 30


def test_motion_trigger_ignores_sensor_noise(tmp_path):
    video = make_night_video(tmp_path / "n.avi", seconds=30, fps=10, move_times=[(12, 15)], loud_times=[],
                             with_audio=False)
    trace = compute_motion(video, TriggerConfig())
    assert abs(trace.info.duration_s - 30) < 0.5
    segs = motion_segments(trace, TriggerConfig())
    assert segs, "movement was not detected"
    assert all(11 <= s.start_s and s.end_s <= 16 for s in segs)
