"""Sound: a whole-night loudness trigger, then YAMNet sound classes per segment."""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import TriggerConfig
from .segments import Segment, blockwise_baseline, mask_to_intervals

SAMPLE_RATE = 16000  # what YAMNet expects

# YAMNet (AudioSet) class names grouped by what they mean for RBD.
# Names missing from a model version are simply ignored.
SOUND_GROUPS: dict[str, tuple[str, ...]] = {
    "shout": ("Shout", "Yell", "Screaming", "Battle cry", "Children shouting"),
    "speech": ("Speech", "Whispering", "Conversation", "Narration, monologue"),
    "groan": ("Groan", "Grunt", "Whimper", "Crying, sobbing", "Wail, moan"),
    "laugh": ("Laughter", "Giggle", "Chuckle, chortle"),
    "impact": ("Thump, thud", "Slap, smack", "Knock"),
    "snore": ("Snoring", "Snort", "Breathing", "Gasp"),
    "cough": ("Cough", "Sneeze", "Throat clearing"),
}
IGNORED_TOP_LABELS = {"Silence", "Inside, small room", "White noise", "Static", "Hum"}


def extract_audio(video_path: str | Path, wav_path: str | Path) -> bool:
    """Write a 16 kHz mono WAV next to the results. Returns False if there is no audio."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not on PATH") from exc
    if result.returncode != 0 or not Path(wav_path).exists():
        return False
    with wave.open(str(wav_path)) as w:
        return w.getnframes() > 0


def read_wav(path: str | Path, start_s: float = 0.0, end_s: float | None = None) -> tuple[np.ndarray, int]:
    """Read part of a 16-bit mono WAV as float32 in [-1, 1]."""
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected 16-bit mono WAV")
        n = w.getnframes()
        a = min(n, max(0, int(start_s * sr)))
        b = n if end_s is None else min(n, max(a, int(end_s * sr)))
        w.setpos(a)
        raw = w.readframes(b - a)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sr


@dataclass
class LoudnessTrace:
    times: np.ndarray
    dbfs: np.ndarray
    window_s: float


def compute_loudness(wav_path: str | Path, window_s: float, chunk_s: float = 60.0) -> LoudnessTrace:
    """RMS level per window, in dB relative to full scale, read in chunks."""
    with wave.open(str(wav_path)) as w:
        sr, n = w.getframerate(), w.getnframes()
    win = max(1, int(window_s * sr))
    chunk = max(win, int(chunk_s * sr) // win * win)
    times, levels = [], []
    for a in range(0, n, chunk):
        x, _ = read_wav(wav_path, a / sr, min(n, a + chunk) / sr)
        k = len(x) // win
        if k == 0:
            continue
        rms = np.sqrt(np.mean(x[: k * win].reshape(k, win) ** 2, axis=1))
        levels.append(20 * np.log10(rms + 1e-10))
        times.append(a / sr + np.arange(k) * win / sr)
    if not times:
        return LoudnessTrace(np.array([]), np.array([]), win / sr)
    return LoudnessTrace(np.concatenate(times), np.concatenate(levels), win / sr)


def loudness_segments(trace: LoudnessTrace, cfg: TriggerConfig) -> list[Segment]:
    if trace.dbfs.size == 0:
        return []
    # The 20th percentile of each block approximates the quiet room level.
    floor = blockwise_baseline(trace.dbfs, trace.times, cfg.baseline_block_s, q=20)
    above = trace.dbfs - floor
    mask = (above > cfg.loudness_db_above_floor) & (trace.dbfs > cfg.loudness_min_dbfs)
    segments = []
    for start, end in mask_to_intervals(trace.times, mask, trace.window_s):
        sel = (trace.times >= start) & (trace.times < end)
        segments.append(Segment(start, end, {"sound"}, loudness_peak_db=float(above[sel].max())))
    return segments


@dataclass
class SoundFeatures:
    groups: dict[str, float] = field(default_factory=dict)  # group -> max probability
    group_labels: dict[str, str] = field(default_factory=dict)  # group -> best class name
    top_label: str = ""
    top_score: float = 0.0


def summarise_scores(frames: list[dict[str, float]]) -> SoundFeatures:
    """Collapse per-frame {class name: probability} dicts into segment features."""
    feats = SoundFeatures(groups={g: 0.0 for g in SOUND_GROUPS})
    for scores in frames:
        for group, names in SOUND_GROUPS.items():
            for name in names:
                p = scores.get(name, 0.0)
                if p > feats.groups[group]:
                    feats.groups[group] = p
                    feats.group_labels[group] = name
        for name, p in scores.items():
            if name not in IGNORED_TOP_LABELS and p > feats.top_score:
                feats.top_label, feats.top_score = name, p
    return feats


class SoundClassifier:
    """YAMNet through MediaPipe's audio classifier (no TensorFlow needed)."""

    def __init__(self, model_path: str | Path):
        from mediapipe.tasks.python import BaseOptions, audio

        opts = audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=audio.RunningMode.AUDIO_CLIPS,
            max_results=-1,
        )
        self._clf = audio.AudioClassifier.create_from_options(opts)

    def __enter__(self) -> "SoundClassifier":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._clf is not None:
            self._clf.close()
            self._clf = None

    def classify(self, samples: np.ndarray, sr: int = SAMPLE_RATE) -> SoundFeatures:
        from mediapipe.tasks.python.components.containers import audio_data

        min_len = int(sr * 1.0)  # YAMNet needs about one second of audio
        if len(samples) < min_len:
            samples = np.pad(samples, (0, min_len - len(samples)))
        results = self._clf.classify(audio_data.AudioData.create_from_array(samples.astype(np.float32), sr))
        frames = [
            {c.category_name: float(c.score) for c in r.classifications[0].categories}
            for r in results
            if r.classifications
        ]
        return summarise_scores(frames)
