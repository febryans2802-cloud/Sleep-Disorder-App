import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def make_night_video(path: Path, seconds: float, fps: int, move_times, loud_times, with_audio: bool = True) -> Path:
    """A dim, noisy 'bedroom' with a bright block that jumps around during move_times
    and loud tone bursts during loud_times."""
    rng = np.random.default_rng(0)
    w, h = 320, 240
    silent = path.with_suffix(".silent.avi")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(int(seconds * fps)):
        t = i / fps
        frame = np.full((h, w, 3), 30, np.uint8)
        frame = np.clip(frame + rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)
        if any(a <= t < b for a, b in move_times):
            x = int(40 + 200 * ((i * 7) % 10) / 10)
            cv2.rectangle(frame, (x, 80), (x + 60, 160), (200, 200, 200), -1)
        else:
            cv2.rectangle(frame, (40, 80), (100, 160), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    if not with_audio:
        return silent
    sr = 16000
    tt = np.arange(int(seconds * sr)) / sr
    audio = rng.normal(0, 0.003, tt.size)
    for a, b in loud_times:
        sel = (tt >= a) & (tt < b)
        audio[sel] += 0.5 * np.sin(2 * np.pi * 300 * tt[sel])
    wav = path.with_suffix(".wav")
    write_wav(wav, audio, sr)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(wav),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
