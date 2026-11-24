"""Download the pretrained MediaPipe models the pipeline uses."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"
POSE_VARIANTS = ("lite", "full", "heavy")
SOUND_FILE = "yamnet.tflite"
SOUND_URL = f"{_BASE}/audio_classifier/yamnet/float32/1/yamnet.tflite"


def pose_file(variant: str) -> str:
    return f"pose_landmarker_{variant}.task"


def pose_url(variant: str) -> str:
    return f"{_BASE}/pose_landmarker/pose_landmarker_{variant}/float16/latest/{pose_file(variant)}"


def _download(url: str, dest: Path, log) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError(f"empty download from {url}")
    tmp.replace(dest)


def download_models(models_dir: str | Path, pose_variant: str = "full", force: bool = False, log=print) -> None:
    if pose_variant not in POSE_VARIANTS:
        raise ValueError(f"pose variant must be one of {POSE_VARIANTS}")
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, url in ((pose_file(pose_variant), pose_url(pose_variant)), (SOUND_FILE, SOUND_URL)):
        dest = d / name
        if dest.exists() and not force:
            log(f"{dest} already present")
            continue
        _download(url, dest, log)
    log("models ready")


def find_pose_model(models_dir: str | Path, variant: str | None = None) -> Path | None:
    """Return the requested pose model, or the best one present."""
    d = Path(models_dir)
    order = [variant] if variant else ["full", "heavy", "lite"]
    for v in order:
        p = d / pose_file(v)
        if p.exists():
            return p
    return None


def find_sound_model(models_dir: str | Path) -> Path | None:
    p = Path(models_dir) / SOUND_FILE
    return p if p.exists() else None
