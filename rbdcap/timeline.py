"""Convert between clock times (23:40) and seconds since the recording started."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

DAY_S = 24 * 3600


def parse_clock(text: str) -> float:
    """'23:40' or '23:40:15' -> seconds since midnight."""
    parts = text.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"expected HH:MM or HH:MM:SS, got {text!r}")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2]) if len(parts) == 3 else 0.0
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"invalid clock time {text!r}")
    return h * 3600 + m * 60 + s


def format_clock(seconds_of_day: float) -> str:
    s = int(round(seconds_of_day)) % DAY_S
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def clock_to_offset(clock_s: float, start_clock_s: float) -> float:
    """Seconds after the recording start, allowing the night to cross midnight."""
    offset = clock_s - start_clock_s
    if offset < -60:  # small negatives are clock jitter at the very start
        offset += DAY_S
    return offset


def offset_to_clock(offset_s: float, start_clock_s: float | None) -> str:
    if start_clock_s is None:
        return ""
    return format_clock(start_clock_s + offset_s)


def parse_time_or_offset(text: str, start_clock_s: float | None) -> float:
    """Ground-truth times may be a clock time ('03:12') or plain seconds ('11520')."""
    text = text.strip()
    if ":" in text:
        if start_clock_s is None:
            raise ValueError(f"clock time {text!r} needs the recording start time (--start-time)")
        return clock_to_offset(parse_clock(text), start_clock_s)
    return float(text)


def guess_start_clock(video_path: str | Path) -> float | None:
    """Read the file's creation_time tag and convert it to local clock time.

    Phones differ: some write the time recording started, some when it stopped,
    and the tag is in UTC. Always check the value this prints.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        tag = json.loads(out)["format"]["tags"]["creation_time"]
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        return None
    try:
        dt = datetime.fromisoformat(tag.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return None
    return dt.hour * 3600 + dt.minute * 60 + dt.second
