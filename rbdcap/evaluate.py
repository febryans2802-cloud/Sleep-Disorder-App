"""Compare detected segments with a log of acted episodes from a test night.

Ground-truth CSV columns: start, end, label
  start/end: clock time (03:12:30) or seconds from the recording start
  label:     "episode" for acted dream enactment (default), "normal" for
             deliberate ordinary movement such as rolling over
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .timeline import parse_time_or_offset

DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass
class Interval:
    start_s: float
    end_s: float
    label: str = "episode"
    score: float = 0.0


def load_ground_truth(path: str | Path, start_clock_s: float | None, default_len_s: float = 10.0) -> list[Interval]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            start = parse_time_or_offset(row["start"], start_clock_s)
            end_text = (row.get("end") or "").strip()
            end = parse_time_or_offset(end_text, start_clock_s) if end_text else start + default_len_s
            label = (row.get("label") or "episode").strip().lower() or "episode"
            out.append(Interval(start, max(end, start), label))
    return out


def load_detections(episodes_csv: str | Path) -> list[Interval]:
    with open(episodes_csv, newline="") as f:
        return [Interval(float(r["start_s"]), float(r["end_s"]), "detection", float(r["score"]))
                for r in csv.DictReader(f)]


def _overlaps(a: Interval, b: Interval, tol: float) -> bool:
    return a.start_s <= b.end_s + tol and b.start_s <= a.end_s + tol


def evaluate(detections: list[Interval], truth: list[Interval], hours: float,
             thresholds=DEFAULT_THRESHOLDS, tolerance_s: float = 15.0) -> list[dict]:
    episodes = [t for t in truth if t.label == "episode"]
    normals = [t for t in truth if t.label != "episode"]
    rows = []
    for th in thresholds:
        kept = [d for d in detections if d.score >= th]
        found = sum(any(_overlaps(e, d, tolerance_s) for d in kept) for e in episodes)
        false_alarms = [d for d in kept if not any(_overlaps(d, e, tolerance_s) for e in episodes)]
        on_normal = sum(any(_overlaps(d, n, tolerance_s) for n in normals) for d in false_alarms)
        rows.append({
            "threshold": th,
            "detections": len(kept),
            "episodes_found": found,
            "episodes_total": len(episodes),
            "recall": round(found / len(episodes), 3) if episodes else None,
            "false_alarms": len(false_alarms),
            "false_alarms_on_normal_movement": on_normal,
            "false_alarms_per_hour": round(len(false_alarms) / hours, 2) if hours > 0 else None,
        })
    return rows


def format_table(rows: list[dict]) -> str:
    head = f"{'threshold':>9} {'found':>9} {'recall':>7} {'false alarms':>13} {'on normal':>10} {'per hour':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        recall = "-" if r["recall"] is None else f"{r['recall']:.2f}"
        per_h = "-" if r["false_alarms_per_hour"] is None else f"{r['false_alarms_per_hour']:.2f}"
        lines.append(
            f"{r['threshold']:>9.2f} {r['episodes_found']:>4}/{r['episodes_total']:<4} {recall:>7} "
            f"{r['false_alarms']:>13} {r['false_alarms_on_normal_movement']:>10} {per_h:>9}"
        )
    return "\n".join(lines)
