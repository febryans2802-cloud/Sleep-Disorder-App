"""Command line: download-models, analyze, evaluate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import Config
from .timeline import format_clock, guess_start_clock, parse_clock


def _clock_arg(text: str) -> float:
    try:
        return parse_clock(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def cmd_download(args) -> int:
    from .models import download_models

    download_models(args.models, pose_variant=args.pose_model, force=args.force)
    return 0


def cmd_analyze(args) -> int:
    from .pipeline import AnalyzeOptions, analyze

    video = Path(args.video)
    if not video.exists():
        print(f"error: {video} not found", file=sys.stderr)
        return 2
    start = args.start_time
    if start is None:
        start = guess_start_clock(video)
        if start is not None:
            print(f"recording start guessed from file metadata: {format_clock(start)} "
                  "(check it; pass --start-time HH:MM to override)")
        else:
            print("recording start unknown; clock times will be blank (pass --start-time HH:MM)")
    cfg = Config()
    cfg.pose.subject = args.subject
    cfg.report.top_clips = args.top
    opts = AnalyzeOptions(
        models_dir=Path(args.models),
        start_clock_s=start,
        lights_out_clock_s=args.lights_out,
        wake_up_clock_s=args.wake_up,
        pose_variant=args.pose_model,
        use_pose=not args.no_pose,
        use_sound_model=not args.no_sound_model,
        export_clips=not args.no_clips,
        keep_audio=args.keep_audio,
    )
    out = Path(args.out) if args.out else Path("results") / video.stem
    analyze(video, out, cfg, opts)
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate, format_table, load_detections, load_ground_truth

    episodes_csv = Path(args.episodes)
    summary_path = episodes_csv.with_name("summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    start = args.start_time if args.start_time is not None else summary.get("start_clock_s")
    detections = load_detections(episodes_csv)
    truth = load_ground_truth(args.ground_truth, start)
    hours = summary.get("duration_s", max([d.end_s for d in detections] + [0.0])) / 3600
    rows = evaluate(detections, truth, hours, tolerance_s=args.tolerance)
    print(f"{len(truth)} ground-truth rows, {len(detections)} detected segments, {hours:.2f} h recorded")
    print(format_table(rows))
    out = episodes_csv.with_name("evaluation.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rbdcap", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download-models", help="fetch the MediaPipe pose and YAMNet models")
    d.add_argument("--models", default="models", help="folder to store models (default: models)")
    d.add_argument("--pose-model", default="full", choices=["lite", "full", "heavy"])
    d.add_argument("--force", action="store_true", help="download again even if present")
    d.set_defaults(func=cmd_download)

    a = sub.add_parser("analyze", help="analyse one overnight recording")
    a.add_argument("video")
    a.add_argument("--out", help="output folder (default: results/<video name>)")
    a.add_argument("--models", default="models")
    a.add_argument("--start-time", type=_clock_arg, help="clock time the recording started, HH:MM[:SS]")
    a.add_argument("--lights-out", type=_clock_arg, help="when the sleeper went to bed (default: recording start)")
    a.add_argument("--wake-up", type=_clock_arg, help="when the sleeper got up (default: recording end)")
    a.add_argument("--subject", default="auto", choices=["auto", "left", "right"],
                   help="which side of the image the sleeper is on, if a bed partner is in frame")
    a.add_argument("--pose-model", choices=["lite", "full", "heavy"], help="default: best one downloaded")
    a.add_argument("--top", type=int, default=30, help="number of clips to export (default: 30)")
    a.add_argument("--no-pose", action="store_true", help="skip pose analysis")
    a.add_argument("--no-sound-model", action="store_true", help="skip YAMNet sound classes")
    a.add_argument("--no-clips", action="store_true", help="do not export video clips")
    a.add_argument("--keep-audio", action="store_true", help="keep the extracted audio.wav")
    a.set_defaults(func=cmd_analyze)

    e = sub.add_parser("evaluate", help="compare results with a log of acted episodes")
    e.add_argument("episodes", help="episodes.csv written by analyze")
    e.add_argument("ground_truth", help="CSV with columns start,end,label")
    e.add_argument("--start-time", type=_clock_arg, help="override the start time saved in summary.json")
    e.add_argument("--tolerance", type=float, default=15.0, help="seconds of slack when matching (default: 15)")
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
