"""Run the whole analysis for one overnight recording."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import report
from .audio import SoundClassifier, compute_loudness, extract_audio, loudness_segments, read_wav
from .config import Config
from .models import find_pose_model, find_sound_model
from .motion import compute_motion, motion_segments, open_video
from .pose import PoseTracker, pose_features
from .scoring import Episode, score_episode
from .segments import merge_segments
from .timeline import clock_to_offset


@dataclass
class AnalyzeOptions:
    models_dir: Path = Path("models")
    start_clock_s: float | None = None  # wall-clock time the recording started
    lights_out_clock_s: float | None = None
    wake_up_clock_s: float | None = None
    pose_variant: str | None = None
    use_pose: bool = True
    use_sound_model: bool = True
    export_clips: bool = True
    keep_audio: bool = False


def _night_bounds(duration_s: float, opts: AnalyzeOptions) -> tuple[float, float]:
    start, end = 0.0, duration_s
    if opts.start_clock_s is not None:
        if opts.lights_out_clock_s is not None:
            start = clock_to_offset(opts.lights_out_clock_s, opts.start_clock_s)
        if opts.wake_up_clock_s is not None:
            end = clock_to_offset(opts.wake_up_clock_s, opts.start_clock_s)
    if end <= start:
        start, end = 0.0, duration_s
    return start, end


def analyze(video: str | Path, out_dir: str | Path, cfg: Config, opts: AnalyzeOptions, log=print) -> list[Episode]:
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("[1/5] scanning the night for motion")
    trace = compute_motion(video, cfg.trigger, progress=lambda f: log(f"      {f:5.0%}"))
    info = trace.info
    log(f"      {info.duration_s / 3600:.2f} h at {info.fps:.1f} fps, {info.width}x{info.height}")
    segments = motion_segments(trace, cfg.trigger)
    n_motion = len(segments)

    log("[2/5] scanning the night for sound")
    wav = out_dir / "audio.wav"
    has_audio = extract_audio(video, wav)
    n_sound = 0
    if has_audio:
        loud = compute_loudness(wav, cfg.trigger.loudness_window_s)
        sound_segs = loudness_segments(loud, cfg.trigger)
        n_sound = len(sound_segs)
        segments += sound_segs
    else:
        log("      no audio track found; continuing with video only")

    t = cfg.trigger
    segments = merge_segments(
        segments, gap_s=t.merge_gap_s, pad_s=t.pad_s, duration_s=info.duration_s,
        min_trigger_s=t.min_trigger_s, max_segment_s=t.max_segment_s,
    )
    log(f"      {n_motion} motion + {n_sound} sound triggers -> {len(segments)} segments to analyse")

    night_start, night_end = _night_bounds(info.duration_s, opts)
    episodes = [
        Episode(s, None, None, (s.mid_s - night_start) / (night_end - night_start)) for s in segments
    ]

    log("[3/5] body pose in each segment")
    pose_model = find_pose_model(opts.models_dir, opts.pose_variant) if opts.use_pose else None
    if opts.use_pose and pose_model is None:
        log("      no pose model found; run `python -m rbdcap download-models` (continuing without pose)")
    if pose_model is not None and episodes:
        cap = open_video(video)
        try:
            with PoseTracker(pose_model, cfg.pose) as tracker:
                for i, ep in enumerate(episodes, 1):
                    times, track = tracker.track(cap, info.fps, ep.segment.start_s, ep.segment.end_s)
                    if len(times):
                        ep.pose = pose_features(times, track, cfg.pose)
                    if i % 20 == 0:
                        log(f"      {i}/{len(episodes)}")
        finally:
            cap.release()

    log("[4/5] sound classes in each segment")
    sound_model = find_sound_model(opts.models_dir) if (opts.use_sound_model and has_audio) else None
    if opts.use_sound_model and has_audio and sound_model is None:
        log("      no sound model found; run `python -m rbdcap download-models` (continuing without it)")
    if sound_model is not None and episodes:
        with SoundClassifier(sound_model) as clf:
            for ep in episodes:
                samples, sr = read_wav(wav, ep.segment.start_s, ep.segment.end_s)
                ep.sound = clf.classify(samples, sr)

    for ep in episodes:
        ep.score, ep.reasons = score_episode(ep.segment, ep.pose, ep.sound, ep.night_fraction, cfg.scoring)
    episodes.sort(key=lambda e: e.score, reverse=True)

    log("[5/5] writing results")
    clip_names: dict[int, str] = {}
    if opts.export_clips:
        clip_names = report.export_clips(video, episodes, out_dir / "clips", cfg.report, info.duration_s)
    report.write_episodes_csv(out_dir / "episodes.csv", episodes, opts.start_clock_s, clip_names)
    report.write_review_html(out_dir / "review.html", episodes, opts.start_clock_s, info.duration_s,
                             clip_names, title=video.name)
    summary = {
        "video": str(video),
        "duration_s": info.duration_s,
        "fps": info.fps,
        "resolution": [info.width, info.height],
        "start_clock_s": opts.start_clock_s,
        "night_bounds_s": [night_start, night_end],
        "has_audio": has_audio,
        "pose_model": str(pose_model) if pose_model else None,
        "sound_model": str(sound_model) if sound_model else None,
        "segments": len(episodes),
        "runtime_s": round(time.time() - t0, 1),
        "config": cfg.to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if has_audio and not opts.keep_audio:
        wav.unlink(missing_ok=True)
    log(f"done in {summary['runtime_s']} s; open {out_dir / 'review.html'} to review")
    return episodes
