# Sleep Disorder App: bedside RBD capture (student prototype)

This prototype flags moments in an overnight phone recording that **might** be
REM sleep behavior disorder (RBD) episodes: acting out dreams by punching,
kicking or shouting. It ranks clips for a person to review and tag.

> **It does not diagnose anything.** RBD is confirmed only by video-polysomnography
> in a sleep lab. The score here ranks clips for review. It is not a probability
> of RBD and should never be shown as one.

## How it works

```
phone video (+ audio)
  -> 1. motion scan       whole night, frame differencing at 5 fps
  -> 2. loudness scan     whole night, dB above the room's noise floor
  -> 3. segments          triggers merged, padded, long ones split
  -> 4. per segment:      body pose (MediaPipe Pose Landmarker)
                          sound classes (YAMNet via MediaPipe)
  -> 5. rule-based score  fast limbs, limbs faster than trunk, vocalisation,
                          impact sounds, late in the night; snoring/coughing penalised
  -> episodes.csv, clips/, review.html (tag each clip, export tags.csv)
```

Every model is pretrained on general data, so no RBD training data is needed.
The tags you export from `review.html` become the labelled data for a better
model later.

Movement is measured in **torso lengths per second**, so it doesn't depend on
how far the phone is from the bed. The pose step smooths landmark jitter,
ignores frames where the tracker jumps to another person, and handles the pose
model swapping left and right limbs.

## Setup

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on your PATH.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m rbdcap download-models          # ~15 MB into ./models
python -m pytest                          # optional: run the tests
```

On a minimal Linux install, MediaPipe may also need `sudo apt install libegl1`.

## Recording a night

- **Phone:** a spare Android phone on a stand beside the bed, **plugged in**, in
  do-not-disturb mode. Android is easier than iOS for all-night recording.
- **Framing:** landscape, about 1.5–2 m from the bed at or above mattress height,
  the whole body in frame. If a bed partner is in frame, note which side of the
  image the sleeper is on (`--subject left|right`).
- **Light:** phone cameras block infrared, so use a dim warm night-light. Before
  anything else, record 5 minutes in that light, run the analysis, and check that
  `detected_fraction` in `episodes.csv` is high. If the body isn't detected, add
  light or move to an infrared camera.
- **Settings:** about 480p at 10 fps is enough; higher settings work but make
  bigger files. Any camera app that can record 8+ hours at a chosen resolution
  and frame rate will do (Open Camera on Android offers these options; check
  your version). If the app splits the night into several files, analyse each
  with its own `--start-time`.
- **Write down** the clock time when recording starts.

## Analysing a night

```bash
python -m rbdcap analyze recordings/night1.mp4 --start-time 23:12 \
    --lights-out 23:20 --wake-up 07:05 --subject auto
```

Results go to `results/night1/`:

| File | What it is |
| --- | --- |
| `review.html` | Night timeline and ranked clips. Tag each clip, then **Export tags (CSV)** |
| `episodes.csv` | Every segment with its score, features and reasons, highest score first |
| `clips/` | The top 30 segments as short MP4s (`--top` to change) |
| `summary.json` | Recording info, runtime and every threshold used |

A 4-core laptop scans an 8-hour 480p night in roughly 15 minutes, plus pose
analysis time for each flagged segment.

Useful flags: `--no-pose`, `--no-sound-model`, `--no-clips`, `--pose-model lite|full|heavy`,
`--keep-audio`.

## Testing on acted nights (the first milestone)

1. A team member records 3–5 nights. During each night they act out several
   short dream-enactment movements (punching, kicking, shouting) and a few
   deliberate *normal* movements (rolling over, pulling up the blanket). They
   write down the clock time of each.
2. Put the log in a CSV like [`examples/ground_truth_template.csv`](examples/ground_truth_template.csv):
   `label` is `episode` for acted enactment and `normal` for ordinary movement.
   Times can be clock times or seconds from the recording start.
3. Run:

   ```bash
   python -m rbdcap evaluate results/night1/episodes.csv night1_log.csv
   ```

   It prints, for several score thresholds, how many acted episodes were found
   (recall) and how many false alarms per hour were raised, including how many
   were on deliberate normal movement. It also saves `evaluation.csv`.

This shows whether the **pipeline** works. It does not show that it detects
real RBD, because acted movements differ from real episodes. Say so in any
write-up. Real validation needs labelled video from a sleep lab's
video-polysomnography, with ethics approval.

## Tuning

All thresholds live in [`rbdcap/config.py`](rbdcap/config.py), with comments.
They are starting guesses. Change one thing at a time, re-run `evaluate`, and
record what changed. `summary.json` saves the exact config used for each run.

## Privacy and ethics

- Recordings never leave the team's own phone and laptop. `.gitignore` excludes
  videos, audio, models and results: **never commit recordings**.
- Everyone who sleeps in the room must consent to being recorded.
- Recording anyone outside the team (patients, volunteers) needs approval from
  your university's research ethics committee first.

## Project layout

```
rbdcap/
  cli.py        command line: download-models, analyze, evaluate
  pipeline.py   runs the five steps for one recording
  motion.py     whole-night motion trigger
  audio.py      loudness trigger, YAMNet sound classes
  segments.py   triggers -> merged segments
  pose.py       pose tracking and movement features
  scoring.py    rule-based score and its reasons
  report.py     CSV, clips, review page
  evaluate.py   comparison with an acted-episode log
  config.py     every threshold
tests/          unit tests and an end-to-end test on a synthetic video
```

## Known limitations

- Untested on real sleepers so far: thresholds come from synthetic data and
  public footage of walking people, not bedrooms.
- Dim-light pose detection is the biggest risk; check it on night one.
- Blankets hide limbs. The pipeline still flags covered movement as "movement
  seen, but no body detected", with lower credit.
- "Second half of the night" stands in for REM sleep. A phone cannot measure
  sleep stages.
