"""Outputs: ranked CSV, video clips, and a review page for tagging clips."""

from __future__ import annotations

import csv
import html
import json
import subprocess
from pathlib import Path

from .config import ReportConfig
from .scoring import Episode
from .timeline import offset_to_clock

TAGS = ("dream enactment", "normal movement", "bed partner", "noise / other", "not sure")

CSV_FIELDS = [
    "rank", "start_clock", "end_clock", "start_s", "end_s", "duration_s", "score", "triggers",
    "peak_limb_speed", "limb_vs_trunk", "fast_fraction", "detected_fraction", "arm_reach",
    "shout", "speech", "groan", "laugh", "impact", "snore", "cough", "top_sound",
    "night_fraction", "reasons", "clip",
]


def _r(x: float | None, nd: int = 3):
    return "" if x is None else round(float(x), nd)


def write_episodes_csv(path: Path, episodes: list[Episode], start_clock_s: float | None,
                       clip_names: dict[int, str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, ep in enumerate(episodes):
            s, p, snd = ep.segment, ep.pose, ep.sound
            g = snd.groups if snd else {}
            w.writerow({
                "rank": i + 1,
                "start_clock": offset_to_clock(s.start_s, start_clock_s),
                "end_clock": offset_to_clock(s.end_s, start_clock_s),
                "start_s": _r(s.start_s, 1), "end_s": _r(s.end_s, 1), "duration_s": _r(s.duration_s, 1),
                "score": _r(ep.score), "triggers": "+".join(sorted(s.sources)),
                "peak_limb_speed": _r(p.peak_limb_speed) if p else "",
                "limb_vs_trunk": _r(p.limb_vs_trunk) if p else "",
                "fast_fraction": _r(p.fast_fraction) if p else "",
                "detected_fraction": _r(p.detected_fraction) if p else "",
                "arm_reach": _r(p.arm_reach) if p else "",
                **{k: _r(g.get(k)) if snd else "" for k in ("shout", "speech", "groan", "laugh", "impact", "snore", "cough")},
                "top_sound": snd.top_label if snd else "",
                "night_fraction": _r(ep.night_fraction),
                "reasons": "; ".join(ep.reasons),
                "clip": clip_names.get(i, ""),
            })


def export_clips(video: Path, episodes: list[Episode], clips_dir: Path, cfg: ReportConfig,
                 duration_s: float) -> dict[int, str]:
    """Cut the top-ranked segments into small MP4s. Returns {rank index: relative path}."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    names: dict[int, str] = {}
    for i, ep in enumerate(episodes[: cfg.top_clips]):
        a = max(0.0, ep.segment.start_s - cfg.clip_before_s)
        b = min(duration_s, ep.segment.end_s + cfg.clip_after_s)
        name = f"{i + 1:03d}_{int(ep.segment.start_s)}s.mp4"
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{a:.2f}", "-i", str(video),
            "-t", f"{b - a:.2f}", "-vf", f"scale=-2:min({cfg.clip_height}\\,ih)",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(clips_dir / name),
        ]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            names[i] = f"clips/{name}"
    return names


def write_review_html(path: Path, episodes: list[Episode], start_clock_s: float | None,
                      duration_s: float, clip_names: dict[int, str], title: str) -> None:
    """A self-contained page: night timeline, ranked clips, a tag per clip, CSV export."""
    rows = []
    for i, ep in enumerate(episodes):
        s = ep.segment
        rows.append({
            "rank": i + 1,
            "start_s": round(s.start_s, 1),
            "end_s": round(s.end_s, 1),
            "clock": offset_to_clock(s.start_s, start_clock_s),
            "score": round(ep.score, 3),
            "reasons": "; ".join(ep.reasons) or "trigger only",
            "clip": clip_names.get(i, ""),
        })
    data = json.dumps({"title": title, "duration": duration_s, "rows": rows, "tags": TAGS})
    data = data.replace("</", "<\\/")  # keep the JSON from closing the script tag
    page = _HTML.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)
    path.write_text(page, encoding="utf-8")


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Night review</title>
<style>
:root { --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --line:#e7e5e4; --card:#fff; --hi:#b91c1c; --mid:#d97706; --lo:#a8a29e; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --line:#44403c; --card:#292524; } }
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.note { color: var(--muted); margin: 0 0 16px; }
svg.tl { width:100%; height:56px; display:block; margin-bottom:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:12px;
        display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1.3fr); gap:12px; }
@media (max-width: 640px) { .card { grid-template-columns: 1fr; } }
video { width:100%; border-radius:6px; background:#000; }
.meta b { font-size: 16px; }
.score { font-variant-numeric: tabular-nums; }
.tags { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.tags label { border:1px solid var(--line); border-radius:999px; padding:2px 10px; cursor:pointer; font-size:13px; }
.tags input { margin-right:4px; }
.bar { position:sticky; top:0; background:var(--bg); padding:8px 0; display:flex; gap:12px; align-items:center; z-index:1; }
button { font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }
.muted { color: var(--muted); }
</style></head>
<body><main>
<h1>Night review: __TITLE__</h1>
<p class="note">Clips are ranked by a rule-based score for review only. The score is not a diagnosis.
Tag each clip; the tags become labelled data.</p>
<div class="bar"><button id="export">Export tags (CSV)</button><span id="count" class="muted"></span></div>
<svg class="tl" id="tl" viewBox="0 0 1000 56" preserveAspectRatio="none" role="img" aria-label="Segments across the night"></svg>
<div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const key = "rbdcap-tags:" + DATA.title;
let tags = {};
try { tags = JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) {}
function save() { try { localStorage.setItem(key, JSON.stringify(tags)); } catch (e) {} count(); }
function color(s) { return s >= 0.6 ? "var(--hi)" : s >= 0.35 ? "var(--mid)" : "var(--lo)"; }
function esc(t) { const d = document.createElement("div"); d.textContent = t; return d.innerHTML; }
function count() {
  const n = Object.keys(tags).length;
  document.getElementById("count").textContent = n + " of " + DATA.rows.length + " segments tagged";
}
const tl = document.getElementById("tl");
let svg = '<line x1="0" x2="1000" y1="40" y2="40" stroke="var(--line)" stroke-width="2"/>';
for (const r of DATA.rows) {
  const x = 1000 * r.start_s / DATA.duration, h = 8 + 28 * r.score;
  svg += `<rect x="${x.toFixed(1)}" y="${(40 - h).toFixed(1)}" width="3" height="${h.toFixed(1)}" fill="${color(r.score)}"><title>#${r.rank} ${r.clock} score ${r.score}</title></rect>`;
}
tl.innerHTML = svg;
const list = document.getElementById("list");
for (const r of DATA.rows) {
  if (!r.clip) continue;
  const id = String(r.start_s);
  const opts = DATA.tags.map(t => `<label><input type="radio" name="t${r.rank}" value="${t}" ${tags[id] && tags[id].tag === t ? "checked" : ""}>${t}</label>`).join("");
  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML = `<video controls preload="none" src="${r.clip}"></video>
    <div class="meta"><b>#${r.rank}</b> <span class="muted">${r.clock || (r.start_s + " s")}</span>
    &middot; score <span class="score" style="color:${color(r.score)}">${r.score.toFixed(2)}</span>
    <div>${esc(r.reasons)}</div><div class="tags">${opts}</div></div>`;
  div.querySelectorAll("input").forEach(inp => inp.addEventListener("change", () => {
    tags[id] = { rank: r.rank, start_s: r.start_s, end_s: r.end_s, clock: r.clock, score: r.score, tag: inp.value };
    save();
  }));
  list.appendChild(div);
}
const unclipped = DATA.rows.filter(r => !r.clip).length;
if (unclipped) list.insertAdjacentHTML("beforeend", `<p class="muted">${unclipped} lower-ranked segments have no clip; see episodes.csv.</p>`);
count();
document.getElementById("export").addEventListener("click", () => {
  const head = "rank,start_s,end_s,clock,score,tag\\n";
  const body = Object.values(tags).sort((a, b) => a.start_s - b.start_s)
    .map(t => [t.rank, t.start_s, t.end_s, t.clock, t.score, '"' + t.tag + '"'].join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([head + body + "\\n"], { type: "text/csv" }));
  a.download = "tags.csv";
  a.click();
});
</script>
</body></html>
"""
