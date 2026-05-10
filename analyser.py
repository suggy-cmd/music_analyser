#!/usr/bin/env python3
"""
Music transition point analyser for video editing.
Detects hi-hats, claps, and beats that make good cut points.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np


# Frequency bands for percussion detection (Hz)
HIHAT_FMIN = 6000
HIHAT_FMAX = 16000
SNARE_FMIN = 150
SNARE_FMAX = 5000
KICK_FMIN = 40
KICK_FMAX = 200


def load_audio(path: str, target_sr: int = 22050):
    print(f"Loading: {path}")
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duration: {duration:.1f}s  |  Sample rate: {sr}Hz")
    return y, sr, duration


def detect_beats(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Detect downbeats (bar 1 beats) using beat strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strengths = onset_env[beat_frames]

    # Every 4th beat is typically the strongest (downbeat) — use autocorrelation to confirm
    # Fall back to every 4th beat if heuristic fails
    try:
        bpm_float = float(np.atleast_1d(tempo)[0])
        beats_per_bar = 4
        downbeat_indices = np.arange(0, len(beat_times), beats_per_bar)
        downbeat_times = beat_times[downbeat_indices]
    except Exception:
        downbeat_times = beat_times[::4]
        bpm_float = 120.0

    return {
        "bpm": round(float(np.atleast_1d(tempo)[0]), 1),
        "beat_times": beat_times,
        "beat_strengths": beat_strengths,
        "downbeat_times": downbeat_times,
    }


def detect_band_onsets(y, sr, fmin, fmax, delta=0.07):
    """Detect onsets within a specific frequency band."""
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        fmin=fmin,
        fmax=min(fmax, sr // 2),
        aggregate=np.median,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        delta=delta,
        wait=2,  # minimum 2 frames between onsets (~46ms at hop_length=512)
    )
    times = librosa.frames_to_time(onset_frames, sr=sr)
    strengths = onset_env[onset_frames] if len(onset_frames) else np.array([])
    return times, strengths


def _norm(arr):
    """Normalise a strength array to 0–1 using its own max, so sources are comparable."""
    m = float(arr.max()) if len(arr) and arr.max() > 0 else 1.0
    return arr / m


def build_transition_list(beat_data, hihat_times, hihat_strengths,
                          snare_times, snare_strengths, min_gap: float = 0.1):
    """
    Merge all candidates, score them, and deduplicate within min_gap seconds.
    Returns list of dicts sorted by time.
    """
    candidates = []

    # Normalise each source independently so their strengths are on the same 0–1 scale
    # before the type multipliers are applied.
    beat_s = _norm(beat_data["beat_strengths"])
    hihat_s = _norm(hihat_strengths)
    snare_s = _norm(snare_strengths)

    # Add beats
    for t, s in zip(beat_data["beat_times"], beat_s):
        is_down = any(abs(t - d) < 0.05 for d in beat_data["downbeat_times"])
        candidates.append({
            "time": float(t),
            "type": "downbeat" if is_down else "beat",
            "strength": float(s),
            "score": float(s) * (3.0 if is_down else 1.5),
        })

    # Add hi-hats
    for t, s in zip(hihat_times, hihat_s):
        candidates.append({
            "time": float(t),
            "type": "hihat",
            "strength": float(s),
            "score": float(s) * 1.0,
        })

    # Add snares/claps
    for t, s in zip(snare_times, snare_s):
        candidates.append({
            "time": float(t),
            "type": "snare",
            "strength": float(s),
            "score": float(s) * 1.0,
        })

    if not candidates:
        return []

    # Sort by time
    candidates.sort(key=lambda x: x["time"])

    # If a hi-hat or snare lands on/near a beat, boost score and merge type
    merged = []
    used = set()
    for i, c in enumerate(candidates):
        if i in used:
            continue
        # Collect all candidates within min_gap
        cluster = [c]
        for j, other in enumerate(candidates[i + 1:], start=i + 1):
            if other["time"] - c["time"] <= min_gap:
                cluster.append(other)
                used.add(j)
            else:
                break
        used.add(i)

        # Merge cluster: pick highest score, combine types
        best = max(cluster, key=lambda x: x["score"])
        types = list(dict.fromkeys(x["type"] for x in cluster))  # unique, ordered
        merged.append({
            "time": round(best["time"], 4),
            "type": "+".join(types) if len(types) > 1 else types[0],
            "score": round(best["score"], 3),
            "strength": round(best["strength"], 3),
        })

    # Normalise scores 0–1
    scores = [m["score"] for m in merged]
    max_score = max(scores) if scores else 1.0
    for m in merged:
        m["score"] = round(m["score"] / max_score, 3)

    return merged


def apply_min_clip_length(transitions, min_clip: float):
    """Remove transitions closer than min_clip seconds to a higher-scored neighbour."""
    sorted_by_score = sorted(transitions, key=lambda x: -x["score"])
    kept = []
    for t in sorted_by_score:
        if all(abs(t["time"] - k["time"]) >= min_clip for k in kept):
            kept.append(t)
    kept.sort(key=lambda x: x["time"])
    return kept


def apply_clip_bounds(transitions, min_clip: float, max_clip: float):
    """
    Select transitions so clip lengths fall within [min_clip, max_clip].

    Top-quartile scored transitions are eligible at min_clip distance (fast cuts).
    Lower-scored transitions require a progressively larger gap before they are
    kept, which creates natural variety rather than uniform minimum-boundary clips.
    Any gap that still exceeds max_clip gets the best available transition inserted.
    """
    if not transitions:
        return []

    scores = sorted([t["score"] for t in transitions], reverse=True)
    score_75th = scores[max(0, len(scores) // 4)]
    mid_gap = (min_clip + max_clip) / 2.0

    def eff_min(score):
        if score_75th <= 0 or score >= score_75th:
            return min_clip
        frac = score / score_75th          # 0..1
        return mid_gap - frac * (mid_gap - min_clip)

    sorted_by_score = sorted(transitions, key=lambda x: -x["score"])
    kept = []
    for t in sorted_by_score:
        eg = eff_min(t["score"])
        if all(abs(t["time"] - k["time"]) >= eg for k in kept):
            kept.append(t)
    kept.sort(key=lambda x: x["time"])

    # Fill remaining gaps > max_clip with the best available transition
    changed = True
    while changed:
        changed = False
        kept_times = {t["time"] for t in kept}
        prev = 0.0
        for t in kept:
            if t["time"] - prev > max_clip:
                lo, hi = prev + min_clip, t["time"] - min_clip
                cands = [
                    c for c in transitions
                    if lo <= c["time"] <= hi and c["time"] not in kept_times
                ]
                if cands:
                    best = max(cands, key=lambda x: x["score"])
                    kept.append(best)
                    kept.sort(key=lambda x: x["time"])
                    changed = True
                    break
            prev = t["time"]

    return kept


def snap_to_beats(transitions, beat_times, tolerance: float = 0.060):
    """Snap each transition to the nearest beat within tolerance seconds.
    Transitions with no beat within tolerance are dropped — they are off-beat
    percussion hits that would produce a misaligned cut."""
    result = []
    for t in transitions:
        diffs = np.abs(beat_times - t["time"])
        idx = int(diffs.argmin())
        if diffs[idx] <= tolerance:
            t = dict(t)
            t["time"] = round(float(beat_times[idx]), 4)
            result.append(t)
        # else: discard — no beat close enough to make a clean cut
    return result


def format_timecode(seconds: float, fps: float = None) -> str:
    m = int(seconds) // 60
    s = seconds - m * 60
    if fps:
        frame = int((s % 1) * fps)
        return f"{m:02d}:{int(s):02d}:{frame:02d}"
    return f"{m:02d}:{s:06.3f}"


def filter_top(transitions, top_n: int = None, min_score: float = 0.0,
               every_n_beats: int = 1, beat_times=None):
    """Optional filtering to thin out results."""
    result = [t for t in transitions if t["score"] >= min_score]

    if every_n_beats > 1 and beat_times is not None:
        # Keep only transitions that land near every Nth beat
        target_beats = beat_times[::every_n_beats]
        result = [
            t for t in result
            if any(abs(t["time"] - bt) < 0.08 for bt in target_beats)
        ]

    if top_n:
        # Keep top N by score, re-sort by time
        result = sorted(result, key=lambda x: -x["score"])[:top_n]
        result.sort(key=lambda x: x["time"])

    return result


def export_text(transitions, fps, path):
    with open(path, "w") as f:
        f.write(f"{'Timecode':<14} {'Time (s)':<12} {'Type':<20} {'Score'}\n")
        f.write("-" * 56 + "\n")
        for t in transitions:
            tc = format_timecode(t["time"], fps)
            f.write(f"{tc:<14} {t['time']:<12.4f} {t['type']:<20} {t['score']}\n")


def export_csv(transitions, fps, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timecode", "time_seconds", "type", "score", "strength"]
        )
        writer.writeheader()
        for t in transitions:
            writer.writerow({
                "timecode": format_timecode(t["time"], fps),
                "time_seconds": t["time"],
                "type": t["type"],
                "score": t["score"],
                "strength": t["strength"],
            })


def export_json(transitions, fps, bpm, path):
    data = {
        "bpm": bpm,
        "fps": fps,
        "count": len(transitions),
        "transitions": [
            {**t, "timecode": format_timecode(t["time"], fps)}
            for t in transitions
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_timing_plan(transitions, duration, path):
    """Write a clip-length timing plan: clip number + length in seconds."""
    clips = []
    times = [0.0] + [t["time"] for t in sorted(transitions, key=lambda x: x["time"])] + [duration]
    for i in range(len(times) - 1):
        clips.append({
            "clip_number": i + 1,
            "length_seconds": round(times[i + 1] - times[i], 3),
        })
    with open(path, "w") as f:
        json.dump({"timing_plan": clips}, f, indent=2)


def export_edl_markers(transitions, fps, path):
    """Simple marker list compatible with DaVinci Resolve / Premiere import."""
    with open(path, "w") as f:
        f.write("Marker\tTimecode\tDuration\tName\tComment\n")
        for i, t in enumerate(transitions, 1):
            tc = format_timecode(t["time"], fps)
            f.write(f"{i}\t{tc}\t00:00:01\t{t['type']}\tscore={t['score']}\n")


def _find_font() -> str:
    candidates = [
        "/Library/Fonts/Inter_18pt-Regular.ttf",
        "/Library/Fonts/Inter-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Geneva.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    # Last resort: first .ttf anywhere in /Library/Fonts
    import glob
    hits = glob.glob("/Library/Fonts/*.ttf")
    return hits[0] if hits else ""


def export_video(transitions, audio_path, duration, output_path, width=1280, height=720, fps: int = 25, bpm: float = None):
    """Generate a styled HUD video: clip number, type, score, duration, IN/OUT, progress bar."""
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("Error: ffmpeg not found — run: pip install imageio-ffmpeg", file=sys.stderr)
        return False

    sorted_t = sorted(transitions, key=lambda x: x["time"])
    palette = ["0x5C0F0F", "0x0F0F52", "0x0D3D52", "0x31094A"]

    segments = []
    prev_time = 0.0
    for idx, t in enumerate(sorted_t):
        seg_dur = round(t["time"] - prev_time, 6)
        if seg_dur >= 0.001:
            segments.append({
                "color": palette[idx % len(palette)],
                "duration": seg_dur,
                "start_time": prev_time,
                "clip_num": idx + 1,
                "tr_in": sorted_t[idx - 1] if idx > 0 else None,
            })
        prev_time = t["time"]

    final_dur = round(duration - prev_time, 6)
    if final_dur >= 0.001:
        idx = len(sorted_t)
        segments.append({
            "color": palette[idx % len(palette)],
            "duration": final_dur,
            "start_time": prev_time,
            "clip_num": idx + 1,
            "tr_in": sorted_t[-1] if sorted_t else None,
        })

    if not segments:
        segments = [{"color": palette[0], "duration": duration, "start_time": 0.0,
                     "clip_num": 1, "tr_in": None}]

    total_clips = len(segments)
    font = _find_font()
    ff = f":fontfile='{font}'" if font else ""

    def esc(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")

    # Panel layout
    pw, ph = 590, 290
    tb_h = 36
    px = 30
    py = height - 30 - ph

    CYAN   = "0x5EC8DC"
    AMBER  = "0xFFAA00"
    RED_S  = "0xFF6655"
    GREY   = "0x888888"

    filter_parts = []
    labels = []

    for i, seg in enumerate(segments):
        seg_dur   = seg["duration"]
        start_t   = seg["start_time"]
        clip_num  = seg["clip_num"]
        tr_in     = seg["tr_in"]

        trans_type = tr_in["type"].upper() if tr_in else "START"
        score_val  = tr_in["score"] if tr_in else 0.0
        sc = CYAN if score_val >= 0.7 else (AMBER if score_val >= 0.4 else (RED_S if tr_in else GREY))

        in_str    = esc(f"{start_t:.2f}s")
        out_str   = esc(f"{start_t + seg_dur:.2f}s")
        dur_str   = esc(f"{seg_dur:.2f}s")
        score_str = esc(f"{score_val:.2f}")
        type_str  = esc(trans_type)
        clip_str  = esc(f"CLIP {clip_num}")
        ctr_str   = esc(f"{clip_num} / {total_clips}")
        bpm_str   = esc(f"  {bpm:.1f} BPM") if bpm else ""

        score_bw   = 350
        score_fill = max(2, round(score_bw * score_val)) if score_val > 0 else 0

        slider_w   = 200
        slider_x   = px + pw - slider_w - 14
        slider_pos = max(4, min(slider_w - 4, round(slider_w * start_t / max(duration, 1))))

        # Static progress fill width at the start of this segment
        prog_fill_w = round(pw * start_t / max(duration, 0.001))

        # Y positions inside panel
        y_scrub    = py + tb_h + 8
        y_clip     = py + tb_h + 30
        y_tp       = py + tb_h + 102
        y_row1     = py + tb_h + 118
        y_row2     = py + tb_h + 170
        y_score_bar = py + tb_h + 214

        xl = px + 16        # left column x
        xr = px + 202       # right column x

        f_chain = ",".join([
            # Panel bg + title bar
            f"drawbox=x={px}:y={py}:w={pw}:h={ph}:color=0x0d0604@0.88:t=fill",
            f"drawbox=x={px}:y={py}:w={pw}:h={tb_h}:color=0x1f0d09@0.95:t=fill",
            f"drawbox=x={px}:y={py+tb_h-1}:w={pw}:h=1:color=0xffffff@0.10:t=fill",
            f"drawbox=x={px}:y={py}:w={pw}:h={ph}:color=0xffffff@0.16:t=2",
            # Title bar text
            f"drawtext=text='TRANSITION LOCATOR{bpm_str}'{ff}:fontsize=13:fontcolor=0xaaaaaa@0.85:x={xl}:y={py+11}",
            f"drawtext=text='{ctr_str}'{ff}:fontsize=12:fontcolor=0x555555:x={px+pw-58}:y={py+13}",
            # Mini scrubber (top-right body area)
            f"drawtext=text='{in_str}'{ff}:fontsize=11:fontcolor=0xcccccc@0.8:x={slider_x-46}:y={y_scrub}",
            f"drawbox=x={slider_x}:y={y_scrub+2}:w={slider_w}:h=3:color=0x444444@0.7:t=fill",
            f"drawbox=x={slider_x}:y={y_scrub+2}:w={slider_pos}:h=3:color={CYAN}@0.5:t=fill",
            f"drawbox=x={slider_x+slider_pos-2}:y={y_scrub-3}:w=5:h=11:color={CYAN}@0.9:t=fill",
            # CLIP N (large)
            f"drawtext=text='{clip_str}'{ff}:fontsize=54:fontcolor=white:x={xl}:y={y_clip}",
            # "TIMING PARAMETERS:"
            f"drawtext=text='TIMING PARAMETERS\\:'{ff}:fontsize=12:fontcolor=0x777777:x={xl}:y={y_tp}",
            # Row 1 — [IN box] | TYPE
            f"drawbox=x={xl}:y={y_row1}:w=174:h=46:color={CYAN}@0.07:t=fill",
            f"drawbox=x={xl}:y={y_row1}:w=174:h=46:color={CYAN}@0.50:t=2",
            f"drawtext=text='IN\\:'{ff}:fontsize=12:fontcolor={CYAN}@0.85:x={xl+8}:y={y_row1+5}",
            f"drawtext=text='{in_str}'{ff}:fontsize=22:fontcolor={CYAN}:x={xl+42}:y={y_row1+3}",
            f"drawtext=text='TYPE'{ff}:fontsize=12:fontcolor={GREY}:x={xr}:y={y_row1}",
            f"drawtext=text='{type_str}'{ff}:fontsize=16:fontcolor=white:x={xr}:y={y_row1+18}",
            # Row 2 — [OUT box] | SCORE
            f"drawbox=x={xl}:y={y_row2}:w=174:h=58:color=0x333333@0.35:t=fill",
            f"drawbox=x={xl}:y={y_row2}:w=174:h=58:color=0x666666@0.40:t=2",
            f"drawtext=text='OUT\\:'{ff}:fontsize=12:fontcolor=0x999999:x={xl+8}:y={y_row2+5}",
            f"drawtext=text='{out_str}'{ff}:fontsize=22:fontcolor=white:x={xl+42}:y={y_row2+3}",
            f"drawtext=text='SCORE'{ff}:fontsize=12:fontcolor={GREY}:x={xr}:y={y_row2}",
            f"drawtext=text='{score_str}'{ff}:fontsize=20:fontcolor={sc}:x={xr}:y={y_row2+17}",
            # Duration label (bottom-right of row 2 box area)
            f"drawtext=text='DUR  {dur_str}'{ff}:fontsize=12:fontcolor=0x888888:x={xl+8}:y={y_row2+40}",
            # Score bar
            f"drawbox=x={xr}:y={y_score_bar}:w={score_bw}:h=5:color=0x333333@0.7:t=fill",
        ])

        if score_fill > 0:
            f_chain += f",drawbox=x={xr}:y={y_score_bar}:w={score_fill}:h=5:color={sc}@0.9:t=fill"

        # Static progress bar above panel (position at start of segment)
        f_chain += (
            f",drawbox=x={px}:y={py-9}:w={pw}:h=5:color=0x1a1a1a@0.8:t=fill"
            f",drawbox=x={px}:y={py-9}:w={prog_fill_w}:h=5:color={CYAN}@0.9:t=fill"
        )

        filter_parts.append(f"color=c={seg['color']}:s={width}x{height}:r=1000:d={seg_dur:.6f}[c{i}]")
        filter_parts.append(f"[c{i}]{f_chain}[s{i}]")
        labels.append(f"[s{i}]")

    filter_parts.append(
        f"{''.join(labels)}concat=n={len(segments)}:v=1:a=0[vconcat];"
        f"[vconcat]fps=fps={fps}[vout]"
    )
    filter_complex = ";".join(filter_parts)

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        str(output_path),
    ]

    print(f"Generating video ({len(segments)} segments)...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed (exit {e.returncode})", file=sys.stderr)
        return False


def plot_waveform(y, sr, transitions, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(y))
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(times, y, color="#444", linewidth=0.3, alpha=0.8)

        colors = {
            "hihat": "#00bfff",
            "snare": "#ff6b35",
            "downbeat": "#ff0055",
            "beat": "#aaa",
        }

        for t in transitions:
            base_type = t["type"].split("+")[0]
            color = colors.get(base_type, "#fff")
            ax.axvline(t["time"], color=color, alpha=0.6 + 0.4 * t["score"],
                       linewidth=0.8 + t["score"])

        # Legend
        from matplotlib.lines import Line2D
        legend_items = [Line2D([0], [0], color=c, label=k) for k, c in colors.items()]
        ax.legend(handles=legend_items, loc="upper right", fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title("Transition Points")
        ax.set_xlim(0, times[-1])
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Waveform plot saved: {output_path}")
    except ImportError:
        print("matplotlib not installed — skipping waveform plot")


def main():
    parser = argparse.ArgumentParser(
        description="Find video transition points in an audio file (beats, hi-hats, claps)."
    )
    parser.add_argument("audio", help="Audio file (MP3, WAV, FLAC, AAC, etc.)")
    parser.add_argument(
        "--format", choices=["text", "csv", "json", "edl", "timing", "all"],
        default="timing", help="Output format (default: timing)"
    )
    parser.add_argument(
        "--out", default=None, help="Output file path (default: <audio>_transitions.<ext>)"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Video frame rate for timecode output (e.g. 24, 25, 29.97)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum score threshold 0–1 to include a point (default: 0)"
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Keep only top N transition points by score"
    )
    parser.add_argument(
        "--every", type=int, default=1,
        help="Only keep points on every Nth beat (e.g. 2 = every half-bar)"
    )
    parser.add_argument(
        "--min-gap", type=float, default=0.1,
        help="Minimum gap in seconds between transition points (default: 0.1)"
    )
    parser.add_argument(
        "--hihat-sensitivity", type=float, default=0.07,
        help="Hi-hat onset detection delta — lower = more sensitive (default: 0.07)"
    )
    parser.add_argument(
        "--snare-sensitivity", type=float, default=0.07,
        help="Snare/clap onset detection delta — lower = more sensitive (default: 0.07)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a waveform visualisation with transition points marked"
    )
    parser.add_argument(
        "--no-hihats", action="store_true", help="Exclude hi-hat detection"
    )
    parser.add_argument(
        "--no-snares", action="store_true", help="Exclude snare/clap detection"
    )
    parser.add_argument(
        "--beats-only", action="store_true", help="Only output beat positions"
    )
    parser.add_argument(
        "--min-clip", type=float, default=1.0,
        help="Minimum clip length in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--max-clip", type=float, default=4.0,
        help="Maximum clip length in seconds (default: 4.0)"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Generate a red/blue alternating MP4 visualisation (requires ffmpeg)"
    )
    parser.add_argument(
        "--video-size", default="1280x720",
        help="Video resolution WxH (default: 1280x720)"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load ---
    y, sr, duration = load_audio(str(audio_path))

    # --- Beat tracking ---
    print("Detecting beats...")
    beat_data = detect_beats(y, sr)
    print(f"Tempo: {beat_data['bpm']} BPM  |  {len(beat_data['beat_times'])} beats detected")

    # --- Percussive onsets ---
    hihat_times, hihat_strengths = np.array([]), np.array([])
    snare_times, snare_strengths = np.array([]), np.array([])

    if not args.beats_only:
        if not args.no_hihats:
            print("Detecting hi-hats...")
            hihat_times, hihat_strengths = detect_band_onsets(
                y, sr, HIHAT_FMIN, HIHAT_FMAX, delta=args.hihat_sensitivity
            )
            print(f"  {len(hihat_times)} hi-hat onsets found")

        if not args.no_snares:
            print("Detecting snares/claps...")
            snare_times, snare_strengths = detect_band_onsets(
                y, sr, SNARE_FMIN, SNARE_FMAX, delta=args.snare_sensitivity
            )
            print(f"  {len(snare_times)} snare/clap onsets found")

    # --- Build and score transitions ---
    print("Scoring transition points...")
    transitions = build_transition_list(
        beat_data,
        hihat_times, hihat_strengths,
        snare_times, snare_strengths,
        min_gap=args.min_gap,
    )

    # --- Filter ---
    transitions = filter_top(
        transitions,
        top_n=args.top,
        min_score=args.min_score,
        every_n_beats=args.every,
        beat_times=beat_data["beat_times"],
    )

    transitions = apply_clip_bounds(transitions, min_clip=args.min_clip, max_clip=args.max_clip)

    transitions = snap_to_beats(transitions, beat_data["beat_times"])

    print(f"\nFound {len(transitions)} transition points")
    print(f"{'Timecode':<14} {'Time (s)':<12} {'Type':<22} Score")
    print("-" * 58)
    for t in transitions:
        tc = format_timecode(t["time"], args.fps)
        print(f"{tc:<14} {t['time']:<12.4f} {t['type']:<22} {t['score']:.3f}")

    # --- Export ---
    stem = audio_path.stem
    formats = (
        ["text", "csv", "json", "edl", "timing"] if args.format == "all" else [args.format]
    )
    ext_map = {"text": "txt", "csv": "csv", "json": "json", "edl": "tsv", "timing": "json"}
    # timing plan uses a distinct filename to avoid overwriting the transitions json
    name_map = {fmt: f"{stem}_transitions" for fmt in ext_map}
    name_map["timing"] = f"{stem}_timing_plan"

    for fmt in formats:
        if args.out and len(formats) == 1:
            out_path = Path(args.out)
        else:
            out_path = audio_path.parent / f"{name_map[fmt]}.{ext_map[fmt]}"

        if fmt == "text":
            export_text(transitions, args.fps, out_path)
        elif fmt == "csv":
            export_csv(transitions, args.fps, out_path)
        elif fmt == "json":
            export_json(transitions, args.fps, beat_data["bpm"], out_path)
        elif fmt == "edl":
            export_edl_markers(transitions, args.fps, out_path)
        elif fmt == "timing":
            export_timing_plan(transitions, duration, out_path)

        print(f"Saved: {out_path}")

    # --- Optional plot ---
    if args.plot:
        plot_path = audio_path.parent / f"{stem}_transitions.png"
        plot_waveform(y, sr, transitions, str(plot_path))

    # --- Optional video ---
    if args.video:
        w, h = map(int, args.video_size.split("x"))
        video_path = audio_path.parent / f"{stem}_transitions.mp4"
        video_fps = int(args.fps) if args.fps else 25
        export_video(transitions, audio_path, duration, video_path, w, h, fps=video_fps, bpm=beat_data["bpm"])


if __name__ == "__main__":
    main()
