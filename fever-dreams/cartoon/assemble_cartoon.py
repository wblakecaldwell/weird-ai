"""Step 4 of the cartoon pipeline: assemble animated LTX-2.5 clips, smooth Ken Burns
extensions on original illustrations, character voices, 1950s read-along page-turn
chimes, subtitles, and end credits into a final 24fps MP4 cartoon.

Usage:
    cartoon/.venv/bin/python cartoon/assemble_cartoon.py books/03-the-very-big-world-problem/cartoon/script_voiced.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 24
W = 896
H = 768


def run_ffmpeg(args: list[str], desc: str) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({desc}):\n{proc.stderr}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stdout.strip())


def has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip())


def make_page_chime(path: Path, sr: int = 24000) -> Path:
    """Synthesize a nostalgic 1950s Little Golden Book read-along *ding!* chime."""
    if path.exists():
        return path
    dur = 0.75
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    # Warm celesta/bell partials
    sig = (
        0.55 * np.sin(2 * math.pi * 1046.5 * t) * np.exp(-t * 5.2)
        + 0.28 * np.sin(2 * math.pi * 2093.0 * t) * np.exp(-t * 8.0)
        + 0.12 * np.sin(2 * math.pi * 3136.0 * t) * np.exp(-t * 11.0)
    )
    sig *= np.minimum(1.0, t / 0.006)
    sig *= 0.35
    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def make_silence(path: Path, dur: float, sr: int = 24000) -> Path:
    n = max(1, int(round(dur * sr)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n)
    return path


def build_shot_audio(
    shot: dict,
    cartoon_dir: Path,
    chime_wav: Path,
    out_wav: Path,
    is_cover: bool,
) -> tuple[float, list[tuple[float, float, str, str]]]:
    """Concatenate segment WAVs with natural pauses and return (total_dur, subtitle_spans)."""
    tmp_dir = cartoon_dir / "build" / "audio_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pnum = int(shot["page_number"])

    parts: list[Path] = []
    subs: list[tuple[float, float, str, str]] = []
    t_cursor = 0.0

    # Lead-in pause (or page-turn chime for pages > 1)
    if pnum > 1:
        parts.append(chime_wav)
        t_cursor += probe_duration(chime_wav)
    else:
        lead = make_silence(tmp_dir / f"lead_{pnum:02d}.wav", 0.45)
        parts.append(lead)
        t_cursor += 0.45

    prev_spk = None
    for idx, seg in enumerate(shot["segments"]):
        spk = seg["speaker"]
        if idx > 0:
            gap_s = 0.25 if (prev_spk == "narrator" and spk == "narrator") else 0.38
            gap = make_silence(tmp_dir / f"gap_{pnum:02d}_{idx:02d}.wav", gap_s)
            parts.append(gap)
            t_cursor += gap_s

        seg_wav = cartoon_dir / seg["audio_path"]
        dur = probe_duration(seg_wav)
        parts.append(seg_wav)
        subs.append((t_cursor, t_cursor + dur, spk, seg["text"]))
        t_cursor += dur
        prev_spk = spk

    tail_s = 0.9 if is_cover else 0.75
    tail = make_silence(tmp_dir / f"tail_{pnum:02d}.wav", tail_s)
    parts.append(tail)
    t_cursor += tail_s

    concat_list = tmp_dir / f"concat_{pnum:02d}.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in parts) + "\n", encoding="utf-8")
    run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", str(out_wav)],
        f"concat audio p{pnum:02d}",
    )
    return probe_duration(out_wav), subs


def make_ken_burns_extension(
    img_path: Path,
    out_mp4: Path,
    dur: float,
    page_num: int,
    width: int = W,
    height: int = H,
) -> None:
    """Render a smooth 24fps Ken Burns motion pass over the crisp illustration."""
    frames = max(2, int(round(dur * FPS)))
    # Alternate gentle zoom-in / zoom-out / slight pan across pages
    if page_num % 2 == 0:
        z_expr = f"1.0+0.06*(on/{frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"1.06-0.05*(on/{frames})"
        x_expr = f"iw/2-(iw/zoom/2)+18*sin(on/{frames}*3.14)"
        y_expr = "ih/2-(ih/zoom/2)"

    vf = (
        f"scale={width * 2}:{height * 2}:flags=lanczos,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={width}x{height}:fps={FPS},"
        "format=yuv420p"
    )
    run_ffmpeg(
        ["-loop", "1", "-i", str(img_path), "-t", f"{dur:.3f}", "-vf", vf, "-c:v", "libx264", "-crf", "16", "-preset", "medium", str(out_mp4)],
        f"ken burns p{page_num:02d}",
    )


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


SPEAKER_COLORS = {
    "narrator": (248, 242, 224, 255),
    "chester": (122, 240, 176, 255),
    "marisol": (255, 166, 138, 255),
    "winston": (245, 220, 118, 255),
    "teacher": (196, 222, 255, 255),
    "principal": (255, 206, 128, 255),
}


def render_caption_png(out_png: Path, spk: str, text: str) -> Path:
    """Render a clean, translucent Little Golden Book-style bottom caption strip (W x 150 RGBA)."""
    band_h = 150
    img = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_spk = _load_font("/System/Library/Fonts/HelveticaNeue.ttc", 20)
    font_txt = _load_font("/System/Library/Fonts/Supplemental/Georgia.ttf", 22)

    prefix = "" if spk == "narrator" else f"{spk.upper()}: "
    full_text = f"{prefix}{text}"

    # Word wrap inside W - 96 px
    max_w = W - 96
    words = full_text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip()
        bbox = draw.textbbox((0, 0), cand, font=font_txt)
        if bbox[2] - bbox[0] > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)

    line_h = 28
    pad_y = 12
    box_h = len(lines) * line_h + pad_y * 2
    box_top = band_h - box_h - 14
    draw.rounded_rectangle([28, box_top, W - 28, band_h - 14], radius=12, fill=(18, 16, 14, 196))

    y = box_top + pad_y
    for i, ln in enumerate(lines):
        if i == 0 and prefix and ln.startswith(prefix):
            rest = ln[len(prefix) :]
            pw = draw.textlength(prefix, font=font_spk)
            rw = draw.textlength(rest, font=font_txt)
            x0 = (W - (pw + rw)) / 2
            draw.text((x0, y + 1), prefix, fill=SPEAKER_COLORS.get(spk, (248, 242, 224, 255)), font=font_spk)
            draw.text((x0 + pw, y), rest, fill=(248, 242, 224, 255), font=font_txt)
        else:
            draw.text((W // 2, y + line_h // 2 - 2), ln, fill=(248, 242, 224, 255), font=font_txt, anchor="mm")
        y += line_h

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def format_srt_time(sec: float) -> str:
    ms = int(round((sec % 1) * 1000))
    total_s = int(sec)
    if ms == 1000:
        total_s += 1
        ms = 0
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(srt_path: Path, global_subs: list[tuple[float, float, str, str]]) -> None:
    blocks = []
    for i, (start, end, spk, txt) in enumerate(global_subs, 1):
        if end <= start:
            continue
        label = txt if spk == "narrator" else f"[{spk.upper()}] {txt}"
        blocks.append(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{label}\n")
    srt_path.write_text("\n".join(blocks), encoding="utf-8")


def build_endcard(story: dict, out_mp4: Path, dur: float = 4.5) -> None:
    img = Image.new("RGB", (W, H), (33, 52, 58))
    draw = ImageDraw.Draw(img)
    # Vintage Little Golden Book gold border
    draw.rectangle([24, 24, W - 24, H - 24], outline=(232, 194, 102), width=4)
    draw.rectangle([34, 34, W - 34, H - 34], outline=(232, 194, 102), width=1)

    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 52)
        font_med = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", 26)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Italic.ttf", 21)
    except Exception:
        font_big = font_med = font_sm = ImageFont.load_default()

    draw.text((W // 2, 190), "THE END", fill=(246, 214, 128), font=font_big, anchor="mm")
    draw.text((W // 2, 265), story.get("title", ""), fill=(248, 242, 228), font=font_med, anchor="mm")

    moral = story.get("moral", "")
    if moral:
        # simple word wrap
        words, lines, cur = moral.split(), [], ""
        for w in words:
            if len(cur + " " + w) > 54:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        y = 355
        for ln in lines[:4]:
            draw.text((W // 2, y), ln, fill=(215, 226, 224), font=font_sm, anchor="mm")
            y += 32

    cast_names = "Starring " + " • ".join(c["name"] for c in story.get("cast", []))
    draw.text((W // 2, 560), cast_names, fill=(232, 194, 102), font=font_sm, anchor="mm")
    draw.text((W // 2, 620), "100% Local AI Cartoon (Gemma 4 + mflux + LTX-2.5 + Qwen3-TTS + Kokoro)", fill=(165, 185, 188), font=font_sm, anchor="mm")

    png = out_mp4.with_suffix(".png")
    img.save(png)
    run_ffmpeg(
        [
            "-loop", "1", "-i", str(png),
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", f"{dur:.2f}",
            "-vf", f"fade=t=in:st=0:d=0.6,fade=t=out:st={dur-0.8:.2f}:d=0.8,format=yuv420p",
            "-c:v", "libx264", "-crf", "16", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4),
        ],
        "endcard",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble animated clips and cartoon voices into final MP4.")
    parser.add_argument("voiced_script_json", type=Path)
    parser.add_argument("--no-subtitles", action="store_true")
    args = parser.parse_args()

    script_path = args.voiced_script_json.resolve()
    script = json.loads(script_path.read_text(encoding="utf-8"))
    cartoon_dir = script_path.parent
    book_dir = cartoon_dir.parent
    story = json.loads((book_dir / "story.json").read_text(encoding="utf-8"))

    build_dir = cartoon_dir / "build"
    shots_out = build_dir / "shots"
    shots_out.mkdir(parents=True, exist_ok=True)
    chime_wav = make_page_chime(build_dir / "page_chime.wav")

    shot_mp4s: list[Path] = []
    global_subs: list[tuple[float, float, str, str]] = []
    timeline_t = 0.0

    for shot in script["shots"]:
        pnum = int(shot["page_number"])
        is_cover = shot["kind"] == "cover"
        shot_wav = build_dir / f"shot_{pnum:02d}_voice.wav"
        target_dur, local_subs = build_shot_audio(shot, cartoon_dir, chime_wav, shot_wav, is_cover=is_cover)

        cond_frame = cartoon_dir / "cond_frames" / f"cond_{pnum:02d}.png"
        still_img = cond_frame if cond_frame.exists() else (book_dir / shot["image"])
        anim_clip = cartoon_dir / "clips" / f"clip_{pnum:02d}.mp4"
        shot_video = build_dir / f"shot_{pnum:02d}_video.mp4"

        if anim_clip.exists():
            clip_dur = probe_duration(anim_clip)
            if clip_dur >= target_dur - 0.1:
                # Trim/pad animated clip cleanly to target_dur
                run_ffmpeg(
                    [
                        "-i", str(anim_clip),
                        "-t", f"{target_dur:.3f}",
                        "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS},format=yuv420p",
                        "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium", str(shot_video),
                    ],
                    f"trim anim p{pnum:02d}",
                )
            else:
                # Smooth cross-dissolve from LTX-2.5 animated shot into Ken Burns motion over crisp art
                xfade_d = 0.45
                kb_dur = (target_dur - clip_dur) + xfade_d
                kb_mp4 = build_dir / f"shot_{pnum:02d}_kb.mp4"
                make_ken_burns_extension(still_img, kb_mp4, kb_dur, pnum)
                offset = max(0.1, clip_dur - xfade_d)
                run_ffmpeg(
                    [
                        "-i", str(anim_clip),
                        "-i", str(kb_mp4),
                        "-filter_complex",
                        (
                            f"[0:v]scale={W}:{H}:flags=lanczos,fps={FPS},settb=AVTB,format=yuv420p[v0];"
                            f"[1:v]scale={W}:{H}:flags=lanczos,fps={FPS},settb=AVTB,format=yuv420p[v1];"
                            f"[v0][v1]xfade=transition=fade:duration={xfade_d}:offset={offset:.3f},format=yuv420p[v]"
                        ),
                        "-map", "[v]", "-t", f"{target_dur:.3f}",
                        "-c:v", "libx264", "-crf", "16", "-preset", "medium", str(shot_video),
                    ],
                    f"xfade p{pnum:02d}",
                )
        else:
            make_ken_burns_extension(still_img, shot_video, target_dur, pnum)

        # Overlay per-segment Pillow RGBA caption strips cleanly onto shot_video
        if not args.no_subtitles and local_subs:
            cap_dir = build_dir / "captions"
            cap_inputs: list[str] = ["-i", str(shot_video)]
            filter_steps: list[str] = []
            prev_label = "[0:v]"
            for idx, (s0, s1, spk, txt) in enumerate(local_subs, 1):
                cap_png = render_caption_png(cap_dir / f"cap_{pnum:02d}_{idx:02d}.png", spk, txt)
                cap_inputs.extend(["-i", str(cap_png)])
                out_label = "[vcap]" if idx == len(local_subs) else f"[v{idx}]"
                filter_steps.append(
                    f"{prev_label}[{idx}:v]overlay=0:H-h:enable='between(t,{s0:.3f},{s1:.3f})'{out_label}"
                )
                prev_label = out_label
            captioned_video = build_dir / f"shot_{pnum:02d}_captioned.mp4"
            run_ffmpeg(
                [
                    *cap_inputs,
                    "-filter_complex",
                    ";".join(filter_steps),
                    "-map",
                    "[vcap]",
                    "-t",
                    f"{target_dur:.3f}",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "16",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                    str(captioned_video),
                ],
                f"overlay captions p{pnum:02d}",
            )
            shot_video = captioned_video

        # Mix voice + soft ambient SFX from LTX-2.5 clip (ducked at -22dB so voices stay crystal clear)
        final_shot = shots_out / f"shot_{pnum:02d}.mp4"
        if anim_clip.exists() and has_audio_stream(anim_clip):
            run_ffmpeg(
                [
                    "-i", str(shot_video),
                    "-i", str(shot_wav),
                    "-i", str(anim_clip),
                    "-filter_complex",
                    (
                        "[1:a]aresample=24000[voice];"
                        "[2:a]aresample=24000,highpass=f=120,lowpass=f=6000,volume=0.08,apad[amb];"
                        "[voice][amb]amix=inputs=2:duration=first:normalize=0[a]"
                    ),
                    "-map", "0:v", "-map", "[a]",
                    "-t", f"{target_dur:.3f}",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(final_shot),
                ],
                f"mux p{pnum:02d}",
            )
        else:
            run_ffmpeg(
                [
                    "-i", str(shot_video),
                    "-i", str(shot_wav),
                    "-map", "0:v", "-map", "1:a",
                    "-t", f"{target_dur:.3f}",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(final_shot),
                ],
                f"mux voice-only p{pnum:02d}",
            )

        actual_dur = probe_duration(final_shot)
        for s0, s1, spk, txt in local_subs:
            global_subs.append((timeline_t + s0, timeline_t + s1, spk, txt))
        timeline_t += actual_dur
        shot_mp4s.append(final_shot)
        print(f"  Built {final_shot.name} ({actual_dur:.2f}s)")

    end_mp4 = shots_out / "shot_99_end.mp4"
    build_endcard(story, end_mp4)
    shot_mp4s.append(end_mp4)

    raw_concat = build_dir / "assembled_raw.mp4"
    concat_txt = build_dir / "shots_concat.txt"
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in shot_mp4s) + "\n", encoding="utf-8")
    run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(raw_concat)],
        "concat all shots",
    )

    out_mp4 = cartoon_dir / f"{story['slug']}.mp4"
    raw_concat.replace(out_mp4)
    srt_path = cartoon_dir / f"{story['slug']}.srt"
    write_srt(srt_path, global_subs)

    total_dur = probe_duration(out_mp4)
    print(f"\nCartoon ready! {out_mp4} ({total_dur:.1f}s / {total_dur/60:.1f} min)")
    print(f"Subtitles (.srt): {srt_path}")


if __name__ == "__main__":
    main()
