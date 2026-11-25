"""Step 2 of the cartoon pipeline: cast consistent cartoon voices (via Qwen3-TTS
VoiceDesign -> reference clips -> Qwen3-TTS Base ICL cloning, plus Kokoro fallback)
and apply per-character ffmpeg DSP filters, loudness normalization, and audio verification.

Usage:
    cartoon/.venv/bin/python cartoon/render_voices.py books/01-the-very-big-world-problem/cartoon/script.json
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_log import RenderLog  # noqa: E402

VOICE_DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
BASE_CLONE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"

# Role casting profiles:
# - instruct: used by Qwen3-TTS VoiceDesign to mint the character's canonical reference voice clip
# - ref_text: line spoken in the reference clip (gives rich prosody for ICL cloning)
# - seed: reproducible seed for the reference voice
# - ffmpeg_fx: subtle character DSP that reinforces their mixed-media material while keeping speech crisp
ROLE_PROFILES: dict[str, dict] = {
    "narrator": {
        "engine": "kokoro",  # Kokoro bm_george / am_adam is rock-solid for 1950s deadpan narration; or voice_design
        "kokoro_voice": "bm_george",
        "kokoro_lang": "b",
        "speed": 0.96,
        "instruct": (
            "Warm, polished 1950s educational-filmstrip narrator, male baritone, crisp mid-Atlantic diction, "
            "earnestly cheerful deadpan delivery as if explaining toaster safety."
        ),
        "ref_text": "Goodness gracious, look how carefully our young citizens cooperate on a bright Tuesday morning!",
        "seed": 1955,
        # Subtle 1950s filmstrip optical-sound warmth (gentle presence boost, warm tape rolloff)
        "ffmpeg_fx": "highpass=f=90,lowpass=f=7500,equalizer=f=2400:width_type=q:width=1.2:g=2.5,acompressor=threshold=-18dB:ratio=2.5",
    },
    "chester": {
        "engine": "qwen_clone",
        "instruct": (
            "Awkward, slightly nasal teenage cartoon boy, flat deadpan monotone, speaks a touch hesitantly "
            "as if pretending he heard the question."
        ),
        "ref_text": "Haha, yeah, totally. Wait, did someone say something about a giant pancake?",
        "seed": 1601,
        # 16-bit retro console sheen: gentle sample-rate crunch + slight bit reduction while staying 100% clear
        "ffmpeg_fx": "aresample=22050,acrusher=bits=11:mode=log:mix=0.28,aresample=24000,highpass=f=120,acompressor=threshold=-18dB:ratio=2.5",
    },
    "marisol": {
        "engine": "qwen_clone",
        "instruct": (
            "Bossy, hyper-organized teenage cartoon girl, crisp rapid-fire enunciation, clipped high-strung "
            "authority like a class president running on cold coffee."
        ),
        "ref_text": "Everything must be alphabetized immediately, and I have brought three separate rulers to verify!",
        "seed": 3003,
        # Stop-motion clay crispness: slight pitch lift (+5%) + crisp high-mid edge
        "ffmpeg_fx": "asetrate=24000*1.05,atempo=1/1.05,aresample=24000,highpass=f=140,equalizer=f=3200:width_type=q:width=1.4:g=3.0",
    },
    "winston": {
        "engine": "qwen_clone",
        "instruct": (
            "Timid, gentle, mildly alarmed cartoon boy, soft breathy higher-pitched puppet voice, nervous "
            "tremor as if startled by a nearby wind chime."
        ),
        "ref_text": "Oh goodness, mercy me, is anyone else hearing that terribly loud breathing sound?",
        "seed": 6006,
        # Fuzzy felt puppet: soft warm lowpass + slight pitch lift (+7%)
        "ffmpeg_fx": "asetrate=24000*1.07,atempo=1/1.07,aresample=24000,lowpass=f=5200,equalizer=f=350:width_type=q:width=1.0:g=2.0",
    },
    "teacher": {
        "engine": "qwen_clone",
        "instruct": (
            "Prim, calm 1950s elementary schoolteacher, gentle polite mid-century female voice, mildly amused."
        ),
        "ref_text": "What a curious decoration for our Tuesday geography lesson, children.",
        "seed": 4004,
        "ffmpeg_fx": "highpass=f=110,lowpass=f=7200",
    },
    "principal": {
        "engine": "qwen_clone",
        "instruct": (
            "Booming, overly pompous 1950s high-school principal, deep theatrical radio baritone full of misplaced pride."
        ),
        "ref_text": "This magnificent sphere shall be the centerpiece of our brand new Global Initiative!",
        "seed": 5005,
        # Subtle schoolyard megaphone/PA warmth
        "ffmpeg_fx": "highpass=f=240,lowpass=f=4200,equalizer=f=1600:width_type=q:width=1.1:g=3.5,acompressor=threshold=-16dB:ratio=3.0",
    },
}


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def read_wav_stats(path: Path) -> tuple[float, float, float]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    dur = n / max(sr, 1)
    rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return dur, rms, peak


def apply_fx_and_normalize(raw_wav: Path, out_wav: Path, fx: str) -> None:
    chain = f"{fx},loudnorm=I=-16:TP=-1.5:LRA=9,aresample=24000" if fx else "loudnorm=I=-16:TP=-1.5:LRA=9,aresample=24000"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_wav),
            "-af",
            chain,
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fx failed for {raw_wav}: {proc.stderr}")


def collect_audio(results) -> tuple[np.ndarray, int]:
    chunks = []
    sr = 24000
    for r in results:
        sr = int(r.sample_rate)
        chunks.append(np.asarray(r.audio, dtype=np.float32).squeeze())
    if not chunks:
        raise RuntimeError("TTS returned no audio chunks")
    return np.concatenate(chunks), sr


def is_sane_audio(wav_path: Path, text: str) -> tuple[bool, str]:
    dur, rms, peak = read_wav_stats(wav_path)
    words = max(1, len(text.split()))
    min_dur = max(0.35, words * 0.14)
    max_dur = max(6.0, words * 1.15)
    if rms < 0.01 or peak < 0.05:
        return False, f"too quiet (rms={rms:.4f}, peak={peak:.4f})"
    if dur < min_dur or dur > max_dur:
        return False, f"duration {dur:.2f}s out of expected [{min_dur:.2f}, {max_dur:.2f}] for {words} words"
    return True, f"{dur:.2f}s rms={rms:.3f}"


def ensure_voice_refs(refs_dir: Path, roles_needed: set[str], overwrite: bool = False) -> dict[str, Path]:
    from mlx_audio.tts.utils import load_model

    ref_paths: dict[str, Path] = {}
    missing = []
    for role in sorted(roles_needed):
        prof = ROLE_PROFILES.get(role, ROLE_PROFILES["narrator"])
        if prof["engine"] != "qwen_clone":
            continue
        p = refs_dir / f"{role}_ref.wav"
        ref_paths[role] = p
        if overwrite or not p.exists():
            missing.append(role)

    if not missing:
        return ref_paths

    print(f"Minting canonical voice references via {VOICE_DESIGN_MODEL} for: {missing}")
    vd_model = load_model(VOICE_DESIGN_MODEL)
    try:
        for role in missing:
            prof = ROLE_PROFILES[role]
            out_p = ref_paths[role]
            for attempt in range(4):
                mx.random.seed(prof["seed"] + attempt * 17)
                audio, sr = collect_audio(
                    vd_model.generate_voice_design(
                        text=prof["ref_text"],
                        instruct=prof["instruct"],
                        language="english",
                        temperature=0.75,
                    )
                )
                write_wav(out_p, audio, sr)
                ok, info = is_sane_audio(out_p, prof["ref_text"])
                print(f"  ref [{role}] try {attempt+1}: {info} -> {'OK' if ok else 'RETRY'}")
                if ok:
                    break
    finally:
        del vd_model
        gc.collect()
        mx.clear_cache()

    return ref_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cartoon character voices for script.json.")
    parser.add_argument("script_json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-refs", action="store_true")
    parser.add_argument("--narrator-engine", choices=["kokoro", "qwen_clone"], default="kokoro")
    args = parser.parse_args()

    script_path = args.script_json.resolve()
    script = json.loads(script_path.read_text(encoding="utf-8"))
    cartoon_dir = script_path.parent
    voices_dir = cartoon_dir / "voices"
    raw_dir = voices_dir / "raw"
    fx_dir = voices_dir / "fx"
    refs_dir = cartoon_dir.parent.parent.parent / "cartoon" / "voice_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    ROLE_PROFILES["narrator"]["engine"] = args.narrator_engine

    # Collect tasks
    tasks: list[tuple[int, int, str, str]] = []
    roles_needed: set[str] = set()
    for shot in script["shots"]:
        pnum = int(shot["page_number"])
        for idx, seg in enumerate(shot["segments"]):
            spk = seg["speaker"] if seg["speaker"] in ROLE_PROFILES else "narrator"
            roles_needed.add(spk)
            tasks.append((pnum, idx, spk, seg["text"]))

    ref_paths = ensure_voice_refs(refs_dir, roles_needed, overwrite=args.overwrite_refs)

    log = RenderLog(
        cartoon_dir,
        "voices",
        {
            "voice_design_model": VOICE_DESIGN_MODEL,
            "clone_model": BASE_CLONE_MODEL,
            "kokoro_model": KOKORO_MODEL,
            "narrator_engine": args.narrator_engine,
            "roles": {
                r: {k: v for k, v in ROLE_PROFILES[r].items() if k in ("engine", "kokoro_voice", "speed", "seed", "ffmpeg_fx")}
                for r in sorted(roles_needed)
            },
            "voice_refs": {r: f"cartoon/voice_refs/{Path(p).name}" for r, p in ref_paths.items()},
            "overwrite": args.overwrite,
        },
    )
    warnings = 0

    from mlx_audio.tts.utils import load_model

    # Split tasks by engine so we only load one TTS model at a time
    kokoro_tasks = [t for t in tasks if ROLE_PROFILES[t[2]]["engine"] == "kokoro"]
    clone_tasks = [t for t in tasks if ROLE_PROFILES[t[2]]["engine"] == "qwen_clone"]

    if kokoro_tasks:
        todo = [t for t in kokoro_tasks if args.overwrite or not (fx_dir / f"p{t[0]:02d}_s{t[1]:02d}_{t[2]}.wav").exists()]
        if todo:
            print(f"Rendering {len(todo)} narrator lines via {KOKORO_MODEL}...")
            k_model = load_model(KOKORO_MODEL)
            try:
                for pnum, idx, spk, text in todo:
                    prof = ROLE_PROFILES[spk]
                    raw_p = raw_dir / f"p{pnum:02d}_s{idx:02d}_{spk}.wav"
                    fx_p = fx_dir / f"p{pnum:02d}_s{idx:02d}_{spk}.wav"
                    audio, sr = collect_audio(
                        k_model.generate(
                            text=text,
                            voice=prof["kokoro_voice"],
                            speed=prof["speed"],
                            lang_code=prof["kokoro_lang"],
                        )
                    )
                    write_wav(raw_p, audio, sr)
                    apply_fx_and_normalize(raw_p, fx_p, prof["ffmpeg_fx"])
                    ok, info = is_sane_audio(fx_p, text)
                    print(f"  p{pnum:02d}_s{idx:02d} [{spk}]: {info} ({'OK' if ok else 'WARN'})")
                    warnings += 0 if ok else 1
                    log.add(page_number=pnum, segment=idx, speaker=spk, engine="kokoro", attempts=1, ok=ok, check=info,
                            audio=fx_p.relative_to(cartoon_dir).as_posix())
            finally:
                del k_model
                gc.collect()
                mx.clear_cache()

    if clone_tasks:
        todo = [t for t in clone_tasks if args.overwrite or not (fx_dir / f"p{t[0]:02d}_s{t[1]:02d}_{t[2]}.wav").exists()]
        if todo:
            print(f"Rendering {len(todo)} character lines via {BASE_CLONE_MODEL}...")
            c_model = load_model(BASE_CLONE_MODEL)
            try:
                for pnum, idx, spk, text in todo:
                    prof = ROLE_PROFILES[spk]
                    raw_p = raw_dir / f"p{pnum:02d}_s{idx:02d}_{spk}.wav"
                    fx_p = fx_dir / f"p{pnum:02d}_s{idx:02d}_{spk}.wav"
                    ok = False
                    info = ""
                    attempt = 0
                    for attempt in range(4):
                        mx.random.seed(prof["seed"] + pnum * 100 + idx * 10 + attempt)
                        audio, sr = collect_audio(
                            c_model.generate(
                                text=text,
                                ref_audio=str(ref_paths[spk]),
                                ref_text=prof["ref_text"],
                                lang_code="english",
                                temperature=0.72 + attempt * 0.05,
                                repetition_penalty=1.35,
                            )
                        )
                        write_wav(raw_p, audio, sr)
                        apply_fx_and_normalize(raw_p, fx_p, prof["ffmpeg_fx"])
                        ok, info = is_sane_audio(fx_p, text)
                        if ok:
                            break
                        print(f"  p{pnum:02d}_s{idx:02d} [{spk}] retry {attempt+1}: {info}")
                    print(f"  p{pnum:02d}_s{idx:02d} [{spk}]: {info} ({'OK' if ok else 'WARN'})")
                    warnings += 0 if ok else 1
                    log.add(page_number=pnum, segment=idx, speaker=spk, engine="qwen_clone", attempts=attempt + 1,
                            seed=prof["seed"] + pnum * 100 + idx * 10 + attempt, temperature=round(0.72 + attempt * 0.05, 2),
                            ok=ok, check=info, audio=fx_p.relative_to(cartoon_dir).as_posix())
            finally:
                del c_model
                gc.collect()
                mx.clear_cache()

    # Record rendered audio paths + durations back into script_voiced.json
    for shot in script["shots"]:
        pnum = int(shot["page_number"])
        for idx, seg in enumerate(shot["segments"]):
            spk = seg["speaker"] if seg["speaker"] in ROLE_PROFILES else "narrator"
            fx_p = fx_dir / f"p{pnum:02d}_s{idx:02d}_{spk}.wav"
            dur, _, _ = read_wav_stats(fx_p)
            seg["audio_path"] = str(fx_p.relative_to(cartoon_dir))
            seg["duration_s"] = round(dur, 3)

    voiced_path = cartoon_dir / "script_voiced.json"
    voiced_path.write_text(json.dumps(script, indent=2) + "\n", encoding="utf-8")
    log.finish(lines_rendered=len(log.run["items"]), warnings=warnings)
    print(f"Saved voiced script -> {voiced_path}")


if __name__ == "__main__":
    main()
