"""Genera la narración (Piper TTS, offline), una música ambiente suave,
la línea de tiempo (timeline.json) y los subtítulos (.srt).

Uso: python3 make_audio.py --voice ruta/al/modelo.onnx --out build/
"""
import argparse
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(__file__))
from script import SCENES  # noqa: E402

SR = 48000
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SCENE_PAD_IN = 0.9     # silencio al empezar cada escena
SCENE_PAD_OUT = 1.0    # silencio al terminar cada escena
LINE_GAP = 0.45        # silencio entre frases
INTRO_EXTRA = 1.6      # título inicial antes de hablar
OUTRO_EXTRA = 4.0      # título final


def tts(text, model, path, length_scale):
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", model, "-f", path + ".raw.wav",
         "--length-scale", str(length_scale), "--noise-scale", "0.5",
         "--noise-w-scale", "0.6", "--sentence-silence", "0.25"],
        input=text.encode("utf-8"), check=True, capture_output=True)
    # Remuestrear a 48 kHz, recortar silencios de borde y ecualizar un poco la voz.
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", path + ".raw.wav",
         "-af", "silenceremove=start_periods=1:start_threshold=-45dB,"
                "areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                "highpass=f=70,equalizer=f=3000:t=q:w=1:g=2,"
                "aresample=48000",
         "-ac", "1", "-ar", str(SR), path], check=True)
    os.remove(path + ".raw.wav")
    write_wav(path, compress_silences(read_wav(path)))


def compress_silences(a, max_gap=0.5, keep=0.32, thr=0.012):
    """Acorta pausas internas demasiado largas que a veces genera el modelo."""
    fr = int(0.02 * SR)
    n = len(a) // fr
    rms = np.sqrt((a[: n * fr].reshape(n, fr) ** 2).mean(1))
    quiet = rms < thr
    out, i = [], 0
    while i < n:
        j = i
        while j < n and quiet[j] == quiet[i]:
            j += 1
        seg = a[i * fr: j * fr]
        if quiet[i] and (j - i) * fr > max_gap * SR:
            k = int(keep * SR) // 2
            seg = np.concatenate([seg[:k], seg[-k:]])
        out.append(seg)
        i = j
    out.append(a[n * fr:])
    return np.concatenate(out)


def read_wav(path):
    with wave.open(path) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def write_wav(path, data):
    data = np.clip(data, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1 if data.ndim == 1 else data.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((data * 32767).astype(np.int16).tobytes())


def split_subtitle(text, max_chars=84):
    """Parte una frase larga en bloques de subtítulo legibles (<= 2 líneas)."""
    if len(text) <= max_chars:
        return [text]
    # Primero por puntuación fuerte, después por comas, después por palabras.
    for sep in (r"(?<=[.:;?!])\s+", r"(?<=,)\s+"):
        parts = re.split(sep, text)
        if len(parts) > 1:
            chunks, cur = [], ""
            for p in parts:
                if cur and len(cur) + 1 + len(p) > max_chars:
                    chunks.append(cur)
                    cur = p
                else:
                    cur = (cur + " " + p).strip()
            chunks.append(cur)
            out = []
            for c in chunks:
                out.extend(split_subtitle(c, max_chars) if len(c) > max_chars else [c])
            if all(len(c) <= max_chars for c in out):
                return out
    words = text.split()
    n = max(2, -(-len(text) // max_chars))
    target = len(text) / n
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > target * 1.15:
            chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    chunks.append(cur)
    return chunks


def ambient_music(duration):
    """Colchón ambiente: acordes suaves con envolventes lentas (sintetizado)."""
    t = np.arange(int(duration * SR)) / SR
    out = np.zeros((len(t), 2), np.float32)
    # Progresión lenta: Re menor add9 - Si bemol maj7 - Fa - Do (frecuencias en Hz)
    chords = [
        [146.83, 220.00, 293.66, 329.63, 440.00],
        [116.54, 174.61, 233.08, 293.66, 440.00],
        [174.61, 220.00, 261.63, 349.23, 523.25],
        [130.81, 196.00, 261.63, 329.63, 392.00],
    ]
    chord_len = 9.0
    rng = np.random.default_rng(3)
    n_ch = int(np.ceil(duration / chord_len)) + 1
    for k in range(n_ch):
        start = k * chord_len - 2.0
        notes = chords[k % len(chords)]
        seg_len = chord_len + 4.0
        i0 = max(0, int(start * SR))
        i1 = min(len(t), int((start + seg_len) * SR))
        if i1 <= i0:
            continue
        tt = t[i0:i1] - start
        env = np.sin(np.pi * np.clip(tt / seg_len, 0, 1)) ** 2
        for j, f in enumerate(notes):
            det = 1 + rng.uniform(-0.002, 0.002)
            ph = rng.uniform(0, 2 * np.pi)
            w = np.sin(2 * np.pi * f * det * tt + ph) + 0.25 * np.sin(4 * np.pi * f * det * tt + ph)
            trem = 0.8 + 0.2 * np.sin(2 * np.pi * (0.1 + 0.03 * j) * tt)
            pan = 0.5 + 0.35 * np.sin(j * 1.7)
            sig = (w * env * trem * 0.06).astype(np.float32)
            out[i0:i1, 0] += sig * (1 - pan)
            out[i0:i1, 1] += sig * pan
    # Destellos tipo "celesta" muy suaves.
    for k in range(int(duration / 2.3)):
        st = k * 2.3 + rng.uniform(0, 1.2)
        f = rng.choice([587.33, 659.25, 880.0, 1046.5, 1174.66])
        i0 = int(st * SR)
        n = int(2.5 * SR)
        if i0 + n >= len(t):
            break
        tt = np.arange(n) / SR
        sig = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 2.2) * 0.02
        pan = rng.uniform(0.2, 0.8)
        out[i0:i0 + n, 0] += sig * (1 - pan)
        out[i0:i0 + n, 1] += sig * pan
    # Eco simple para dar espacio.
    d = int(0.37 * SR)
    wet = np.zeros_like(out)
    wet[d:] += out[:-d] * 0.35
    wet[2 * d:] += out[:-2 * d] * 0.15
    out = out + wet
    fade = int(3 * SR)
    out[:fade] *= np.linspace(0, 1, fade)[:, None]
    out[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return out / (np.abs(out).max() + 1e-9) * 0.9


def srt_time(s):
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", default="build")
    ap.add_argument("--length-scale", type=float, default=1.08)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "lines"), exist_ok=True)

    timeline = {"scenes": [], "subs": []}
    cursor = 0.0
    narration_parts = []
    for si, sc in enumerate(SCENES):
        scene = {"id": sc["id"], "start": cursor, "lines": []}
        t = cursor + SCENE_PAD_IN + (INTRO_EXTRA if si == 0 else 0)
        for li, line in enumerate(sc["lines"]):
            path = os.path.join(args.out, "lines", f"{si:02d}_{li:02d}.wav")
            tts(line.get("say", line["sub"]), args.voice, path, args.length_scale)
            audio = read_wav(path)
            dur = len(audio) / SR
            narration_parts.append((t, audio))
            scene["lines"].append({"start": t, "end": t + dur})
            # Subtítulos: repartir la duración según la cantidad de caracteres.
            chunks = split_subtitle(line["sub"])
            total = sum(len(c) for c in chunks)
            ct = t
            for c in chunks:
                cd = dur * len(c) / total
                timeline["subs"].append({"start": ct, "end": ct + cd, "text": c})
                ct += cd
            t += dur + LINE_GAP
            print(f"[{sc['id']}] {dur:5.2f}s  {line['sub'][:60]}", flush=True)
        t = t - LINE_GAP + SCENE_PAD_OUT + (OUTRO_EXTRA if si == len(SCENES) - 1 else 0)
        scene["end"] = t
        timeline["scenes"].append(scene)
        cursor = t
    timeline["duration"] = cursor

    total_n = int(np.ceil(cursor * SR))
    voice = np.zeros(total_n, np.float32)
    for st, a in narration_parts:
        i0 = int(st * SR)
        voice[i0:i0 + len(a)] += a[: total_n - i0]
    voice = voice / (np.abs(voice).max() + 1e-9) * 0.89

    music = ambient_music(total_n / SR)[:total_n]
    # Ducking: la música baja cuando hay voz.
    env = np.convolve(np.abs(voice), np.ones(int(0.3 * SR)) / (0.3 * SR), mode="same")
    duck = 1.0 - 0.55 * np.clip(env / 0.05, 0, 1)
    duck = np.convolve(duck, np.ones(int(0.2 * SR)) / (0.2 * SR), mode="same")
    mix = music * 0.16 * duck[:, None] + voice[:, None]
    mix = mix / max(1.0, np.abs(mix).max() / 0.97)

    write_wav(os.path.join(args.out, "narracion.wav"), voice)
    write_wav(os.path.join(args.out, "mix.wav"), mix)
    with open(os.path.join(args.out, "timeline.json"), "w") as f:
        json.dump(timeline, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "subtitulos_es.srt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(timeline["subs"], 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n")
    print(f"Duración total: {cursor:.1f}s")


if __name__ == "__main__":
    main()
