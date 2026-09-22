#!/usr/bin/env bash
# Genera el video completo: voz (Piper TTS offline) + animación + subtítulos.
set -euo pipefail
cd "$(dirname "$0")"
pip install -q -r requirements.txt
VOICE_DIR=build/voice
VOICE=$VOICE_DIR/es-mls_10246-low.onnx
if [ ! -f "$VOICE" ]; then
  mkdir -p "$VOICE_DIR"
  curl -sSL https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-mls_10246-low.tar.gz \
    | tar xz -C "$VOICE_DIR"
fi
python3 tau_video/make_audio.py --voice "$VOICE" --out build
mkdir -p output
python3 tau_video/render.py --build build --out output/proteina_tau.mp4
cp build/subtitulos_es.srt output/proteina_tau.srt
echo "Video: output/proteina_tau.mp4"
