#!/usr/bin/env bash
# Genera el video completo: voz (Piper TTS offline) + animación + subtítulos.
set -euo pipefail
cd "$(dirname "$0")"
pip install -q -r requirements.txt
VOICE_DIR=build/voice
VOICE=$VOICE_DIR/es_AR-daniela-high.onnx
if [ ! -f "$VOICE" ]; then
  # Voz argentina de Piper (requiere acceso a huggingface.co y *.hf.co).
  mkdir -p "$VOICE_DIR"
  BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high
  curl -fsSL -o "$VOICE.json" "$BASE/es_AR-daniela-high.onnx.json"
  curl -fsSL -o "$VOICE" "$BASE/es_AR-daniela-high.onnx"
fi
python3 tau_video/make_audio.py --voice "$VOICE" --out build
mkdir -p output
python3 tau_video/render.py --build build --out output/proteina_tau.mp4
cp build/subtitulos_es.srt output/proteina_tau.srt
echo "Video: output/proteina_tau.mp4"
