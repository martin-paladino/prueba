# La proteína tau — video educativo

Video animado (~3,5 min, 1280×720) en español, con narración y subtítulos, que explica
cómo actúa la proteína tau en el cerebro a nivel biológico:

1. Neuronas y axón: transporte de mitocondrias y vesículas.
2. Microtúbulos (tubulina) como rieles y proteínas motoras (quinesina).
3. Tau (gen *MAPT*): se une a los microtúbulos, los estabiliza y favorece su ensamblaje.
4. Regulación por fosforilación (quinasas / fosfatasas): un equilibrio dinámico normal.
5. Hiperfosforilación: tau se desprende, los microtúbulos se desarman y el transporte falla.
6. Agregación: oligómeros → filamentos helicoidales apareados → ovillos neurofibrilares.
7. Muerte neuronal, propagación tipo prion y avance típico en el Alzheimer (estadios de Braak, simplificado).
8. Tauopatías: Alzheimer (ovillos + placas de beta-amiloide), demencia frontotemporal,
   parálisis supranuclear progresiva, encefalopatía traumática crónica.
9. Líneas de investigación: anticuerpos contra tau, inhibidores de la agregación, terapias que reducen su producción.

La estética (ilustración plana tipo dibujo a mano, fondo de papel, contornos marrones,
verde menta / naranja / rosa, zooms lentos y rayos de luz) toma como referencia el video
que se pasó de ejemplo.

## Archivos

- `output/proteina_tau.mp4`: video final (subtítulos incrustados).
- `output/proteina_tau.srt`: subtítulos en archivo aparte.
- `tau_video/script.py`: guion (texto de subtítulos y texto para la voz).
- `tau_video/make_audio.py`: narración con [Piper](https://github.com/rhasspy/piper) (offline), música ambiente sintetizada, línea de tiempo y `.srt`.
- `tau_video/render.py`: animación cuadro por cuadro con Pillow y codificación con ffmpeg.
- `assets/fonts/`: Fredoka y Nunito (licencia SIL OFL).

## Regenerar

```bash
./build.sh
```

Para cambiar el texto, editá `tau_video/script.py` y volvé a correr `./build.sh`; la animación
se sincroniza sola con la duración de cada frase.

> Contenido simplificado con fines de divulgación; no reemplaza material médico.
