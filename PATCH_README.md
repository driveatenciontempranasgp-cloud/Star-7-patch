# Star7 + parche de audio FP32 (audiofp32-1)

Copia de [star7code/minimax-h3-fp16-exact-star7](https://github.com/star7code/minimax-h3-fp16-exact-star7) v2.0.15 (commit 27c44a7, licencia MIT) con un parche para el audio raro (pausas y ruiditos) al usar Star7 FP16 junto con Spectrum en T4.

Qué cambia en `nodes.py`:
- Los wrappers del modelo (Spectrum y su forecast) reciben el contexto y los latentes en FP32, así el forecast ya no se recorta en el límite de FP16 (±65504).
- El núcleo de H3 sigue calculando en FP16/INT8 como el Star7 normal.
- La capa final (cabezal de audio y video) corre en FP32.
- Se apaga con la variable `STAR7_FP32_CARRIER=0` y vuelve al comportamiento original.

El diff original está en `patch/star7_audio_fp32.diff`.

## Instalación (Kaggle / Colab)
```
git clone --depth 1 https://github.com/driveatenciontempranasgp-cloud/Star-7-patch.git ComfyUI/custom_nodes/minimax-h3-fp16-exact-star7
```
No lo instales al lado del Star7 original: usa la misma carpeta y reemplázalo.

En el log debe salir `fp32-carrier=on (audiofp32-1)` al cargar y `FP32 carrier active (audiofp32-1)` en el primer paso.
