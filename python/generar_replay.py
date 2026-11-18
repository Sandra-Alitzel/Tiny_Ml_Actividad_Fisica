#!/usr/bin/env python3
"""
generar_replay.py - Prepara el sketch HAR_Replay (reproducción de ventanas grabadas, SIN IMU).

Toma unas ventanas de 2 s de cada actividad del dataset, las escribe en
arduino/HAR_Replay/replay_data.h y copia model.h y normalizador.h al sketch.

Uso (desde la carpeta HAR):
    python python/generar_replay.py                      # 3 ventanas por actividad de persona2
    python python/generar_replay.py --persona persona3 --por-actividad 4
Antes hay que haber ejecutado entrenar.py o entrenar_v2.py (generan model.h y normalizador.h).
"""
import argparse, shutil
from pathlib import Path
import numpy as np
from entrenar_v2 import cargar_ventanas, CLASES

BASE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--persona', default='persona2')
ap.add_argument('--por-actividad', type=int, default=3)
a = ap.parse_args()

W, y, p = cargar_ventanas()
sel_w, sel_y = [], []
for c in range(5):                      # actividades en orden: reposo, caminar, correr, subir, bajar
    idx = np.where((y == c) & (p == a.persona))[0]
    if len(idx) < a.por_actividad:
        raise SystemExit(f'{a.persona} no tiene suficientes ventanas de {CLASES[c]}')
    elegidas = idx[np.linspace(5, len(idx) - 6, a.por_actividad).astype(int)]   # repartidas en el registro
    for i in elegidas:
        sel_w.append(W[i]); sel_y.append(c)

dest = BASE / 'arduino' / 'HAR_Replay'
dest.mkdir(parents=True, exist_ok=True)
L = ['// Generado por generar_replay.py - ventanas grabadas (100 muestras x 6 canales: ax ay az gx gy gz)',
     '#pragma once', f'// Origen: {a.persona}',
     f'static constexpr int kReplayWindows = {len(sel_w)};',
     'const char* const kReplayName[5] = {"REPOSO","CAMINAR","CORRER","SUBIR_ESCALERAS","BAJAR_ESCALERAS"};',
     'const unsigned char kReplayLabel[kReplayWindows] = {' + ', '.join(map(str, sel_y)) + '};',
     'const float kReplayData[kReplayWindows][100][6] = {']
for w in sel_w:
    L.append(' {' + ','.join('{' + ','.join(f'{v:.5g}' for v in fila) + '}' for fila in w) + '},')
L.append('};')
(dest / 'replay_data.h').write_text('\n'.join(L) + '\n')

for f in ('model.h', 'normalizador.h'):
    src = BASE / 'arduino' / 'HAR_Inferencia' / f
    if not src.exists():
        raise SystemExit(f'Falta {src}. Ejecute primero entrenar.py o entrenar_v2.py')
    shutil.copy(src, dest / f)
print(f'Listo: {len(sel_w)} ventanas en {dest / "replay_data.h"} (+ model.h y normalizador.h copiados)')
