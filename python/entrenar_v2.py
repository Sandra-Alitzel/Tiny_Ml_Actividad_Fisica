#!/usr/bin/env python3
"""
entrenar_v2.py -(misma red, mismo dataset, misma semilla).

Añade:
  1) Curva de aprendizaje (pérdida y exactitud por época).
  2) Comparación FP32 contra INT8: acuerdo de predicciones ventana a ventana.
  3) Mejora del modelo: características invariantes a la orientación de la placa
     (variantes 'acel_inv' y 'imu_inv') comparadas con las originales.
  4) Lista de operadores del modelo y tamaños.
  5) Figuras PDF listas para el reporte y filas de tabla en LaTeX.

Uso (desde la carpeta HAR):
    python python/entrenar_v2.py

Salidas:
    reporte/figures/*.pdf          -> copiar junto a main.tex (carpeta figures/)
    resultados/metricas_v2.json    -> todas las métricas
    resultados/snippets_reporte.tex-> filas de tabla para pegar en el reporte
    modelos_v2/*.tflite            -> modelos FP32 e INT8 de las 4 variantes
    arduino/HAR_Inferencia/model.h y normalizador.h  (modelo 'imu' original, el que usa el firmware)
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
CLASES = ['reposo', 'caminar', 'correr', 'subir_escaleras', 'bajar_escaleras']
ETIQ = ['Reposo', 'Caminar', 'Correr', 'Subir', 'Bajar']
COLUMNAS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
N, PASO, SEMILLA, EPOCAS = 100, 50, 42, 65


# ----------------------------------------------------------------------------- datos
def cargar_ventanas():
    """Devuelve ventanas crudas (n,100,6), etiquetas y persona. Igual que entrenar.py."""
    W, ys, ps = [], [], []
    archivos = sorted((BASE / 'dataset').glob('*/*/*.csv'))
    if not archivos:
        raise SystemExit(f'No hay CSV en {BASE / "dataset"}')
    for f in archivos:
        persona, actividad = f.parts[-3:-1]
        if actividad not in CLASES:
            continue
        d = pd.read_csv(f)
        t = d.t_us.to_numpy(dtype=np.float64)
        tiempo = np.arange(t[0], t[-1], 20000.0)
        s = np.stack([np.interp(tiempo, t, d[c].to_numpy(dtype=float)) for c in COLUMNAS],
                     axis=1).astype(np.float32)
        for i in range(0, len(s) - N + 1, PASO):
            W.append(s[i:i + N]); ys.append(CLASES.index(actividad)); ps.append(persona)
    return np.stack(W), np.asarray(ys), np.asarray(ps)


# ----------------------------------------------------------------------------- características
def f_original(v):
    """6 estadísticas por canal: media, desv, mín, máx, cambio absoluto medio, RMS (como entrenar.py)."""
    o = []
    for e in v.T:
        o.extend((e.mean(), e.std(), e.min(), e.max(), np.abs(np.diff(e)).mean(), np.sqrt((e * e).mean())))
    return np.asarray(o, dtype=np.float32)


def f_invariante(v):
    """Características que NO dependen de cómo está inclinada la placa.
    Por cada tríada (acelerómetro y, si hay, giróscopo):
      - magnitud |v|: media, desv, mín, máx, cambio absoluto medio   (5)
      - por eje, quitando su valor medio: desv, cambio abs. medio, rango (3 x 3 = 9)
    """
    o = []
    for g in range(0, v.shape[1], 3):
        c = v[:, g:g + 3]
        m = np.linalg.norm(c, axis=1)
        o.extend((m.mean(), m.std(), m.min(), m.max(), np.abs(np.diff(m)).mean()))
        d = c - c.mean(axis=0)
        for e in d.T:
            o.extend((e.std(), np.abs(np.diff(e)).mean(), e.max() - e.min()))
    return np.asarray(o, dtype=np.float32)


VARIANTES = {  # nombre: (canales, función de características, etiqueta corta de figuras)
    'acelerometro': (3, f_original, 'acc'),
    'acelerometro_giroscopio': (6, f_original, 'imu'),
    'acel_inv': (3, f_invariante, 'acc_inv'),
    'imu_inv': (6, f_invariante, 'imu_inv'),
}
TITULOS = {'acelerometro': 'Acelerómetro', 'acelerometro_giroscopio': 'Acelerómetro + giróscopo',
           'acel_inv': 'Acelerómetro (invariante)', 'imu_inv': 'Acel. + giróscopo (invariante)'}


# ----------------------------------------------------------------------------- modelo
def construir(n_in):
    red = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_in,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5)])
    red.compile(optimizer=tf.keras.optimizers.Adam(0.002),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return red


def entrenar(x, y, tr, val=None):
    media = x[tr].mean(axis=0).astype(np.float32)
    desv = x[tr].std(axis=0).astype(np.float32)
    desv[desv < 1e-6] = 1.0
    z = ((x - media) / desv).astype(np.float32)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    red = construir(x.shape[1])
    # La persona reservada se usa SOLO para dibujar la curva; no hay parada temprana ni selección.
    vd = (z[val], y[val]) if val is not None else None
    h = red.fit(z[tr], y[tr], epochs=EPOCAS, batch_size=64, verbose=0, validation_data=vd)
    return red, media, desv, z, h.history


def cuantizar(red, calib):
    c = tf.lite.TFLiteConverter.from_keras_model(red)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    muestras = calib[np.random.default_rng(SEMILLA).choice(len(calib), min(400, len(calib)), replace=False)]
    c.representative_dataset = lambda: ([m[None, :].astype(np.float32)] for m in muestras)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def predecir_int8(binario, x):
    it = tf.lite.Interpreter(model_content=binario)
    it.allocate_tensors()
    e, s = it.get_input_details()[0], it.get_output_details()[0]
    esc, cero = e['quantization']
    out = []
    for fila in x:
        q = np.clip(np.rint(fila / esc + cero), -128, 127).astype(np.int8).reshape(1, -1)
        it.set_tensor(e['index'], q)
        it.invoke()
        out.append(int(it.get_tensor(s['index']).argmax()))
    return np.asarray(out)


def operadores(binario):
    try:
        it = tf.lite.Interpreter(model_content=binario)
        return sorted({d['op_name'] for d in it._get_ops_details()})
    except Exception:
        return []


def evaluar(y, p):
    return {'accuracy': float(accuracy_score(y, p)),
            'f1_macro': float(f1_score(y, p, average='macro', zero_division=0)),
            'matriz_confusion': confusion_matrix(y, p, labels=range(5)).tolist(), 'ventanas': int(len(y))}


# ----------------------------------------------------------------------------- cabeceras Arduino
def cab_modelo(b):
    l = [', '.join(f'0x{x:02x}' for x in b[i:i + 12]) for i in range(0, len(b), 12)]
    return ('#pragma once\n#include <stdint.h>\nalignas(16) const unsigned char g_har_model[] = {\n' +
            ',\n'.join(l) + f'\n}};\nconst unsigned int g_har_model_len = {len(b)};\n')


def cab_norm(m, d):
    def n(x):
        s = format(float(x), '.9g')
        return (s if any(c in s for c in '.eE') else s + '.0') + 'f'
    v = lambda a: ', '.join(map(n, a))
    return (f'#pragma once\nconstexpr int kHarFeatureCount = {len(m)};\n'
            f'const float kHarMean[kHarFeatureCount] = {{{v(m)}}};\n'
            f'const float kHarStd[kHarFeatureCount] = {{{v(d)}}};\n')


# ----------------------------------------------------------------------------- figuras
def fig_cm(cm, titulo, ruta):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(ETIQ, rotation=30)
    ax.set_yticks(range(5)); ax.set_yticklabels(ETIQ)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] / cm[i].sum() > .5 else 'black')
    ax.set_xlabel('Predicha'); ax.set_ylabel('Real')
    ax.set_title(f'{titulo}\n(3 grupos reservados, ventanas sumadas)', fontsize=9)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-exportar', action='store_true', help='no sobrescribir model.h / normalizador.h')
    a = ap.parse_args()
    figs = BASE / 'reporte' / 'figures'; figs.mkdir(parents=True, exist_ok=True)
    (BASE / 'resultados').mkdir(exist_ok=True); (BASE / 'modelos_v2').mkdir(exist_ok=True)

    W, y, personas = cargar_ventanas()
    ids = sorted(set(personas))
    print(f'Ventanas: {len(y)} | grupos: {ids}')
    R, hist_plot = {}, {}

    for nombre, (canales, fe, corto) in VARIANTES.items():
        x = np.stack([fe(w[:, :canales]) for w in W])
        folds, cm_f, cm_i, hv = {}, np.zeros((5, 5), int), np.zeros((5, 5), int), {}
        for p in ids:
            te, tr = personas == p, personas != p
            red, _, _, z, h = entrenar(x, y, tr, val=te)
            pf = red.predict(z[te], verbose=0).argmax(1)
            pi = predecir_int8(cuantizar(red, z[tr]), z[te])
            folds[p] = {'FP32': evaluar(y[te], pf), 'INT8': evaluar(y[te], pi),
                        'acuerdo_fp32_int8': float(np.mean(pf == pi))}
            cm_f += np.array(folds[p]['FP32']['matriz_confusion']); cm_i += np.array(folds[p]['INT8']['matriz_confusion'])
            hv[p] = {k: [float(v) for v in h[k]] for k in ('val_loss', 'val_accuracy')}
            print(f'  {nombre:25s} reservado={p}: FP32={folds[p]["FP32"]["accuracy"]:.3f} '
                  f'INT8={folds[p]["INT8"]["accuracy"]:.3f} acuerdo={folds[p]["acuerdo_fp32_int8"]:.3f}')

        # modelo final con los tres grupos
        todo = np.ones(len(y), bool)
        red, media, desv, z, h = entrenar(x, y, todo)
        b32 = tf.lite.TFLiteConverter.from_keras_model(red).convert()
        b8 = cuantizar(red, z)
        (BASE / 'modelos_v2' / f'{nombre}_FP32.tflite').write_bytes(b32)
        (BASE / 'modelos_v2' / f'{nombre}_INT8.tflite').write_bytes(b8)
        acc = lambda q: float(np.mean([folds[p][q]['accuracy'] for p in ids]))
        f1 = lambda q: float(np.mean([folds[p][q]['f1_macro'] for p in ids]))
        R[nombre] = {'caracteristicas': int(x.shape[1]), 'parametros': int(red.count_params()),
                     'bytes_fp32': len(b32), 'bytes_int8': len(b8),
                     'operadores_int8': operadores(b8), 'operadores_fp32': operadores(b32),
                     'acc_media': {'FP32': acc('FP32'), 'INT8': acc('INT8')},
                     'f1_media': {'FP32': f1('FP32'), 'INT8': f1('INT8')},
                     'acuerdo_medio': float(np.mean([folds[p]['acuerdo_fp32_int8'] for p in ids])),
                     'por_grupo': folds,
                     'cm_FP32': cm_f.tolist(), 'cm_INT8': cm_i.tolist(),
                     'perdida_final_entrenamiento': float(h['loss'][-1]),
                     'exactitud_final_entrenamiento': float(h['accuracy'][-1])}
        hist_plot[nombre] = {'final': {k: [float(v) for v in h[k]] for k in ('loss', 'accuracy')}, 'folds': hv}
        if nombre == 'acelerometro_giroscopio' and not a.no_exportar:
            d = BASE / 'arduino' / 'HAR_Inferencia'
            d.mkdir(parents=True, exist_ok=True)
            (d / 'model.h').write_text(cab_modelo(b8)); (d / 'normalizador.h').write_text(cab_norm(media, desv))
            print('  -> model.h y normalizador.h actualizados (variante original con giróscopo)')
        if corto in ('acc', 'imu', 'acc_inv', 'imu_inv'):
            for q, cm in (('fp32', cm_f), ('int8', cm_i)):
                fig_cm(cm, f'{TITULOS[nombre]}, {q.upper()}', figs / f'confusion_{corto}_{q}.pdf')

    # ---- curva de aprendizaje (variante con giróscopo original)
    hp = hist_plot['acelerometro_giroscopio']
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for k, (clave, tit) in enumerate((('loss', 'Pérdida'), ('accuracy', 'Exactitud'))):
        ax[k].plot(hp['final'][clave], color='#1f4e79', lw=2, label='entrenamiento (modelo final)')
        for i, p in enumerate(ids):
            ax[k].plot(hp['folds'][p]['val_' + clave], lw=1, alpha=.8, label=f'validación: {p} reservada')
        ax[k].set_xlabel('Época'); ax[k].set_title(tit); ax[k].grid(alpha=.25)
    ax[1].legend(fontsize=6.5, loc='lower right', frameon=False)
    fig.tight_layout(); fig.savefig(figs / 'curvas_aprendizaje.pdf'); plt.close(fig)

    # ---- comparación de tamaños (variantes originales)
    fig, ax = plt.subplots(figsize=(5.5, 3.2)); xs = np.arange(2); w = .35
    nm = ['acelerometro', 'acelerometro_giroscopio']
    for off, key, col, lab in ((-w / 2, 'bytes_fp32', '#6b7a8f', 'FP32'), (w / 2, 'bytes_int8', '#2a9d8f', 'INT8')):
        v = [R[n][key] for n in nm]; ax.bar(xs + off, v, w, color=col, label=lab)
        for i, val in enumerate(v):
            ax.text(xs[i] + off, val, val, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(['Acelerómetro', 'Acel. + giróscopo']); ax.set_ylabel('Tamaño del .tflite (bytes)'); ax.legend()
    fig.tight_layout(); fig.savefig(figs / 'comparacion_tamanios.pdf'); plt.close(fig)

    # ---- acuerdo FP32 vs INT8 por grupo reservado
    fig, ax = plt.subplots(figsize=(5.8, 3.2)); w = .35; xs = np.arange(len(ids))
    for off, n, col in ((-w / 2, 'acelerometro', '#6b7a8f'), (w / 2, 'acelerometro_giroscopio', '#2a9d8f')):
        v = [100 * R[n]['por_grupo'][p]['acuerdo_fp32_int8'] for p in ids]
        ax.bar(xs + off, v, w, color=col, label=TITULOS[n])
        for i, val in enumerate(v):
            ax.text(xs[i] + off, val, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f'{p} reservado' for p in ids]); ax.set_ylim(80, 102)
    ax.set_ylabel('Ventanas con igual predicción (%)'); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(figs / 'fp32_vs_int8.pdf'); plt.close(fig)

    # ---- mejora: variantes originales frente a invariantes
    fig, ax = plt.subplots(figsize=(6.4, 3.3)); xs = np.arange(len(VARIANTES)); w = .35
    for off, q, col in ((-w / 2, 'FP32', '#6b7a8f'), (w / 2, 'INT8', '#2a9d8f')):
        v = [R[n]['acc_media'][q] for n in VARIANTES]; ax.bar(xs + off, v, w, color=col, label=q)
        for i, val in enumerate(v):
            ax.text(xs[i] + off, val, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([TITULOS[n].replace(' (', '\n(').replace(' + ', '\n+ ') for n in VARIANTES], fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Exactitud media (3 grupos reservados)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / 'comparacion_variantes.pdf'); plt.close(fig)

    json.dump({'clases': CLASES, 'ventanas': int(len(y)), 'epocas': EPOCAS, 'semilla': SEMILLA,
               'variantes': R}, open(BASE / 'resultados' / 'metricas_v2.json', 'w'), ensure_ascii=False, indent=2)

    # ---- filas LaTeX
    L = ['% Generado por entrenar_v2.py - copie/pegue en main.tex', '% --- Tabla de tamaños y operadores']
    for n, tit, rep in (('acelerometro', 'Acelerómetro', 'FP32'), ('acelerometro', 'Acelerómetro', 'INT8'),
                        ('acelerometro_giroscopio', 'Acelerómetro + giróscopo', 'FP32'),
                        ('acelerometro_giroscopio', 'Acelerómetro + giróscopo', 'INT8')):
        b = R[n]['bytes_' + rep.lower()]; ops = ', '.join(R[n]['operadores_' + rep.lower()]).replace('_', r'\_')
        L.append(f'{tit} & {rep} & {b:,} & \\texttt{{{ops}}} \\\\'.replace(',', r'\,'))
    L.append('% --- Acuerdo FP32-INT8 (variante, acuerdo medio, delta acc, delta F1)')
    for n in ('acelerometro', 'acelerometro_giroscopio'):
        r = R[n]
        L.append(f'{TITULOS[n]} & {100 * r["acuerdo_medio"]:.1f}\\,\\% & {r["acc_media"]["INT8"] - r["acc_media"]["FP32"]:+.3f} '
                 f'& {r["f1_media"]["INT8"] - r["f1_media"]["FP32"]:+.3f} \\\\')
    L.append('% --- Variantes (características, acc FP32, F1 FP32, acc INT8, F1 INT8)')
    for n in VARIANTES:
        r = R[n]
        L.append(f'{TITULOS[n]} & {r["caracteristicas"]} & {r["acc_media"]["FP32"]:.3f} & {r["f1_media"]["FP32"]:.3f} '
                 f'& {r["acc_media"]["INT8"]:.3f} & {r["f1_media"]["INT8"]:.3f} \\\\')
    L.append('% --- Pérdida final del modelo con giróscopo (tabla de hiperparámetros)')
    L.append(f'% Pérdida final de entrenamiento: {R["acelerometro_giroscopio"]["perdida_final_entrenamiento"]:.4f}')
    (BASE / 'resultados' / 'snippets_reporte.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')

    print('\nRESUMEN (exactitud media, 3 grupos reservados)')
    for n in VARIANTES:
        r = R[n]
        print(f'  {TITULOS[n]:34s} car={r["caracteristicas"]:2d} FP32={r["acc_media"]["FP32"]:.3f} '
              f'INT8={r["acc_media"]["INT8"]:.3f} acuerdo={100*r["acuerdo_medio"]:.1f}% '
              f'bytes={r["bytes_fp32"]}/{r["bytes_int8"]}')
    print('\nListo. Figuras en reporte/figures/, métricas en resultados/.')


if __name__ == '__main__':
    main()
