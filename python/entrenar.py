from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
CLASES = ['reposo', 'caminar', 'correr', 'subir_escaleras', 'bajar_escaleras']
COLUMNAS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
N = 100
PASO = 50
SEMILLA = 42


def extraer(ventana):
    salida = []
    for eje in ventana.T:
        salida.extend((eje.mean(), eje.std(), eje.min(), eje.max(),
                       np.abs(np.diff(eje)).mean(), np.sqrt((eje * eje).mean())))
    return np.asarray(salida, dtype=np.float32)


def cargar(canales):
    xs, ys, personas = [], [], []
    archivos = sorted((BASE / 'dataset').glob('*/*/*.csv'))
    if len(archivos) != 15:
        raise ValueError(f'Se esperaban 15 CSV; encontrados: {len(archivos)}')
    for archivo in archivos:
        persona, actividad = archivo.parts[-3:-1]
        if actividad not in CLASES:
            raise ValueError(f'Actividad no reconocida: {actividad}')
        datos = pd.read_csv(archivo)
        if not {'t_us', *COLUMNAS}.issubset(datos.columns):
            raise ValueError(f'Columnas incorrectas: {archivo.name}')
        t = datos.t_us.to_numpy(dtype=np.float64)
        if len(t) < N or np.any(np.diff(t) <= 0):
            raise ValueError(f'Tiempos incorrectos: {archivo.name}')
        tiempo = np.arange(t[0], t[-1], 20000.0)
        senales = np.stack([np.interp(tiempo, t, datos[c].to_numpy(dtype=float))
                            for c in COLUMNAS[:canales]], axis=1).astype(np.float32)
        for inicio in range(0, len(senales) - N + 1, PASO):
            xs.append(extraer(senales[inicio:inicio + N]))
            ys.append(CLASES.index(actividad))
            personas.append(persona)
    return np.stack(xs), np.asarray(ys), np.asarray(personas)


def modelo(entradas):
    red = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(entradas,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    red.compile(optimizer=tf.keras.optimizers.Adam(0.002),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return red


def entrenar(x, y, mascara):
    media = x[mascara].mean(axis=0).astype(np.float32)
    desv = x[mascara].std(axis=0).astype(np.float32)
    desv[desv < 1e-6] = 1.0
    z = ((x - media) / desv).astype(np.float32)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    red = modelo(x.shape[1])
    red.fit(z[mascara], y[mascara], epochs=65, batch_size=64, verbose=0)
    return red, media, desv, z


def cuantizar(red, calibracion):
    convertidor = tf.lite.TFLiteConverter.from_keras_model(red)
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    muestras = calibracion[np.random.default_rng(SEMILLA).choice(
        len(calibracion), min(400, len(calibracion)), replace=False)]
    def representativos():
        for muestra in muestras:
            yield [muestra[np.newaxis, :].astype(np.float32)]
    convertidor.representative_dataset = representativos
    convertidor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    convertidor.inference_input_type = tf.int8
    convertidor.inference_output_type = tf.int8
    return convertidor.convert()


def predecir_int8(binario, x):
    interprete = tf.lite.Interpreter(model_content=binario)
    interprete.allocate_tensors()
    entrada = interprete.get_input_details()[0]
    salida = interprete.get_output_details()[0]
    escala, cero = entrada['quantization']
    if escala <= 0:
        raise ValueError('El modelo no tiene entrada INT8')
    predicciones = []
    for fila in x:
        valores = np.clip(np.rint(fila / escala + cero), -128, 127)
        interprete.set_tensor(entrada['index'], valores.reshape(1, -1).astype(np.int8))
        interprete.invoke()
        predicciones.append(int(interprete.get_tensor(salida['index']).argmax()))
    return np.asarray(predicciones)


def evaluar(y, pred):
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
        'matriz_confusion': confusion_matrix(y, pred, labels=range(5)).tolist(),
        'ventanas': len(y)
    }


def cabecera_modelo(datos):
    lineas = [', '.join(f'0x{b:02x}' for b in datos[i:i + 12])
              for i in range(0, len(datos), 12)]
    return ('#pragma once\n#include <stdint.h>\n'
            'alignas(16) const unsigned char g_har_model[] = {\n' +
            ',\n'.join(lineas) + '\n};\n' +
            f'const unsigned int g_har_model_len = {len(datos)};\n')


def cabecera_normalizador(media, desv):
    def numero(x):
        s = format(float(x), '.9g')
        return (s if any(c in s for c in '.eE') else s + '.0') + 'f'
    def vector(v):
        return ', '.join(map(numero, v))
    return (f'#pragma once\nconstexpr int kHarFeatureCount = {len(media)};\n'
            f'const float kHarMean[kHarFeatureCount] = {{{vector(media)}}};\n'
            f'const float kHarStd[kHarFeatureCount] = {{{vector(desv)}}};\n')


def main():
    (BASE / 'modelos').mkdir(exist_ok=True)
    (BASE / 'resultados').mkdir(exist_ok=True)
    informe = {'clases': CLASES, 'frecuencia_hz': 50, 'ventana_muestras': N,
               'paso_muestras': PASO, 'variantes': {}}
    for canales, nombre in [(3, 'acelerometro'), (6, 'acelerometro_giroscopio')]:
        x, y, personas = cargar(canales)
        evaluaciones = {}
        for persona in sorted(set(personas)):
            train = personas != persona
            test = personas == persona
            red, media, desv, z = entrenar(x, y, train)
            fp32 = red.predict(z[test], verbose=0).argmax(axis=1)
            int8 = cuantizar(red, z[train])
            pred_int8 = predecir_int8(int8, z[test])
            evaluaciones[persona] = {
                'FP32': evaluar(y[test], fp32),
                'INT8': evaluar(y[test], pred_int8)
            }
            print(nombre, persona, evaluaciones[persona]['FP32']['accuracy'],
                  evaluaciones[persona]['INT8']['accuracy'])
        red, media, desv, z = entrenar(x, y, np.ones(len(y), dtype=bool))
        fp32 = tf.lite.TFLiteConverter.from_keras_model(red).convert()
        int8 = cuantizar(red, z)
        (BASE / 'modelos' / f'{nombre}_FP32.tflite').write_bytes(fp32)
        (BASE / 'modelos' / f'{nombre}_INT8.tflite').write_bytes(int8)
        informe['variantes'][nombre] = {
            'num_ventanas': len(y), 'num_caracteristicas': x.shape[1],
            'modelo_fp32_bytes': len(fp32), 'modelo_int8_bytes': len(int8),
            'evaluacion_por_identificador': evaluaciones
        }
        if canales == 6:
            carpeta = BASE / 'arduino' / 'HAR_Inferencia'
            (carpeta / 'model.h').write_text(cabecera_modelo(int8))
            (carpeta / 'normalizador.h').write_text(cabecera_normalizador(media, desv))
    (BASE / 'resultados' / 'metricas.json').write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Modelos y cabeceras generados.')


if __name__ == '__main__':
    main()
