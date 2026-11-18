// HAR_Replay.ino
// Reproduce ventanas de 2 s YA GRABADAS (replay_data.h) por el mismo camino que usa HAR_Inferencia:
//   extracción de características -> normalización -> cuantización INT8 -> TFLite Micro -> clase.
// NO usa el IMU. Sirve para: medir latencia y memoria reales en la placa y mostrar la
// clasificación de las cinco actividades en el Monitor Serie cuando no hay un IMU funcional.
// IMPORTANTE (reporte): son datos reproducidos, no lectura en vivo.
//
// Antes: python python/entrenar_v2.py  y  python python/generar_replay.py
#include <Chirale_TensorFlowLite.h>
#include <math.h>
#include "model.h"
#include "normalizador.h"
#include "replay_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static constexpr int kN = 100;
static constexpr int kChannels = 6;
static constexpr int kArenaSize = 32 * 1024;   // igual que HAR_Inferencia
static constexpr int kReps = 20;               // Invoke() repetidos por ventana para estadística
alignas(16) uint8_t tensorArena[kArenaSize];
float rawBuffer[kN][kChannels];
int8_t qin[kHarFeatureCount];                  // entrada ya cuantizada (se reescribe antes de cada Invoke)
int writeIndex = 0;                            // 0: la ventana queda en orden temporal
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

// ---- misma extracción que HAR_Inferencia.ino (copiada tal cual) ----
void extractFeatures(float features[kHarFeatureCount]) {
  int p=0;
  for (int c=0;c<kChannels;c++) {
    float sum=0, sum2=0, mn=1e30f, mx=-1e30f, dif=0, previous=0;
    for (int i=0;i<kN;i++) {
      float v=rawBuffer[(writeIndex+i)%kN][c];
      sum+=v; sum2+=v*v;
      if(v<mn)mn=v; if(v>mx)mx=v;
      if(i>0) dif+=fabsf(v-previous);
      previous=v;
    }
    float mean=sum/kN;
    float variance=fmaxf(0.0f,sum2/kN-mean*mean);
    features[p++]=mean; features[p++]=sqrtf(variance);
    features[p++]=mn; features[p++]=mx;
    features[p++]=dif/(kN-1); features[p++]=sqrtf(sum2/kN);
  }
}

// TFLite Micro puede reutilizar la memoria del tensor de entrada durante Invoke(); por eso
// la entrada debe escribirse de nuevo ANTES de cada Invoke().
void cargarEntrada() {
  for (int i=0;i<kHarFeatureCount;i++) inputTensor->data.int8[i]=qin[i];
}

// Carga la ventana k, extrae, normaliza, cuantiza. Devuelve el tiempo de extracción (us).
uint32_t prepararVentana(int k) {
  for (int i=0;i<kN;i++) for (int c=0;c<kChannels;c++) rawBuffer[i][c]=kReplayData[k][i][c];
  writeIndex=0;
  float features[kHarFeatureCount];
  uint32_t t0=micros();
  extractFeatures(features);
  uint32_t dt=micros()-t0;
  const float scale=inputTensor->params.scale;
  const int zp=inputTensor->params.zero_point;
  for (int i=0;i<kHarFeatureCount;i++) {
    float z=(features[i]-kHarMean[i])/kHarStd[i];
    int q=(int)lroundf(z/scale)+zp;
    if(q<-128)q=-128; if(q>127)q=127;
    qin[i]=(int8_t)q;
  }
  cargarEntrada();
  return dt;
}

int claseSalida() {
  int best=0;
  for(int j=1;j<5;j++) if(outputTensor->data.int8[j]>outputTensor->data.int8[best]) best=j;
  return best;
}

void detener(const char* msg) { Serial.println(msg); while(true) delay(1000); }

void setup() {
  Serial.begin(115200);
  while (!Serial && millis()<6000) {}
  Serial.println("#MODO,REPLAY_DE_VENTANAS_GRABADAS (sin IMU)");
  const tflite::Model* model=tflite::GetModel(g_har_model);
  if(model->version()!=TFLITE_SCHEMA_VERSION) detener("#ERROR,SCHEMA");
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterp(model,resolver,tensorArena,kArenaSize);
  interpreter=&staticInterp;
  if(interpreter->AllocateTensors()!=kTfLiteOk) detener("#ERROR,AllocateTensors");
  inputTensor=interpreter->input(0); outputTensor=interpreter->output(0);
  if(inputTensor->type!=kTfLiteInt8 || outputTensor->type!=kTfLiteInt8 ||
     inputTensor->bytes != kHarFeatureCount || outputTensor->bytes != 5) detener("#ERROR,FORMATO_INT8_O_DIMENSIONES");
  Serial.print("#READY,features="); Serial.print(kHarFeatureCount);
  Serial.print(",model_bytes="); Serial.print(g_har_model_len);
  Serial.print(",arena_used="); Serial.print((unsigned long)interpreter->arena_used_bytes());
  Serial.print(",arena_reserved="); Serial.println(kArenaSize);

  // ---------- Prueba: precisión y latencia sobre todas las ventanas ----------
  int aciertos=0; unsigned long n=0;
  double media=0, m2=0, featSum=0; uint32_t mn=0xFFFFFFFF, mx=0;
  for (int k=0;k<kReplayWindows;k++) {
    uint32_t tf_=prepararVentana(k);
    featSum+=tf_;
    double sumaVentana=0;
    int pred=0;
    for (int r=0;r<kReps;r++) {
      cargarEntrada();                          // no se incluye en el tiempo medido
      uint32_t t0=micros();
      TfLiteStatus st=interpreter->Invoke();
      uint32_t dt=micros()-t0;
      if(st!=kTfLiteOk) detener("#ERROR,Invoke");
      n++; double d=dt-media; media+=d/n; m2+=d*(dt-media);   // Welford
      if(dt<mn)mn=dt; if(dt>mx)mx=dt; sumaVentana+=dt;
      if(r==0) pred=claseSalida();              // la clase se lee tras la primera ejecución
    }
    int real=kReplayLabel[k];
    if(pred==real) aciertos++;
    Serial.print("#REPLAY,esperada="); Serial.print(kReplayName[real]);
    Serial.print(",pred="); Serial.print(kReplayName[pred]);
    Serial.print(",ok="); Serial.print(pred==real?1:0);
    Serial.print(",feat_us="); Serial.print(tf_);
    Serial.print(",invoke_medio_us="); Serial.println((unsigned long)(sumaVentana/kReps));
  }
  Serial.print("#RESUMEN,ventanas="); Serial.print(kReplayWindows);
  Serial.print(",aciertos="); Serial.print(aciertos);
  Serial.print(",exactitud_pct="); Serial.print((100*aciertos)/kReplayWindows);
  Serial.print(",invoke_n="); Serial.print(n);
  Serial.print(",media_us="); Serial.print((unsigned long)(media+0.5));
  Serial.print(",desv_us="); Serial.print((unsigned long)(sqrt(n>1?m2/(n-1):0)+0.5));
  Serial.print(",min_us="); Serial.print(mn);
  Serial.print(",max_us="); Serial.print(mx);
  Serial.print(",feat_media_us="); Serial.println((unsigned long)(featSum/kReplayWindows+0.5));
  Serial.println("#DEMO,reproduccion continua (una ventana por segundo)");
}

// ---------- Demostración: una "predicción" por segundo, recorriendo las 5 actividades ----------
int k = 0;
void loop() {
  prepararVentana(k);
  uint32_t t0=micros();
  interpreter->Invoke();
  uint32_t dt=micros()-t0;
  int pred=claseSalida(); int real=kReplayLabel[k];
  Serial.print("#PRED,"); Serial.print(kReplayName[pred]);
  Serial.print(",esperada="); Serial.print(kReplayName[real]);
  Serial.print(",us="); Serial.print(dt);
  Serial.print(",arena_reserved="); Serial.println(kArenaSize);
  k=(k+1)%kReplayWindows;
  delay(1000);
}
