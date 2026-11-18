#include <math.h>
#include "imu_config.h"
#if HAR_IMU_REV2
  #include <Arduino_BMI270_BMM150.h>
#else
  #include <Arduino_LSM9DS1.h>
#endif
#include "model.h"
#include "normalizador.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static constexpr int kN = 100;
static constexpr int kChannels = 6;
static constexpr int kStride = 50;
static constexpr uint32_t kPeriodUs = 20000;
static constexpr int kArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[kArenaSize];
float rawBuffer[kN][kChannels];
int writeIndex = 0, nSamples = 0, sinceInference = 0;
uint32_t nextSampleUs = 0;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

const char* classes[5] = {"REPOSO", "CAMINAR", "CORRER", "SUBIR_ESCALERAS", "BAJAR_ESCALERAS"};

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

void inferWindow() {
  float features[kHarFeatureCount];
  extractFeatures(features);
  for(int i=0;i<kHarFeatureCount;i++) {
    float z=(features[i]-kHarMean[i])/kHarStd[i];
    float scale=inputTensor->params.scale;
    int zp=inputTensor->params.zero_point;
    if(scale<=0) {Serial.println("#ERROR,input_scale=0"); return;}
    int q=(int)lroundf(z/scale)+zp;
    if(q<-128)q=-128; if(q>127)q=127;
    inputTensor->data.int8[i]=(int8_t)q;
  }
  uint32_t t0=micros();
  TfLiteStatus status=interpreter->Invoke();
  uint32_t duration=micros()-t0;
  if(status!=kTfLiteOk) {Serial.println("#ERROR,Invoke"); return;}
  int best=0;
  for(int j=1;j<5;j++) if(outputTensor->data.int8[j]>outputTensor->data.int8[best]) best=j;
  Serial.print("#PRED,"); Serial.print(classes[best]);
  Serial.print(",us="); Serial.print(duration);
  Serial.print(",arena_reserved="); Serial.println(kArenaSize);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis()<6000) {}
  if(!IMU.begin()) {Serial.println("#ERROR,IMU_NO_DETECTADA"); while(true)delay(1000);}
  const tflite::Model* model=tflite::GetModel(g_har_model);
  if(model->version()!=TFLITE_SCHEMA_VERSION) {Serial.println("#ERROR,SCHEMA"); while(true)delay(1000);}
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterp(model,resolver,tensorArena,kArenaSize);
  interpreter=&staticInterp;
  if(interpreter->AllocateTensors()!=kTfLiteOk) {Serial.println("#ERROR,AllocateTensors"); while(true)delay(1000);}
  inputTensor=interpreter->input(0); outputTensor=interpreter->output(0);
  if(inputTensor->type!=kTfLiteInt8 || outputTensor->type!=kTfLiteInt8 ||
     inputTensor->bytes != kHarFeatureCount || outputTensor->bytes != 5) {
    Serial.println("#ERROR,FORMATO_INT8_O_DIMENSIONES"); while(true)delay(1000);
  }
  Serial.print("#READY,features="); Serial.print(kHarFeatureCount);
  Serial.print(",model_bytes="); Serial.print(g_har_model_len);
  Serial.print(",arena_used="); Serial.println(interpreter->arena_used_bytes());
  nextSampleUs=micros()+kPeriodUs;
}

void loop() {
  uint32_t now=micros();
  if((int32_t)(now-nextSampleUs)<0)return;
  nextSampleUs+=kPeriodUs;
  if((int32_t)(now-nextSampleUs)>(int32_t)kPeriodUs) nextSampleUs=now+kPeriodUs;
  if(!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return;
  float ax,ay,az,gx,gy,gz;
  IMU.readAcceleration(ax,ay,az); IMU.readGyroscope(gx,gy,gz);
  rawBuffer[writeIndex][0]=ax; rawBuffer[writeIndex][1]=ay; rawBuffer[writeIndex][2]=az;
  rawBuffer[writeIndex][3]=gx; rawBuffer[writeIndex][4]=gy; rawBuffer[writeIndex][5]=gz;
  writeIndex=(writeIndex+1)%kN;
  if(nSamples<kN) {nSamples++; if(nSamples==kN)inferWindow(); return;}
  if(++sinceInference>=kStride) {sinceInference=0;inferWindow();}
}
