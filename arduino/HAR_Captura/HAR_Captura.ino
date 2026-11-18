#include "imu_config.h"
#if HAR_IMU_REV2
  #include <Arduino_BMI270_BMM150.h>
#else
  #include <Arduino_LSM9DS1.h>
#endif

static constexpr uint32_t SAMPLE_PERIOD_US = 20000;
uint32_t nextSample = 0;
uint32_t dropped = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 6000) {}
  if (!IMU.begin()) {
    Serial.println("#ERROR,IMU_NO_DETECTADA. Revisa imu_config.h y libreria.");
    while (true) { delay(1000); }
  }
  Serial.println("#INFO,50Hz,ax_ay_az=g,gx_gy_gz=grados_s");
  Serial.println("#COLUMNAS,DATA,t_us,ax,ay,az,gx,gy,gz");
  nextSample = micros() + SAMPLE_PERIOD_US;
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextSample) < 0) return;
  nextSample += SAMPLE_PERIOD_US;
  if ((int32_t)(now - nextSample) > (int32_t)SAMPLE_PERIOD_US) nextSample = now + SAMPLE_PERIOD_US;

  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    ++dropped;
    if ((dropped % 100) == 1) Serial.println("#WARN,muestra_sin_lectura_fresca");
    return;
  }
  float ax,ay,az,gx,gy,gz;
  IMU.readAcceleration(ax,ay,az);
  IMU.readGyroscope(gx,gy,gz);
  Serial.print("DATA,"); Serial.print(now); Serial.print(',');
  Serial.print(ax,6); Serial.print(','); Serial.print(ay,6); Serial.print(',');
  Serial.print(az,6); Serial.print(','); Serial.print(gx,6); Serial.print(',');
  Serial.print(gy,6); Serial.print(','); Serial.println(gz,6);
}
