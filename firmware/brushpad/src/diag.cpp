// Wiring check for the mux and ADC path

#include <Arduino.h>

static const uint8_t MUX_S[3] = {19, 20, 21};

static void selectMuxChannel(int ch) {
  for (int b = 0; b < 3; b++) digitalWriteFast(MUX_S[b], (ch >> b) & 1);
}

static int readWithPull(uint8_t pin, uint8_t mode) {
  pinMode(pin, mode);
  delayMicroseconds(200);
  int v = analogRead(pin);
  pinMode(pin, INPUT_DISABLE);
  return v;
}

void setup() {
  Serial.begin(115200);
  for (int b = 0; b < 3; b++) pinMode(MUX_S[b], OUTPUT);
  pinMode(A0, INPUT_DISABLE);
  pinMode(A1, INPUT_DISABLE);
  analogReadResolution(12);
  analogReadAveraging(4);
  pinMode(13, OUTPUT);
}

static void printVolts(const char *label, uint8_t pin) {
  Serial.print(label);
  Serial.print(analogRead(pin) * 3.3f / 4095.0f, 2);
  Serial.print("V ");
}

void loop() {
  // A wire into pin 22 or 23 reads a voltage
  printVolts("pin22=", A8);
  printVolts("pin23=", A9);
  selectMuxChannel(0);
  delayMicroseconds(50);
  printVolts("ch0=", A0);
  Serial.print("| ch0..7:");
  for (int ch = 0; ch < 8; ch++) {
    selectMuxChannel(ch);
    delayMicroseconds(50);
    Serial.print(' ');
    Serial.print(analogRead(A0));
  }
  // D = driven, F = floating
  Serial.print(" | wired:");
  for (int ch = 0; ch < 8; ch++) {
    selectMuxChannel(ch);
    delayMicroseconds(50);
    int pu = readWithPull(A0, INPUT_PULLUP), pd = readWithPull(A0, INPUT_PULLDOWN);
    Serial.print(pu - pd < 500 ? " D" : " F");
  }
  selectMuxChannel(0);
  delayMicroseconds(50);
  Serial.print(" | pu ");
  Serial.print(readWithPull(A0, INPUT_PULLUP));
  Serial.print(" pd ");
  Serial.print(readWithPull(A0, INPUT_PULLDOWN));
  Serial.print(" | A1 ");
  Serial.print(analogRead(A1));

  // Pins tied to ground
  Serial.print(" | grounded:");
  for (int p = 0; p <= 23; p++) {
    if (p == 13) continue;  // LED
    pinMode(p, INPUT_PULLUP);
    delayMicroseconds(100);
    if (!digitalReadFast(p)) {
      Serial.print(' ');
      Serial.print(p);
    }
    pinMode(p, INPUT_DISABLE);
  }
  // Pins tied to 3.3 V
  Serial.print(" | at 3.3V:");
  for (int p = 0; p <= 23; p++) {
    if (p == 13) continue;
    pinMode(p, INPUT_PULLDOWN);
    delayMicroseconds(100);
    if (digitalReadFast(p)) {
      Serial.print(' ');
      Serial.print(p);
    }
    pinMode(p, INPUT_DISABLE);
  }
  for (int b = 0; b < 3; b++) pinMode(MUX_S[b], OUTPUT);
  Serial.println();
  digitalWriteFast(13, !digitalReadFast(13));
  delay(200);
}
