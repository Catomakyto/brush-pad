// Teensy 4.0 scanner for the Velostat brush pad
// Frame: B3 55, seq, w, h, piezo, taxels, CRC-16
// Send t for a text grid, b for binary frames

#include <Arduino.h>

// Grid size
#ifndef NROWS
#define NROWS 16
#endif
#ifndef NCOLS
#define NCOLS 16
#endif

// Pins, R8 uses pin 22
static const uint8_t ROW_PINS[16] = {0, 1, 2, 3, 4, 5, 6, 22, 8, 9, 10, 11, 12, 16, 17, 18};
static const uint8_t MUX_S[3] = {19, 20, 21};  // shared mux select
static const uint8_t ADC_PIN[2] = {A0, A1};    // mux A, mux B
static const uint8_t LED = 13;

// Timing
static const uint32_t ROW_SETTLE_US = 30;  // row settle time
static const uint32_t MUX_SETTLE_US = 2;

static uint16_t grid[NROWS][NCOLS];
static uint8_t frame[8 + 2 * NROWS * NCOLS + 2];
static uint16_t seq = 0;
static bool textMode = false;

static uint16_t crc16_ccitt(const uint8_t *data, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
  }
  return crc;
}

static inline void selectRow(int r) {
  pinMode(ROW_PINS[r], OUTPUT);  // latch set LOW in setup()
  digitalWriteFast(ROW_PINS[r], LOW);
}

static inline void releaseRow(int r) { pinMode(ROW_PINS[r], INPUT); }

static inline void selectMuxChannel(int ch) {
  for (int b = 0; b < 3; b++) digitalWriteFast(MUX_S[b], (ch >> b) & 1);
}

static void scan() {
  for (int r = 0; r < NROWS; r++) {
    selectRow(r);
    delayMicroseconds(ROW_SETTLE_US);
    for (int ch = 0; ch < 8 && ch < NCOLS; ch++) {
      selectMuxChannel(ch);
      delayMicroseconds(MUX_SETTLE_US);
      grid[r][ch] = analogRead(ADC_PIN[0]);
      if (ch + 8 < NCOLS) grid[r][ch + 8] = analogRead(ADC_PIN[1]);
    }
    releaseRow(r);
  }
}

static void sendBinary() {
  frame[0] = 0xB3;
  frame[1] = 0x55;
  frame[2] = seq & 0xFF;
  frame[3] = seq >> 8;
  frame[4] = NCOLS;  // w
  frame[5] = NROWS;  // h
  frame[6] = 0;      // piezo not fitted
  frame[7] = 0;
  size_t i = 8;
  for (int r = 0; r < NROWS; r++)
    for (int c = 0; c < NCOLS; c++) {
      frame[i++] = grid[r][c] & 0xFF;
      frame[i++] = grid[r][c] >> 8;
    }
  uint16_t crc = crc16_ccitt(frame + 2, i - 2);
  frame[i++] = crc & 0xFF;
  frame[i++] = crc >> 8;
  Serial.write(frame, i);
  seq++;
}

static void sendText() {
  static elapsedMillis since;
  if (since < 100) return;
  since = 0;
  Serial.printf("--- frame %u (raw ADC, 0-4095; rest ~680 = V_REF) ---\n", seq++);
  for (int r = 0; r < NROWS; r++) {
    Serial.printf("R%-2d", r + 1);
    for (int c = 0; c < NCOLS; c++) Serial.printf(" %4u", grid[r][c]);
    Serial.println();
  }
}

void setup() {
  for (int r = 0; r < 16; r++) {
    pinMode(ROW_PINS[r], OUTPUT);
    digitalWriteFast(ROW_PINS[r], LOW);  // latch LOW
    pinMode(ROW_PINS[r], INPUT);         // idle = high-Z
  }
  for (int b = 0; b < 3; b++) pinMode(MUX_S[b], OUTPUT);
  pinMode(LED, OUTPUT);
  analogReadResolution(12);
  analogReadAveraging(4);
  Serial.begin(2000000);  // baud ignored over USB
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 't') textMode = true;
    if (c == 'b') textMode = false;
  }
  scan();
  if (textMode) sendText();
  else sendBinary();

  static elapsedMillis blink;  // heartbeat
  if (blink > 250) {
    blink = 0;
    digitalToggleFast(LED);
  }
}
