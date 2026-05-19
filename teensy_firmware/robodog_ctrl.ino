/*
 * Robodog - Teensy 4.1 Control Firmware
 *
 * USB-Serial üzerinden Raspberry Pi'den gelen görüş paketlerini okur,
 * state machine + P-controller ile motor sürücülere komut verir.
 *
 * Paket formatı (Pi → Teensy):
 *   F,<status>,<dx>,<dy>,<dist_cm>,<radius_px>\n
 *
 *   status      : 0 = top yok, 1 = top var
 *   dx, dy      : kare merkezine göre piksel offset (işaretli, dx>0 sağ)
 *   dist_cm     : tahmini mesafe (float)
 *   radius_px   : top yarıçapı (kalite göstergesi)
 *
 * State machine:
 *   SEARCH    : top yok → yerinde dön (taramaya devam)
 *   TRACK     : top var → P-controller (dx ile yaw, dist ile forward)
 *   ARRIVED   : top yeterince yakın → dur
 *   SAFE_STOP : Pi'den WATCHDOG_MS+ zaman paket gelmedi → dur
 *
 * Motor sürücü stub'ı drive(yaw, forward) içinde — kendi motor kütüphanenize
 * göre doldurun (4 bacaklı yürüyüş için inverse kinematics burada eklenmeli).
 *
 * Bağlantı:
 *   Pi USB-A port  →  Teensy 4.1 USB-C  (Pi'de /dev/ttyACM0)
 *   Baud: 115200 (USB CDC için aslında dummy, ama bridge tarafıyla aynı yaz)
 */

#include <Arduino.h>

// ── Frame Parametreleri ──────────────────────────────────────────────────
static const int FRAME_WIDTH  = 640;
static const int FRAME_HEIGHT = 480;

// ── Kontrol Parametreleri ────────────────────────────────────────────────
// Kp_yaw: dx = FRAME_WIDTH/2 (en uç) iken yaw çıktısı 1.0 olsun.
static const float Kp_yaw       = 1.0f / (FRAME_WIDTH / 2.0f);
static const float Kp_forward   = 0.02f;          // her cm hata → forward
static const float TARGET_CM    = 25.0f;          // bu mesafede kalmaya çalış
static const float ARRIVED_CM   = 15.0f;          // bundan yakına gidersek dur
static const float DEAD_BAND_PX = 20.0f;          // ±20 px ortada say
static const float SEARCH_YAW   = 0.4f;           // SEARCH'te dönüş hızı
static const float MAX_FWD      = 0.6f;
static const float TURN_ONLY_FRAC = 0.30f;        // |dx| > %30 → ileri gitme, sadece dön

static const unsigned long WATCHDOG_MS = 500;     // Pi sessizse SAFE_STOP

// ── State ────────────────────────────────────────────────────────────────
enum State { SEARCH, TRACK, ARRIVED, SAFE_STOP };
static State state = SAFE_STOP;
static unsigned long last_packet_ms = 0;

// ─────────────────────────────────────────────────────────────────────────
//  Motor sürücü stub
//  TODO: kendi motor kütüphanenize / inverse kinematics'inize bağlayın
// ─────────────────────────────────────────────────────────────────────────
static void drive(float yaw, float forward) {
  yaw     = constrain(yaw,     -1.0f, 1.0f);
  forward = constrain(forward, -1.0f, 1.0f);

  // Örnek differential drive (4 bacaklılar için inverse kinematics'le değiştirin):
  // float leftSpeed  = forward - yaw;
  // float rightSpeed = forward + yaw;
  // leftMotor.write(leftSpeed);
  // rightMotor.write(rightSpeed);
}

static void stopMotors() {
  drive(0.0f, 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────
//  Paket parser
// ─────────────────────────────────────────────────────────────────────────
struct Packet {
  int   status;
  int   dx;
  int   dy;
  float dist_cm;
  int   radius_px;
};

// "F,1,+120,-5,52.3,28" → Packet
static bool parsePacket(const String &line, Packet &p) {
  if (line.length() < 3 || line.charAt(0) != 'F' || line.charAt(1) != ',') {
    return false;
  }

  String tokens[5];
  int from = 2;
  for (int i = 0; i < 5; i++) {
    int comma = line.indexOf(',', from);
    if (i < 4) {
      if (comma == -1) return false;
      tokens[i] = line.substring(from, comma);
      from = comma + 1;
    } else {
      // son token sonuna kadar
      tokens[i] = line.substring(from);
    }
  }

  p.status    = tokens[0].toInt();
  p.dx        = tokens[1].toInt();
  p.dy        = tokens[2].toInt();
  p.dist_cm   = tokens[3].toFloat();
  p.radius_px = tokens[4].toInt();
  return true;
}

// ─────────────────────────────────────────────────────────────────────────
//  Kontrol mantığı — gelen pakete göre state + motor güncelle
// ─────────────────────────────────────────────────────────────────────────
static void handlePacket(const Packet &p) {
  last_packet_ms = millis();

  // Top yok → tarama
  if (p.status == 0) {
    state = SEARCH;
    drive(SEARCH_YAW, 0.0f);
    return;
  }

  // Yeterince yakın → dur
  if (p.dist_cm > 0.0f && p.dist_cm <= ARRIVED_CM) {
    state = ARRIVED;
    stopMotors();
    return;
  }

  // İzle: ortala + yaklaş
  state = TRACK;

  float yaw = 0.0f;
  if (abs(p.dx) > DEAD_BAND_PX) {
    yaw = Kp_yaw * (float)p.dx;
  }

  float fwd = 0.0f;
  if (p.dist_cm > 0.0f) {
    fwd = Kp_forward * (p.dist_cm - TARGET_CM);
    if (fwd < 0.0f) fwd = 0.0f;      // geri gitme; uzaktaysa ileri, yakındaysa dur
    if (fwd > MAX_FWD) fwd = MAX_FWD;
  }

  // Top çok sapmışsa önce dön, sonra yaklaş (saçma yörüngeyi engelle)
  if (abs(p.dx) > FRAME_WIDTH * TURN_ONLY_FRAC) {
    fwd = 0.0f;
  }

  drive(yaw, fwd);
}

// ─────────────────────────────────────────────────────────────────────────
//  Setup / Loop
// ─────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  stopMotors();
  // USB Serial bekleme (max 2s) — Pi tarafının açılmasını beklemiyoruz,
  // sadece host'un hazır olduğunu varsayıyoruz
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 2000) { /* spin */ }
}

// LED state göstergesi (debug):
//   SAFE_STOP : sönük
//   SEARCH    : yavaş yanıp sönüyor   (~1 Hz)
//   TRACK     : hızlı yanıp sönüyor   (~5 Hz)
//   ARRIVED   : sürekli yanıyor
static void updateStateLed() {
  static unsigned long last_toggle_ms = 0;
  static bool led_on = false;

  unsigned long period = 0;     // 0 = sabit
  bool solid_on = false;

  switch (state) {
    case SAFE_STOP: solid_on = false;          break;
    case SEARCH:    period   = 500;            break;  // 1 Hz toggle
    case TRACK:     period   = 100;            break;  // 5 Hz toggle
    case ARRIVED:   solid_on = true;           break;
  }

  if (period > 0) {
    unsigned long now = millis();
    if (now - last_toggle_ms >= period) {
      last_toggle_ms = now;
      led_on = !led_on;
      digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
    }
  } else {
    digitalWrite(LED_BUILTIN, solid_on ? HIGH : LOW);
    led_on = solid_on;
  }
}

static String inBuf;

void loop() {
  // Gelen baytları satır olarak topla
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      Packet p;
      if (parsePacket(inBuf, p)) {
        handlePacket(p);
      }
      inBuf = "";
    } else if (c != '\r' && inBuf.length() < 80) {
      inBuf += c;
    } else if (inBuf.length() >= 80) {
      inBuf = "";   // çöp/yanlış paket, buffer'ı temizle
    }
  }

  // Watchdog — Pi sessizse durdur
  if (millis() - last_packet_ms > WATCHDOG_MS) {
    if (state != SAFE_STOP) {
      state = SAFE_STOP;
      stopMotors();
    }
  }

  updateStateLed();
}
