/*
 * Robodog - UART Echo Test Sketch (Teensy 4.1)
 *
 * Amaç: Pi <-> Teensy USB-Serial bağlantısını doğrulamak için
 *       basit echo + heartbeat. robodog_ctrl.ino'dan ÖNCE bunu
 *       yükleyip uart_test.py ile bağlantıyı sınamak için kullanılır.
 *
 * Davranış:
 *   - Pi'den gelen her satırı "[ECHO] <gelen>\n" olarak geri yazar
 *   - Her 1 saniyede bir "[ALIVE] uptime=<ms>\n" basar
 *   - Built-in LED (pin 13) her saniye toggle eder → görsel onay
 *
 * Arduino IDE ayarları:
 *   Tools → Board       : Teensy 4.1
 *   Tools → USB Type    : Serial   ← kritik
 *   Tools → CPU Speed   : 600 MHz
 */

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  // İlk satır - upload sonrası canlı olduğunu göster
  Serial.println("[BOOT] uart_echo_test ready");
}

String inBuf;
unsigned long last_alive_ms = 0;
bool led_state = false;

void loop() {
  // Gelen byte'ları satıra topla, \n geldiğinde echo yap
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      Serial.print("[ECHO] ");
      Serial.println(inBuf);
      inBuf = "";
    } else if (c != '\r' && inBuf.length() < 200) {
      inBuf += c;
    }
  }

  // Her saniye heartbeat + LED toggle
  unsigned long now = millis();
  if (now - last_alive_ms >= 1000) {
    last_alive_ms = now;
    Serial.print("[ALIVE] uptime=");
    Serial.println(now);
    led_state = !led_state;
    digitalWrite(LED_BUILTIN, led_state);
  }
}
