"""
Robodog - UART Bağlantı Testi (Pi → Teensy 4.1)

Önce Teensy'ye teensy_firmware/uart_echo_test.ino yüklenmiş olmalı.

Davranış:
  1. /dev/ttyACM0 portunu açar
  2. Birkaç "PING N" yollar, Teensy'nin "[ECHO] PING N" cevabını bekler
  3. Aynı zamanda Teensy'nin saniyede bir gönderdiği "[ALIVE]" satırlarını da gösterir
  4. Sonunda kaç ping başarılı olduğunu yazar

Kullanım:
  python uart_test.py
  python uart_test.py --port /dev/ttyACM0 -n 5
"""

import argparse
import sys
import time

import serial


DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("-n", "--pings", type=int, default=3,
                        help="Kaç ping yollansın (default: 3)")
    args = parser.parse_args()

    print(f"[uart-test] {args.port} @ {args.baud} açılıyor...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    except (serial.SerialException, OSError) as e:
        print(f"[uart-test] HATA: port açılamadı: {e}")
        print("  Kontrol et:")
        print("    - Teensy bağlı mı?           ls /dev/ttyACM*")
        print("    - dialout grubunda mısın?    groups")
        print("    - sudo usermod -a -G dialout $USER  (sonra logout/login)")
        sys.exit(1)

    time.sleep(0.5)            # Teensy USB CDC settle
    ser.reset_input_buffer()
    print(f"[uart-test] {args.pings} ping gönderilecek.\n")

    ok = 0
    for i in range(1, args.pings + 1):
        msg = f"PING {i}"
        ser.write((msg + "\n").encode("ascii"))
        print(f"  -> gönderildi: {msg}")

        deadline = time.monotonic() + 2.0     # echo için max 2s bekle
        echoed = False
        while time.monotonic() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            print(f"  <- alındı:     {line}")
            if line.startswith("[ECHO]") and msg in line:
                echoed = True
                break
            # [ALIVE] veya [BOOT] satırları gelirse onları da bas, eşleşme bekleme

        if echoed:
            ok += 1
            print(f"  [OK] ping {i}\n")
        else:
            print(f"  [TIMEOUT] ping {i}\n")

        time.sleep(0.5)

    ser.close()

    print("=" * 50)
    print(f"  Sonuç: {ok}/{args.pings} ping başarılı")
    if ok == args.pings:
        print("  [OK] UART bağlantısı çalışıyor.")
    elif ok > 0:
        print("  [UYARI] Kısmi cevap. Kablo/parazit olabilir.")
    else:
        print("  [HATA] Hiç cevap gelmedi. Kontrol:")
        print("    - Teensy'de uart_echo_test.ino yüklü mü?")
        print("    - Arduino IDE'de USB Type 'Serial' seçildi mi?")
        print("    - Doğru port mu açtın? (ls /dev/ttyACM*)")
        print("    - Başka bir program portu kullanıyor olabilir mi?")


if __name__ == "__main__":
    main()
