"""
Robodog - Serial Bridge (Pi 4 → Teensy 4.1, USB-Serial)

red_ball_detector modülünü kullanarak kameradan kırmızı topu tespit eder,
sonucu Teensy 4.1'e CSV formatında yollar:

    F,<status>,<dx>,<dy>,<dist_cm>,<radius_px>\n

  status      : 0 = top yok, 1 = top var
  dx, dy      : merkez offset (piksel, işaretli). dx>0 sağ, dy>0 aşağı.
  dist_cm     : tahmini mesafe (float). Yoksa 0.
  radius_px   : tespit edilen yarıçap (kalite göstergesi).

Teensy bu paketleri state machine + P-controller ile motor sürer.

Çalıştırma:
    python serial_bridge.py
    python serial_bridge.py --port /dev/ttyACM0 --hz 20
    python serial_bridge.py --dry-run            # serial açmaz, paketleri stdout'a basar
"""

import argparse
import os
import sys
import time

import serial
from dotenv import load_dotenv

from red_ball_detector import (
    CaptureThread,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    detect_red_ball,
    estimate_distance,
    init_camera,
)

load_dotenv()

DEFAULT_PORT    = os.getenv("SERIAL_PORT", "/dev/ttyACM0")
DEFAULT_BAUD    = int(os.getenv("SERIAL_BAUD", "115200"))
DEFAULT_HZ      = int(os.getenv("BRIDGE_HZ",   "20"))
RECONNECT_DELAY = 1.0  # saniye

NO_BALL_PACKET = b"F,0,0,0,0,0\n"


def make_packet(detection: dict | None) -> bytes:
    if detection is None:
        return NO_BALL_PACKET

    cx, cy = detection["center"]
    dx = cx - FRAME_WIDTH  // 2
    dy = cy - FRAME_HEIGHT // 2
    r  = detection["radius_px"]
    dist = estimate_distance(r)

    return f"F,1,{dx:+d},{dy:+d},{dist:.1f},{r}\n".encode("ascii")


def open_serial(port: str, baud: int) -> serial.Serial | None:
    try:
        ser = serial.Serial(port, baud, timeout=0.1, write_timeout=0.5)
        time.sleep(0.5)             # Teensy USB enum & reset settle
        ser.reset_input_buffer()
        print(f"[bridge] serial açık: {port} @ {baud}")
        return ser
    except (serial.SerialException, OSError) as e:
        print(f"[bridge] serial açılamadı ({port}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--hz",   type=int, default=DEFAULT_HZ)
    parser.add_argument("--dry-run", action="store_true",
                        help="serial port açma, paketleri stdout'a bas")
    parser.add_argument("--verbose", action="store_true",
                        help="gönderilen her paketi logla")
    args = parser.parse_args()

    period = 1.0 / args.hz

    cam = init_camera()
    capture_t = CaptureThread(cam)
    capture_t.start()

    ser = None if args.dry_run else open_serial(args.port, args.baud)

    mode = "DRY-RUN" if args.dry_run else f"{args.port}"
    print(f"[bridge] çalışıyor — {args.hz} Hz | frame {FRAME_WIDTH}x{FRAME_HEIGHT} | {mode}")
    print("[bridge] Ctrl+C ile çık.")

    next_t = time.monotonic()
    last_status = -1   # log throttle için: sadece var/yok değişince bas

    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += period
            # Eğer biz çok geride kaldıysak (örn. kamera takıldı), tempoyu sıfırla
            if next_t < time.monotonic() - period:
                next_t = time.monotonic() + period

            frame = capture_t.get_frame()
            if frame is None:
                continue

            detection = detect_red_ball(frame)
            packet = make_packet(detection)

            status_now = 1 if detection else 0
            if args.verbose or status_now != last_status:
                sys.stdout.write(f"[bridge] {packet.decode('ascii')}")
                sys.stdout.flush()
                last_status = status_now

            if args.dry_run:
                continue

            if ser is None or not ser.is_open:
                ser = open_serial(args.port, args.baud)
                if ser is None:
                    time.sleep(RECONNECT_DELAY)
                    continue

            try:
                ser.write(packet)
            except (serial.SerialException, OSError) as e:
                print(f"[bridge] write hata: {e} — reconnect denenecek")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None

    except KeyboardInterrupt:
        print("\n[bridge] durduruluyor...")
    finally:
        # Son söz: Teensy'e 'top yok' yolla → SAFE_STOP'a düşsün, motorlar kalmasın
        if ser is not None and ser.is_open:
            try:
                ser.write(NO_BALL_PACKET)
                ser.flush()
                ser.close()
            except Exception:
                pass
        capture_t.stop()
        capture_t.join(timeout=2)
        cam.stop()
        print("[bridge] kapandı.")


if __name__ == "__main__":
    main()
