"""
Robodog - Kamera Kalibrasyon Scripti

Kullanım:
  1. Kırmızı topu kameradan BİLİNEN bir mesafeye koy (örn. 50 cm)
  2. Bu scripti çalıştır
  3. Top tespit edilince SPACE tuşuna bas — ölçüm alınır
  4. Birkaç farklı mesafeden ölçüm al (en az 3 önerilir)
  5. 'q' ile çık — ortalama focal length hesaplanır
  6. Çıkan FOCAL_LENGTH_PX değerini .env dosyasına yaz

Formül:
  focal_length = (piksel_çap × bilinen_mesafe) / gerçek_çap
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Ayarlar ──────────────────────────────────────────────────────────────────
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
FPS          = int(os.getenv("FPS",          "30"))

# Topun gerçek çapı (cm) — doğru ölçüp yaz
BALL_DIAMETER_CM = float(os.getenv("BALL_DIAMETER_CM", "6.5"))  # placeholder

MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "300"))
MIN_CIRCULARITY  = float(os.getenv("MIN_CIRCULARITY", "0.5"))

RED_RANGES = [
    (np.array([0,   120, 70]),  np.array([10,  255, 255])),
    (np.array([170, 120, 70]),  np.array([180, 255, 255])),
]


def detect_red_ball(frame):
    """Kırmızı top tespit et, (cx, cy, radius) döndür veya None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        if area > best_area:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            best = (int(cx), int(cy), int(radius), circularity)
            best_area = area

    return best


def init_camera():
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )
    cam.configure(config)
    cam.set_controls({"FrameRate": FPS})
    cam.start()
    time.sleep(1.0)
    return cam


def main():
    cam = init_camera()
    measurements = []

    print("=" * 60)
    print("  ROBODOG — KAMERA KALİBRASYON")
    print("=" * 60)
    print(f"  Top çapı: {BALL_DIAMETER_CM} cm")
    print()
    print("  Adımlar:")
    print("    1. Topu bilinen mesafeye koy")
    print("    2. Yeşil daire topun üstünde görünene kadar bekle")
    print("    3. SPACE tuşuna bas → mesafeyi gir")
    print("    4. Farklı mesafelerden tekrarla (en az 3)")
    print("    5. 'q' ile çık → sonuç hesaplanır")
    print("=" * 60)
    print()

    try:
        while True:
            frame = cam.capture_array()
            result = detect_red_ball(frame)

            if result:
                cx, cy, radius, circ = result
                pixel_diameter = radius * 2

                cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                info = f"r={radius}px  cap={pixel_diameter}px  daire={circ:.2f}"
                cv2.putText(frame, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                hint = "SPACE: olcum al | q: bitir"
                cv2.putText(frame, hint, (10, FRAME_HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "Top bulunamadi...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Ölçüm sayısı
            cv2.putText(frame, f"Olcum: {len(measurements)}", (FRAME_WIDTH - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Kalibrasyon", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") and result:
                cx, cy, radius, circ = result
                pixel_diameter = radius * 2

                # Konsola mesafe sor
                print(f"\n  [ÖLÇÜM {len(measurements) + 1}]")
                print(f"  Piksel çap: {pixel_diameter} px")
                distance_str = input("  Topun kameraya mesafesi (cm): ").strip()

                try:
                    known_distance = float(distance_str)
                except ValueError:
                    print("  ! Geçersiz değer, atlandı.")
                    continue

                # focal_length = (piksel_çap × mesafe) / gerçek_çap
                focal = (pixel_diameter * known_distance) / BALL_DIAMETER_CM
                measurements.append({
                    "distance_cm":    known_distance,
                    "pixel_diameter": pixel_diameter,
                    "focal_length":   round(focal, 2),
                })
                print(f"  → Focal length: {focal:.2f} px")
                print(f"  → Toplam ölçüm: {len(measurements)}")

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    # ── Sonuç ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)

    if not measurements:
        print("  Hiç ölçüm alınmadı. Kalibrasyon yapılamadı.")
        return

    print("  ÖLÇÜMLER:")
    print(f"  {'#':<4} {'Mesafe (cm)':<14} {'Piksel Çap':<14} {'Focal Length':<14}")
    print("  " + "-" * 46)
    for i, m in enumerate(measurements, 1):
        print(f"  {i:<4} {m['distance_cm']:<14} {m['pixel_diameter']:<14} {m['focal_length']:<14}")

    focal_values = [m["focal_length"] for m in measurements]
    avg_focal = sum(focal_values) / len(focal_values)

    print()
    print(f"  ► Ortalama FOCAL_LENGTH_PX = {avg_focal:.2f}")
    print()
    print(f"  .env dosyasına şunu ekle:")
    print(f"    FOCAL_LENGTH_PX={avg_focal:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
