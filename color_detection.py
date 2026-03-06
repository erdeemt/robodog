"""
Robodog - Color Detection Module
Raspberry Pi 4 + Camera Module 2
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Konfigürasyon ────────────────────────────────────────────────────────────
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
FPS          = int(os.getenv("FPS",          "30"))
DISPLAY      = os.getenv("DISPLAY_OUTPUT", "true").lower() == "true"
SAVE_FRAMES  = os.getenv("SAVE_FRAMES", "false").lower() == "true"
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "output")

# ── Renk aralıkları (HSV) ────────────────────────────────────────────────────
# Her renk için (lower_hsv, upper_hsv, bgr_draw_color) tuple'ı
COLOR_RANGES = {
    "kirmizi": [
        (np.array([0,   120,  70]), np.array([10,  255, 255]), (0,   0,   255)),
        (np.array([170, 120,  70]), np.array([180, 255, 255]), (0,   0,   255)),
    ],
    "yesil": [
        (np.array([36,  100,  70]), np.array([86,  255, 255]), (0,   255,   0)),
    ],
    "mavi": [
        (np.array([94,  80,   70]), np.array([126, 255, 255]), (255,   0,   0)),
    ],
    "sari": [
        (np.array([20,  100,  100]), np.array([35,  255, 255]), (0,   255, 255)),
    ],
    "turuncu": [
        (np.array([10,  100,  100]), np.array([20,  255, 255]), (0,   165, 255)),
    ],
    "mor": [
        (np.array([129, 50,   70]), np.array([158, 255, 255]), (255,   0, 255)),
    ],
}

MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "500"))


def detect_colors(frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Verilen karedeki renk bölgelerini tespit eder.

    Returns:
        annotated_frame : bounding box ve etiket çizilmiş kare
        detections      : tespit listesi [{"color", "bbox", "area", "center"}]
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    annotated = frame.copy()
    detections = []

    for color_name, ranges in COLOR_RANGES.items():
        # Birden fazla alt-üst aralığı OR ile birleştir (kırmızı gibi)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        draw_color = (255, 255, 255)
        for lower, upper, bgr in ranges:
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, lower, upper))
            draw_color = bgr

        # Gürültü temizleme
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            detections.append({
                "color":  color_name,
                "bbox":   (x, y, w, h),
                "area":   area,
                "center": (cx, cy),
            })

            # Çizim
            cv2.rectangle(annotated, (x, y), (x + w, y + h), draw_color, 2)
            cv2.circle(annotated, (cx, cy), 5, draw_color, -1)
            label = f"{color_name} ({area:.0f}px)"
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)

    return annotated, detections


def build_color_stats(frame: np.ndarray) -> dict:
    """Her renk için karedeki piksel yüzdesini döndürür."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    total_pixels = frame.shape[0] * frame.shape[1]
    stats = {}

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper, _ in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        stats[color_name] = round(cv2.countNonZero(mask) / total_pixels * 100, 2)

    return stats


def init_camera() -> Picamera2:
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )
    cam.configure(config)
    cam.set_controls({"FrameRate": FPS})
    cam.start()
    time.sleep(1.0)  # kameranın ısınması için bekle
    return cam


def main():
    if SAVE_FRAMES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    cam = init_camera()
    frame_idx = 0

    print(f"[INFO] Renk tespiti başladı — {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
    print("[INFO] Çıkmak için 'q' tuşuna basın.")

    try:
        while True:
            frame = cam.capture_array()

            annotated, detections = detect_colors(frame)
            stats = build_color_stats(frame)

            # Konsol çıktısı
            if detections:
                for d in detections:
                    print(f"  [{d['color']:10s}] merkez={d['center']}  alan={d['area']:.0f}px")

            # Ekran çıktısı
            if DISPLAY:
                cv2.imshow("Robodog - Color Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Kare kaydetme
            if SAVE_FRAMES:
                path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(path, annotated)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stoppped.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
