"""
Robodog - Red Ball Detector (Multithreaded)
Raspberry Pi 4 + Camera Module 2

Mimari (3 thread):
  - CaptureThread : kameradan frame çeker, en güncel frame'i paylaşır
  - ProcessThread : frame'i alır → tespit + mesafe + yön hesaplar
  - Main thread   : sonuçları konsola basar + opsiyonel ekran gösterimi

Konsol çıktısı:
  YÖN: 1 = sol, 2 = orta, 3 = sağ
  Mesafe: cm cinsinden tahmini mesafe
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import threading
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Kamera Ayarları ──────────────────────────────────────────────────────────
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
FPS          = int(os.getenv("FPS",          "30"))
DISPLAY      = os.getenv("DISPLAY_OUTPUT", "true").lower() == "true"

# ── Top Parametreleri ────────────────────────────────────────────────────────
BALL_DIAMETER_CM = float(os.getenv("BALL_DIAMETER_CM", "6.5"))   # placeholder — ölçüp güncelle
FOCAL_LENGTH_PX  = float(os.getenv("FOCAL_LENGTH_PX",  "500.0")) # placeholder — calibrate_camera.py ile bul

# ── Tespit Parametreleri ─────────────────────────────────────────────────────
MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "300"))
MIN_CIRCULARITY  = float(os.getenv("MIN_CIRCULARITY", "0.6"))

# Kırmızı HSV aralıkları (wrap-around)
RED_RANGES = [
    (np.array([0,   120, 70]),  np.array([10,  255, 255])),
    (np.array([170, 120, 70]),  np.array([180, 255, 255])),
]

# Yön bölgeleri — frame genişliğinin %30'u orta bölge
CENTER_BAND_RATIO = 0.3
LEFT_BOUND  = (FRAME_WIDTH / 2) - (FRAME_WIDTH * CENTER_BAND_RATIO / 2)
RIGHT_BOUND = (FRAME_WIDTH / 2) + (FRAME_WIDTH * CENTER_BAND_RATIO / 2)

# Morphology kernel — bir kere oluştur, her frame'de tekrar yaratma
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ═════════════════════════════════════════════════════════════════════════════
#  Tespit Fonksiyonları
# ═════════════════════════════════════════════════════════════════════════════

def detect_red_ball(frame: np.ndarray) -> dict | None:
    """
    Framede kırmızı top arar.
    Returns dict {"center", "radius_px", "area", "circularity"} veya None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, RED_RANGES[0][0], RED_RANGES[0][1])
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, RED_RANGES[1][0], RED_RANGES[1][1]))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None

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

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        candidate = {
            "center":      (int(cx), int(cy)),
            "radius_px":   int(radius),
            "area":        area,
            "circularity": circularity,
        }

        if best is None or area > best["area"]:
            best = candidate

    return best


def estimate_distance(radius_px: int) -> float:
    """piksel yarıçapı → mesafe (cm)"""
    if radius_px <= 0:
        return -1.0
    return round((BALL_DIAMETER_CM * FOCAL_LENGTH_PX) / (radius_px * 2), 1)


def get_direction(cx: int) -> int:
    """1 = sol, 2 = orta, 3 = sağ"""
    if cx < LEFT_BOUND:
        return 1
    if cx > RIGHT_BOUND:
        return 3
    return 2


# ═════════════════════════════════════════════════════════════════════════════
#  Thread'ler
# ═════════════════════════════════════════════════════════════════════════════

class CaptureThread(threading.Thread):
    """Kameradan sürekli frame çeker, en güncel frame'i tutar."""

    def __init__(self, camera: Picamera2):
        super().__init__(daemon=True)
        self.camera = camera
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            buf = self.camera.capture_array()
            with self.lock:
                self.frame = buf

    def get_frame(self) -> np.ndarray | None:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


class ProcessThread(threading.Thread):
    """Frame'i alır, tespit + mesafe + yön hesaplar."""

    def __init__(self, capture: CaptureThread):
        super().__init__(daemon=True)
        self.capture = capture
        self.lock = threading.Lock()
        self.running = True

        # Son sonuç
        self.result: dict | None = None   # {"detection", "distance", "direction", "fps"}
        self.display_frame: np.ndarray | None = None

        self._frame_count = 0
        self._fps_time = time.monotonic()
        self._fps = 0.0

    def run(self):
        while self.running:
            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            detection = detect_red_ball(frame)

            # FPS hesapla
            self._frame_count += 1
            now = time.monotonic()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_time = now

            if detection:
                distance  = estimate_distance(detection["radius_px"])
                direction = get_direction(detection["center"][0])

                result = {
                    "detection": detection,
                    "distance":  distance,
                    "direction": direction,
                    "fps":       self._fps,
                }

                display = self._draw_overlay(frame, detection, distance, direction) if DISPLAY else None
            else:
                result = None
                display = frame if DISPLAY else None

            with self.lock:
                self.result = result
                self.display_frame = display

    def get_result(self) -> tuple:
        """(result_dict | None, display_frame | None, fps)"""
        with self.lock:
            return self.result, self.display_frame, self._fps

    def stop(self):
        self.running = False

    @staticmethod
    def _draw_overlay(frame, detection, distance, direction):
        out = frame.copy()
        cx, cy = detection["center"]
        r = detection["radius_px"]

        cv2.circle(out, (cx, cy), r, (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)

        # Yön bölge çizgileri
        lb, rb = int(LEFT_BOUND), int(RIGHT_BOUND)
        cv2.line(out, (lb, 0), (lb, FRAME_HEIGHT), (100, 100, 100), 1)
        cv2.line(out, (rb, 0), (rb, FRAME_HEIGHT), (100, 100, 100), 1)

        dir_labels = {1: "SOL", 2: "ORTA", 3: "SAG"}
        info = f"Mesafe: {distance:.1f}cm | Yon: {dir_labels[direction]} ({direction})"
        cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        circ_text = f"daire={detection['circularity']:.2f} | r={r}px"
        cv2.putText(out, circ_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return out


# ═════════════════════════════════════════════════════════════════════════════
#  Kamera
# ═════════════════════════════════════════════════════════════════════════════

def init_camera() -> Picamera2:
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )
    cam.configure(config)
    cam.set_controls({"FrameRate": FPS})
    cam.start()
    time.sleep(1.0)
    return cam


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cam = init_camera()

    capture_t = CaptureThread(cam)
    process_t = ProcessThread(capture_t)

    capture_t.start()
    process_t.start()

    print(f"[INFO] Kırmızı top tespiti başladı — {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
    print(f"[INFO] Top çapı: {BALL_DIAMETER_CM} cm | Focal length: {FOCAL_LENGTH_PX} px")
    print(f"[INFO] Thread'ler: capture + process + main")
    print("[INFO] Çıkmak için 'q' tuşuna basın.\n")

    try:
        while True:
            result, display_frame, fps = process_t.get_result()

            if result:
                d = result["detection"]
                print(f"  YÖN: {result['direction']}  |  "
                      f"MESAFE: {result['distance']:.1f} cm  |  "
                      f"merkez=({d['center'][0]}, {d['center'][1]})  "
                      f"r={d['radius_px']}px  "
                      f"daire={d['circularity']:.2f}  "
                      f"[{fps:.1f} fps]")
            else:
                print(f"  --- top bulunamadı --- [{fps:.1f} fps]", end="\r")

            if DISPLAY and display_frame is not None:
                # FPS göster
                cv2.putText(display_frame, f"{fps:.1f} fps",
                            (FRAME_WIDTH - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Robodog - Red Ball Detector", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Headless modda CPU boşa spin yapmasın
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Durduruldu.")
    finally:
        process_t.stop()
        capture_t.stop()
        process_t.join(timeout=2)
        capture_t.join(timeout=2)
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
