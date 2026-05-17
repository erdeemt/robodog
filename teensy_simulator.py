"""
Robodog - Teensy Simulator (PC veya Pi)

Teensy 4.1 henüz elimizde yokken, teensy_firmware/robodog_ctrl.ino firmware'ının
davranışını Python'da taklit eder. State machine + P-controller mantığı bire bir
firmware ile aynıdır (parametreler de aynı yere bakar — değiştirirsen iki yerde de
değiştir).

Üç kullanım modu:

  1) BRIDGE ile pipe (uçtan uca test, Pi üzerinde önerilen):
       python serial_bridge.py --dry-run --verbose | python teensy_simulator.py

  2) HAZIR LOG dosyasından okutma:
       python teensy_simulator.py < captured_packets.txt

  3) GÖMÜLÜ DEMO (Pi/kamera olmadan, davranışı görmek için):
       python teensy_simulator.py --demo

Çıktı her state veya motor komutu değişiminde tek satır:
  [teensy] * TRACK      drive(yaw=+0.38, fwd=+0.55)
  [teensy]   TRACK      drive(yaw=+0.10, fwd=+0.55)
  [teensy] * ARRIVED    drive(yaw=+0.00, fwd=+0.00)
  [teensy] * SAFE_STOP  drive(yaw=+0.00, fwd=+0.00)   (watchdog)
"""

import argparse
import re
import sys
import time
from typing import Optional

# ── teensy_firmware/robodog_ctrl.ino ile birebir aynı sabitler ───────────────
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

Kp_yaw         = 1.0 / (FRAME_WIDTH / 2.0)
Kp_forward     = 0.02
TARGET_CM      = 25.0
ARRIVED_CM     = 15.0
DEAD_BAND_PX   = 20.0
SEARCH_YAW     = 0.4
MAX_FWD        = 0.6
TURN_ONLY_FRAC = 0.30
WATCHDOG_S     = 0.5

# "F,1,+120,-5,52.3,28" — bridge prefix'lerini yutmak için search ile yakala
PACKET_RE = re.compile(r"F,(-?\d+),([+-]?\d+),([+-]?\d+),(-?\d+\.?\d*),(-?\d+)")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TeensySim:
    def __init__(self):
        self.state = "SAFE_STOP"
        self.last_packet_t = None   # type: Optional[float]
        self.last_motor = (0.0, 0.0)

    def on_packet(self, status: int, dx: int, dy: int, dist: float, r: int):
        self.last_packet_t = time.monotonic()

        # Top yok → tarama
        if status == 0:
            self._update("SEARCH", SEARCH_YAW, 0.0)
            return

        # Yeterince yakın → dur
        if 0.0 < dist <= ARRIVED_CM:
            self._update("ARRIVED", 0.0, 0.0)
            return

        # İzle: ortala + yaklaş
        yaw = Kp_yaw * dx if abs(dx) > DEAD_BAND_PX else 0.0

        fwd = 0.0
        if dist > 0.0:
            fwd = Kp_forward * (dist - TARGET_CM)
            fwd = clamp(fwd, 0.0, MAX_FWD)

        # Top çok sapmışsa önce dön, ileri gitme
        if abs(dx) > FRAME_WIDTH * TURN_ONLY_FRAC:
            fwd = 0.0

        self._update("TRACK", yaw, fwd)

    def check_watchdog(self) -> bool:
        if self.last_packet_t is None:
            return False
        if time.monotonic() - self.last_packet_t > WATCHDOG_S:
            if self.state != "SAFE_STOP":
                self._update("SAFE_STOP", 0.0, 0.0, note="watchdog")
                return True
        return False

    def _update(self, state: str, yaw: float, fwd: float, note: str = ""):
        yaw_q = round(clamp(yaw, -1.0, 1.0), 2)
        fwd_q = round(clamp(fwd, -1.0, 1.0), 2)
        new_motor = (yaw_q, fwd_q)

        state_changed = state != self.state
        motor_changed = new_motor != self.last_motor

        if state_changed or motor_changed:
            tag = "*" if state_changed else " "
            note_s = "   ({})".format(note) if note else ""
            print("[teensy] {} {:9s}  drive(yaw={:+.2f}, fwd={:+.2f}){}".format(
                tag, state, yaw_q, fwd_q, note_s))

        self.state = state
        self.last_motor = new_motor


# ─────────────────────────────────────────────────────────────────────────────
#  Pipe / stdin modu
# ─────────────────────────────────────────────────────────────────────────────
def run_stdin(sim: TeensySim):
    print("[teensy-sim] stdin'den paket bekleniyor. Ctrl+C / Ctrl+D ile çık.")
    try:
        for raw in sys.stdin:
            m = PACKET_RE.search(raw)
            if not m:
                continue
            status   = int(m.group(1))
            dx       = int(m.group(2))
            dy       = int(m.group(3))
            dist     = float(m.group(4))
            radius   = int(m.group(5))
            sim.on_packet(status, dx, dy, dist, radius)
            sim.check_watchdog()
    except KeyboardInterrupt:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Demo modu — donanım olmadan davranışı göster
# ─────────────────────────────────────────────────────────────────────────────
DEMO_SCRIPT = [
    # (label, status, dx, dy, dist_cm, radius_px, wait_s)
    ("--- Başlangıç: top yok (Pi tarama yolluyor)",   0, 0,    0,  0.0,   0, 0.3),
    ("",                                              0, 0,    0,  0.0,   0, 0.3),

    ("--- Top sağda göründü (uzakta, sağa dönüş)",    1, +220, -10, 95.0, 12, 0.3),
    ("",                                              1, +180,  -8, 88.0, 14, 0.3),
    ("",                                              1, +130,  -5, 80.0, 15, 0.3),
    ("",                                              1,  +70,  -2, 70.0, 17, 0.3),

    ("--- Topu ortaladık, yaklaşmaya başla",          1,  +10,  -1, 60.0, 20, 0.3),
    ("",                                              1,   -5,   0, 45.0, 27, 0.3),
    ("",                                              1,   +8,   0, 30.0, 40, 0.3),

    ("--- Top yakın, ARRIVED",                        1,   -3,   0, 14.0, 86, 0.3),

    ("--- Top yine uzaklaştı (örn. itildi), TRACK",   1,  +25,   0, 35.0, 35, 0.3),

    ("--- Pi sustu — watchdog tetiklensin (bekle)",   None, 0,  0,  0.0,   0, 1.0),
]


def run_demo(sim: TeensySim):
    print("[teensy-sim] DEMO modu. Firmware davranışı simüle ediliyor.\n")
    for label, status, dx, dy, dist, r, wait in DEMO_SCRIPT:
        if label:
            print(f"\n{label}")
        if status is not None:
            sim.on_packet(status, dx, dy, dist, r)
        time.sleep(wait)
        sim.check_watchdog()

    # Watchdog'un tetiklendiğinden emin olmak için son bir tick daha
    time.sleep(0.6)
    sim.check_watchdog()
    print("\n[teensy-sim] demo bitti.")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true",
                        help="kamera/bridge olmadan gömülü senaryoyu çalıştır")
    args = parser.parse_args()

    sim = TeensySim()
    if args.demo:
        run_demo(sim)
    else:
        run_stdin(sim)


if __name__ == "__main__":
    main()
