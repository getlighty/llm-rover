"""imu.py — IMU polling thread + state container for the rover.

Reads IMU data from the ESP32's continuous T:1001 feedback stream
(QMI8658 accel+gyro, AK09918 magnetometer).  Provides:
  - Stuck detection via encoder speeds (is_stationary)
  - AHRS-fused heading from Madgwick filter (heading_deg)
  - Roll/pitch from AHRS quaternion
  - Motor-state-aware magnetometer gating (mag ignored while driving)

The AHRS filter runs continuously (always-on polling) to prevent
gyro drift and maintain accurate heading.
"""

import math
import threading
import time

from ahrs_filter import MadgwickAHRS, MagCalibration

# ── Tunable thresholds ────────────────────────────────────────────────
# Accel is gravity-subtracted: ~0.2 at rest, higher when moving (vibration).
ACCEL_MOVING_THRESHOLD = 0.35  # accel magnitude above this = vibration/motion
GYRO_MOVING_THRESHOLD = 1.0    # °/s angular rate above this = rotating
STATIONARY_READINGS = 3        # consecutive still reads to confirm stuck (~300ms)
ENCODER_MOVING_THRESHOLD = 0.02  # encoder speed (m/s) above this = moving
TILT_ACCEL_WARN = 20.0         # accel magnitude spike suggesting large tilt/impact
                               # (resting gravity reads ~18.2 in raw units)
TILT_ACCEL_STOP = 22.0         # severe accel spike — emergency stop
POLL_INTERVAL = 0.10           # 10 Hz during wheel activity
DATA_MAX_AGE = 2.0             # seconds before data considered stale


class IMUState:
    """Thread-safe container for latest T:1001 IMU readings."""

    def __init__(self):
        self._lock = threading.Lock()
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.gz = 0.0
        self.mx = 0.0
        self.my = 0.0
        self.mz = 0.0
        self.enc_l = 0.0
        self.enc_r = 0.0
        self.voltage = 0.0
        self.timestamp = 0.0  # time.time() of last successful read

    def update(self, data):
        """Update from a parsed T:1001 feedback dict."""
        with self._lock:
            self.ax = float(data.get("ax", 0))
            self.ay = float(data.get("ay", 0))
            self.az = float(data.get("az", 0))
            self.gx = float(data.get("gx", 0))
            self.gy = float(data.get("gy", 0))
            self.gz = float(data.get("gz", 0))
            self.mx = float(data.get("mx", 0))
            self.my = float(data.get("my", 0))
            self.mz = float(data.get("mz", 0))
            self.enc_l = float(data.get("L", 0))
            self.enc_r = float(data.get("R", 0))
            self.voltage = float(data.get("v", 0))
            self.timestamp = time.time()

    @property
    def age(self):
        """Seconds since last successful read."""
        ts = self.timestamp
        if ts == 0:
            return float("inf")
        return time.time() - ts

    @property
    def fresh(self):
        """True if data is recent enough to trust."""
        return self.age < DATA_MAX_AGE

    @property
    def accel_magnitude(self):
        """Acceleration magnitude (gravity-subtracted, ~0.2 at rest)."""
        with self._lock:
            return math.sqrt(self.ax**2 + self.ay**2 + self.az**2)

    @property
    def gyro_magnitude(self):
        """Total angular rate (0 at rest)."""
        with self._lock:
            return math.sqrt(self.gx**2 + self.gy**2 + self.gz**2)

    @property
    def encoder_speed(self):
        """Average absolute encoder speed (m/s). 0 = stationary."""
        with self._lock:
            return (abs(self.enc_l) + abs(self.enc_r)) / 2.0

    @property
    def mag_heading_raw(self):
        """Raw magnetometer heading in degrees (0-360), uncorrected."""
        with self._lock:
            mx, my = self.mx, self.my
        return math.degrees(math.atan2(my, mx)) % 360


class IMUPoller:
    """Daemon thread that continuously reads T:1001 stream for AHRS fusion.

    Always-on polling — AHRS needs continuous updates for gyro drift
    prevention and magnetometer correction when stopped.

    Args:
        ser: Serial instance with read_imu() method.
        log_fn: Optional log_event(category, message) callable.
        cal_file: Path to mag_cal.json (optional).
    """

    def __init__(self, ser, log_fn=None, cal_file=None):
        self.ser = ser
        self.log_fn = log_fn or (lambda cat, msg: None)
        self.state = IMUState()
        self.wheels_active = threading.Event()
        self._stationary_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="imu-poller")

        # AHRS filter
        mag_cal = None
        if cal_file:
            mag_cal = MagCalibration(cal_file)
            if mag_cal.load():
                self.log_fn("imu", f"Loaded mag cal: offset={mag_cal.offset}")
            else:
                self.log_fn("imu", "No mag calibration file found")
                mag_cal = None
        self.ahrs = MadgwickAHRS(mag_cal=mag_cal)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ── Public properties ─────────────────────────────────────────────

    @property
    def heading_deg(self):
        """AHRS heading in degrees (0-360), offset-corrected so boot=0."""
        return self.ahrs.heading

    @property
    def roll(self):
        """Roll angle in degrees from AHRS."""
        return self.ahrs.euler[0]

    @property
    def pitch(self):
        """Pitch angle in degrees from AHRS."""
        return self.ahrs.euler[1]

    @property
    def is_stationary(self):
        """True when wheels are on but IMU shows no motion for N readings."""
        return self._stationary_count >= STATIONARY_READINGS

    def reset_stationary(self):
        """Reset stuck counter — call when new wheel command is sent."""
        self._stationary_count = 0

    def check_tilt(self):
        """Check for accel magnitude spikes (proxy for tilt/impact).

        Returns:
            ("ok", mag), ("warn", mag), or ("stop", mag)
        """
        if not self.state.fresh:
            return ("ok", 0.0)
        mag = self.state.accel_magnitude
        if mag >= TILT_ACCEL_STOP:
            return ("stop", mag)
        if mag >= TILT_ACCEL_WARN:
            return ("warn", mag)
        return ("ok", mag)

    def set_heading_offset(self):
        """Capture current AHRS yaw as the zero reference."""
        self.ahrs.set_heading_offset()
        self.log_fn("imu", f"AHRS heading offset set (raw yaw={self.ahrs.euler[2]:.1f}°)")

    def read_once(self):
        """Do a single IMU read (blocking). Returns True on success."""
        data = self.ser.read_imu()
        if data:
            self.state.update(data)
            return True
        return False

    @staticmethod
    def _cardinal(deg):
        """Map heading degrees to cardinal direction relative to boot heading."""
        dirs = [
            (0, "N(start)"), (45, "NE"), (90, "E(right)"), (135, "SE"),
            (180, "S(opposite)"), (225, "SW"), (270, "W(left)"), (315, "NW"),
        ]
        # Find closest cardinal
        best = min(dirs, key=lambda d: min(abs(deg - d[0]), 360 - abs(deg - d[0])))
        return best[1]

    def get_prompt_line(self, stuck_headings=None):
        """Return a one-line string for the LLM prompt, or '' if stale.
        Args:
            stuck_headings: optional list of headings tried during stuck recovery.
        """
        if not self.state.fresh:
            return ""
        hdg = self.heading_deg
        cardinal = self._cardinal(hdg)
        line = f"IMU: heading={hdg:.0f}°, voltage={self.state.voltage:.1f}V ({cardinal})"
        # Include tilt if significant
        r, p = self.roll, self.pitch
        if abs(r) > 5 or abs(p) > 5:
            line += f" tilt=({r:+.0f}°r,{p:+.0f}°p)"
        if stuck_headings:
            line += f" [tried: {stuck_headings}]"
        return line

    def get_map_data(self):
        """Return dict for web UI map state, or None if stale."""
        if not self.state.fresh:
            return None
        return {
            "heading": round(self.heading_deg, 1),
            "roll": round(self.roll, 1),
            "pitch": round(self.pitch, 1),
            "mag_weight": round(self.ahrs.mag_weight, 2),
            "accel": round(self.state.accel_magnitude, 3),
            "gyro": round(self.state.gyro_magnitude, 2),
            "voltage": round(self.state.voltage, 2),
            "mag_heading_raw": round(self.state.mag_heading_raw, 1),
        }

    # ── Polling thread ────────────────────────────────────────────────

    def _run(self):
        self.log_fn("imu", "IMU poller started (always-on for AHRS)")
        while not self._stop.is_set():
            data = self.ser.read_imu()
            if data:
                self.state.update(data)

                # Feed AHRS filter
                s = self.state
                self.ahrs.update(
                    s.gx, s.gy, s.gz,
                    s.ax, s.ay, s.az,
                    s.mx, s.my, s.mz,
                    s.encoder_speed,
                )

                # Stuck detection (encoder-based)
                if self.wheels_active.is_set():
                    if s.encoder_speed < ENCODER_MOVING_THRESHOLD:
                        self._stationary_count += 1
                    else:
                        self._stationary_count = 0

            self._stop.wait(POLL_INTERVAL)

        self.log_fn("imu", "IMU poller stopped")
