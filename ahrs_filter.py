"""ahrs_filter.py — Madgwick AHRS filter with motor-aware magnetometer gating.

Quaternion-based 9-axis sensor fusion for the rover's QMI8658 (accel+gyro)
and AK09918 (magnetometer).  Key feature: magnetometer is gated out when
motors are running (motor magnets corrupt readings) and ramped back in
gradually after motors stop.

Units expected:
  - Gyro: deg/s (converted internally to rad/s)
  - Accel: mg or m/s² (normalized, so units don't matter)
  - Mag: raw counts or µT (normalized internally)
  - Encoder speed: m/s (average of |L|+|R|/2)
"""

import json
import math
import os
import time

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class MadgwickAHRS:
    """Madgwick quaternion-based AHRS with motor-state magnetometer gating.

    When motors are running, operates in 6-axis mode (gyro + accel only).
    After motors stop, magnetometer weight ramps from 0 to 1 over ~2 seconds
    to allow motor-induced magnetic disturbance to settle.
    """

    def __init__(self, beta=0.033, mag_tau=0.5, mag_cal=None):
        """
        Args:
            beta: Madgwick filter gain (higher = faster convergence, more noise)
            mag_tau: Time constant for mag weight ramp-up after motors stop (seconds)
            mag_cal: MagCalibration instance (optional, loaded separately)
        """
        self.beta = beta
        self.mag_tau = mag_tau
        self.mag_cal = mag_cal

        # Quaternion [w, x, y, z]
        self.q = [1.0, 0.0, 0.0, 0.0]

        # Timing
        self._last_time = None

        # Magnetometer gating
        self._motors_stopped_at = None  # time when motors last stopped
        self._motors_running = False
        self.mag_weight = 0.0  # 0 = mag ignored, 1 = fully trusted

        # Gyro bias estimation (slow EMA when stationary)
        self._gyro_bias = [0.0, 0.0, 0.0]
        self._bias_alpha = 0.005  # EMA smoothing factor (slow)

        # Heading offset (set at boot so forward = 0)
        self._heading_offset = 0.0

    def update(self, gx, gy, gz, ax, ay, az, mx, my, mz, enc_speed=0.0):
        """Process one sensor sample.

        Args:
            gx, gy, gz: Gyroscope in deg/s
            ax, ay, az: Accelerometer (any units, will be normalized)
            mx, my, mz: Magnetometer (any units, will be normalized)
            enc_speed: Average encoder speed in m/s (0 = stopped)
        """
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            # On first update, try to initialize from accel
            self._init_from_accel(ax, ay, az)
            return
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0 or dt > 1.0:
            return  # skip bogus dt

        # Motor state tracking
        motor_threshold = 0.02  # m/s
        if enc_speed > motor_threshold:
            self._motors_running = True
            self._motors_stopped_at = None
            self.mag_weight = 0.0
        else:
            if self._motors_running:
                # Motors just stopped
                self._motors_stopped_at = now
                self._motors_running = False
            # Ramp mag weight
            if self._motors_stopped_at is not None:
                elapsed = now - self._motors_stopped_at
                self.mag_weight = 1.0 - math.exp(-elapsed / self.mag_tau)
            else:
                self.mag_weight = 1.0  # never ran motors

        # Convert gyro to rad/s and subtract bias
        gx_rad = (gx - self._gyro_bias[0]) * DEG2RAD
        gy_rad = (gy - self._gyro_bias[1]) * DEG2RAD
        gz_rad = (gz - self._gyro_bias[2]) * DEG2RAD

        # Gyro bias estimation when stationary
        if enc_speed < motor_threshold:
            gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
            if gyro_mag < 5.0:  # deg/s — reasonably still
                a = self._bias_alpha
                self._gyro_bias[0] += a * (gx - self._gyro_bias[0])
                self._gyro_bias[1] += a * (gy - self._gyro_bias[1])
                self._gyro_bias[2] += a * (gz - self._gyro_bias[2])

        # Apply mag calibration if available
        if self.mag_cal:
            mx, my, mz = self.mag_cal.apply(mx, my, mz)

        # Run filter
        if self.mag_weight > 0.05:
            self._update_9dof(gx_rad, gy_rad, gz_rad, ax, ay, az,
                              mx, my, mz, dt)
        else:
            self._update_6dof(gx_rad, gy_rad, gz_rad, ax, ay, az, dt)

    def _init_from_accel(self, ax, ay, az):
        """Initialize quaternion from accelerometer (gravity direction)."""
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 0.01:
            return  # no usable accel data
        ax /= norm
        ay /= norm
        az /= norm
        # Compute pitch and roll from gravity
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        roll = math.atan2(ay, az)
        # Convert to quaternion (yaw = 0)
        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        self.q = [cr * cp, sr * cp, cr * sp, -sr * sp]

    def _update_6dof(self, gx, gy, gz, ax, ay, az, dt):
        """Madgwick 6-axis update (gyro + accel, no magnetometer)."""
        q0, q1, q2, q3 = self.q

        # Normalize accelerometer
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 0.01:
            # No valid accel — pure gyro integration
            self._gyro_integrate(gx, gy, gz, dt)
            return
        ax /= norm
        ay /= norm
        az /= norm

        # Gradient descent corrective step (accel only)
        f1 = 2.0 * (q1 * q3 - q0 * q2) - ax
        f2 = 2.0 * (q0 * q1 + q2 * q3) - ay
        f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - az

        j_t_f = [
            -2.0 * q2 * f1 + 2.0 * q1 * f2,
            2.0 * q3 * f1 + 2.0 * q0 * f2 - 4.0 * q1 * f3,
            -2.0 * q0 * f1 + 2.0 * q3 * f2 - 4.0 * q2 * f3,
            2.0 * q1 * f1 + 2.0 * q2 * f2,
        ]

        norm = math.sqrt(sum(x * x for x in j_t_f))
        if norm > 0:
            j_t_f = [x / norm for x in j_t_f]

        # Apply feedback
        beta = self.beta
        q_dot = [
            0.5 * (-q1 * gx - q2 * gy - q3 * gz) - beta * j_t_f[0],
            0.5 * (q0 * gx + q2 * gz - q3 * gy) - beta * j_t_f[1],
            0.5 * (q0 * gy - q1 * gz + q3 * gx) - beta * j_t_f[2],
            0.5 * (q0 * gz + q1 * gy - q2 * gx) - beta * j_t_f[3],
        ]

        # Integrate
        self.q = [q0 + q_dot[0] * dt, q1 + q_dot[1] * dt,
                  q2 + q_dot[2] * dt, q3 + q_dot[3] * dt]
        self._normalize_q()

    def _update_9dof(self, gx, gy, gz, ax, ay, az, mx, my, mz, dt):
        """Madgwick 9-axis update with weighted magnetometer contribution."""
        q0, q1, q2, q3 = self.q

        # Normalize accel
        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if a_norm < 0.01:
            self._gyro_integrate(gx, gy, gz, dt)
            return
        ax /= a_norm
        ay /= a_norm
        az /= a_norm

        # Normalize mag
        m_norm = math.sqrt(mx * mx + my * my + mz * mz)
        if m_norm < 0.01:
            # Bad mag data — fall back to 6-axis
            self._update_6dof(gx, gy, gz, ax * a_norm, ay * a_norm,
                              az * a_norm, dt)
            return
        mx /= m_norm
        my /= m_norm
        mz /= m_norm

        # Reference direction of Earth's magnetic field
        _2q0mx = 2.0 * q0 * mx
        _2q0my = 2.0 * q0 * my
        _2q0mz = 2.0 * q0 * mz
        _2q1mx = 2.0 * q1 * mx

        hx = (mx * (0.5 - q2 * q2 - q3 * q3) + my * (q1 * q2 - q0 * q3) +
              mz * (q1 * q3 + q0 * q2))
        hy = (mx * (q1 * q2 + q0 * q3) + my * (0.5 - q1 * q1 - q3 * q3) +
              mz * (q2 * q3 - q0 * q1))

        _2bx = 2.0 * math.sqrt(hx * hx + hy * hy)
        _2bz = 2.0 * (mx * (q1 * q3 - q0 * q2) + my * (q2 * q3 + q0 * q1) +
                       mz * (0.5 - q1 * q1 - q2 * q2))

        # Gradient descent: accel objective
        f1 = 2.0 * (q1 * q3 - q0 * q2) - ax
        f2 = 2.0 * (q0 * q1 + q2 * q3) - ay
        f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - az

        # Gradient descent: mag objective
        f4 = (_2bx * (0.5 - q2 * q2 - q3 * q3) +
              _2bz * (q1 * q3 - q0 * q2) - mx)
        f5 = (_2bx * (q1 * q2 - q0 * q3) +
              _2bz * (q0 * q1 + q2 * q3) - my)
        f6 = (_2bx * (q0 * q2 + q1 * q3) +
              _2bz * (0.5 - q1 * q1 - q2 * q2) - mz)

        # Jacobian * f (accel part)
        j_t_f = [
            -2.0 * q2 * f1 + 2.0 * q1 * f2,
            2.0 * q3 * f1 + 2.0 * q0 * f2 - 4.0 * q1 * f3,
            -2.0 * q0 * f1 + 2.0 * q3 * f2 - 4.0 * q2 * f3,
            2.0 * q1 * f1 + 2.0 * q2 * f2,
        ]

        # Mag Jacobian contribution (weighted by mag_weight)
        w = self.mag_weight
        j_t_f[0] += w * (-_2bz * q2 * f4 + (-_2bx * q3 + _2bz * q1) * f5 + _2bx * q2 * f6)
        j_t_f[1] += w * (_2bz * q3 * f4 + (_2bx * q2 + _2bz * q0) * f5 + (-4.0 * _2bx * q1 + _2bz * q3 - _2bz * q1) * f6)  # noqa
        j_t_f[2] += w * (-4.0 * _2bx * q2 * f4 + (_2bx * q1 + _2bz * q3) * f5 + (_2bx * q0 - 4.0 * _2bz * q2) * f6)  # noqa  # noqa
        j_t_f[3] += w * (-4.0 * _2bx * q3 * f4 + (-_2bx * q0 + _2bz * q1) * f5 + _2bx * q1 * f6)  # noqa

        norm = math.sqrt(sum(x * x for x in j_t_f))
        if norm > 0:
            j_t_f = [x / norm for x in j_t_f]

        # Apply feedback
        beta = self.beta
        q_dot = [
            0.5 * (-q1 * gx - q2 * gy - q3 * gz) - beta * j_t_f[0],
            0.5 * (q0 * gx + q2 * gz - q3 * gy) - beta * j_t_f[1],
            0.5 * (q0 * gy - q1 * gz + q3 * gx) - beta * j_t_f[2],
            0.5 * (q0 * gz + q1 * gy - q2 * gx) - beta * j_t_f[3],
        ]

        self.q = [q0 + q_dot[0] * dt, q1 + q_dot[1] * dt,
                  q2 + q_dot[2] * dt, q3 + q_dot[3] * dt]
        self._normalize_q()

    def _gyro_integrate(self, gx, gy, gz, dt):
        """Pure gyro integration (no correction)."""
        q0, q1, q2, q3 = self.q
        q0 += 0.5 * (-q1 * gx - q2 * gy - q3 * gz) * dt
        q1 += 0.5 * (q0 * gx + q2 * gz - q3 * gy) * dt
        q2 += 0.5 * (q0 * gy - q1 * gz + q3 * gx) * dt
        q3 += 0.5 * (q0 * gz + q1 * gy - q2 * gx) * dt
        self.q = [q0, q1, q2, q3]
        self._normalize_q()

    def _normalize_q(self):
        norm = math.sqrt(sum(x * x for x in self.q))
        if norm > 0:
            self.q = [x / norm for x in self.q]

    @property
    def euler(self):
        """Return (roll, pitch, yaw) in degrees."""
        q0, q1, q2, q3 = self.q
        # Roll (x-axis)
        sinr = 2.0 * (q0 * q1 + q2 * q3)
        cosr = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr, cosr) * RAD2DEG
        # Pitch (y-axis)
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp) * RAD2DEG
        # Yaw (z-axis)
        siny = 2.0 * (q0 * q3 + q1 * q2)
        cosy = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw = math.atan2(siny, cosy) * RAD2DEG
        return (roll, pitch, yaw)

    @property
    def heading(self):
        """Heading in degrees 0-360, offset-corrected so boot = 0."""
        _, _, yaw = self.euler
        return (yaw - self._heading_offset) % 360

    def set_heading_offset(self):
        """Capture current yaw as zero reference."""
        _, _, yaw = self.euler
        self._heading_offset = yaw

    @property
    def gyro_bias(self):
        """Current gyro bias estimate (deg/s)."""
        return tuple(self._gyro_bias)


class MagCalibration:
    """Hard-iron + soft-iron magnetometer calibration.

    Stores offsets and scale matrix, applies correction to raw readings.
    Calibration data saved/loaded from JSON file.
    """

    def __init__(self, cal_file="mag_cal.json"):
        self.cal_file = cal_file
        # Hard-iron offsets (subtracted from raw)
        self.offset = [0.0, 0.0, 0.0]
        # Soft-iron correction matrix (3x3, identity = no correction)
        self.scale = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
        self.valid = False

    def load(self):
        """Load calibration from JSON file. Returns True if loaded."""
        if not os.path.exists(self.cal_file):
            return False
        try:
            with open(self.cal_file) as f:
                data = json.load(f)
            self.offset = data["offset"]
            self.scale = data.get("scale", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.valid = True
            return True
        except (json.JSONDecodeError, KeyError, IOError):
            return False

    def save(self):
        """Save calibration to JSON file."""
        data = {"offset": self.offset, "scale": self.scale}
        with open(self.cal_file, "w") as f:
            json.dump(data, f, indent=2)

    def apply(self, mx, my, mz):
        """Apply hard-iron and soft-iron correction.

        Returns corrected (mx, my, mz).
        """
        # Subtract hard-iron offset
        x = mx - self.offset[0]
        y = my - self.offset[1]
        z = mz - self.offset[2]
        # Apply soft-iron scale matrix
        s = self.scale
        cx = s[0][0] * x + s[0][1] * y + s[0][2] * z
        cy = s[1][0] * x + s[1][1] * y + s[1][2] * z
        cz = s[2][0] * x + s[2][1] * y + s[2][2] * z
        return (cx, cy, cz)

    def fit_ellipsoid(self, samples):
        """Fit hard-iron offsets from collected magnetometer samples.

        Uses least-squares sphere fit (assumes soft-iron is small).
        Requires numpy.

        Args:
            samples: list of (mx, my, mz) tuples, ideally from a full 360° rotation
        """
        import numpy as np

        if len(samples) < 20:
            raise ValueError(f"Need at least 20 samples, got {len(samples)}")

        S = np.array(samples, dtype=np.float64)
        x, y, z = S[:, 0], S[:, 1], S[:, 2]

        # Fit sphere: (x-a)² + (y-b)² + (z-c)² = r²
        # Expand: x² + y² + z² = 2ax + 2by + 2cz + (r² - a² - b² - c²)
        # Linear system: A @ [a, b, c, d]ᵀ = x² + y² + z²
        A = np.column_stack([2 * x, 2 * y, 2 * z, np.ones(len(x))])
        b = x ** 2 + y ** 2 + z ** 2
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        self.offset = [float(result[0]), float(result[1]), float(result[2])]
        # Keep scale as identity for sphere fit (no soft-iron correction)
        self.scale = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
        self.valid = True
        return self.offset
