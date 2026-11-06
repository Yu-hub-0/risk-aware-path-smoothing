"""
Full 6-DOF quadrotor dynamics model matching paper-level fidelity.

State (13D):  [x, y, z,  vx, vy, vz,  qw, qx, qy, qz,  ωx, ωy, ωz]
Control (4D): [f_total, τx, τy, τz]  (total thrust in body frame, body torques)

Includes:
  - Full nonlinear attitude dynamics on SO(3) via quaternions
  - First-order actuator dynamics (thrust & torque lag)
  - Linear aerodynamic drag
  - Configurable mass, inertia, arm length
"""

import numpy as np


def quat_multiply(q, r):
    """Hamilton product: q ⊗ r."""
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q: q ⊗ [0,v] ⊗ q*."""
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return result[1:4]


def quat_to_rotation(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def quat_normalize(q):
    """Normalize a quaternion."""
    return q / np.linalg.norm(q)


def skew(v):
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def vee(S):
    """Inverse of skew: extract 3-vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class QuadrotorDynamics:
    """
    Full 6-DOF quadrotor model.

    Dynamics:
        ṗ = v
        v̇ = (1/m) * R * [0, 0, f]^T - g*e₃ - D*v     (drag)
        q̇ = 0.5 * q ⊗ [0, ω]
        ω̇ = J⁻¹ * (τ - ω × J ω)

    where D is the diagonal drag coefficient matrix.
    """

    def __init__(self, mass=0.75, inertia=None, drag_coeff=None,
                 dt=0.02, tau_f=0.05, tau_tau=0.03):
        """
        Args:
            mass: vehicle mass (kg). Typical small quadrotor: 0.5-1.0 kg.
            inertia: (3,3) inertia matrix. Default is for a ~250mm quad.
            drag_coeff: diagonal drag coefficients [dx, dy, dz].
            dt: simulation timestep (s).
            tau_f: thrust actuator time constant (s).
            tau_tau: torque actuator time constant (s).
        """
        self.mass = mass

        if inertia is None:
            # Typical for a 5-inch quadrotor
            self.J = np.diag([0.0025, 0.0025, 0.005])
        else:
            self.J = np.asarray(inertia)
        self.J_inv = np.linalg.inv(self.J)

        if drag_coeff is None:
            self.D = np.diag([0.3, 0.3, 0.2])  # moderate drag
        else:
            self.D = np.diag(drag_coeff)

        self.dt = dt
        self.tau_f = tau_f          # thrust lag
        self.tau_tau = tau_tau      # torque lag
        self.g = 9.81

        # State: [p(3), v(3), q(4), ω(3)]  → 13D
        self.state = np.zeros(13)
        self.state[6] = 1.0  # identity quaternion: qw=1

        # Actuator states (first-order lowpass)
        self.f_actual = 0.0           # actual thrust (N)
        self.tau_actual = np.zeros(3)  # actual torque (Nm)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    @property
    def position(self):     return self.state[0:3]

    @property
    def velocity(self):     return self.state[3:6]

    @property
    def quaternion(self):   return self.state[6:10]

    @property
    def omega(self):        return self.state[10:13]

    @property
    def rotation_matrix(self):
        return quat_to_rotation(self.quaternion)

    @property
    def euler(self):
        """Roll, pitch, yaw (XYZ convention) in radians."""
        q = self.quaternion
        qw, qx, qy, qz = q
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        return np.array([roll, pitch, yaw])

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def reset(self, p=None, v=None, q=None, omega=None):
        """Reset to a given initial state."""
        if p is not None:
            self.state[0:3] = np.asarray(p)
        else:
            self.state[0:3] = 0.0
        if v is not None:
            self.state[3:6] = np.asarray(v)
        else:
            self.state[3:6] = 0.0
        if q is not None:
            self.state[6:10] = quat_normalize(np.asarray(q))
        else:
            self.state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        if omega is not None:
            self.state[10:13] = np.asarray(omega)
        else:
            self.state[10:13] = 0.0
        self.f_actual = 0.0
        self.tau_actual = np.zeros(3)

    def set_state_from_reference(self, p_ref, v_ref, a_ref, psi_ref=0.0):
        """
        Initialize the quadrotor at a reference state using differential flatness.

        Given flat outputs (p, v, a, ψ), compute the corresponding attitude
        and set the state. This is used to start the simulation on the reference.
        """
        # Desired body z-axis (thrust direction)
        a_total = a_ref + np.array([0.0, 0.0, self.g])
        a_norm = np.linalg.norm(a_total)
        if a_norm < 1e-6:
            a_total = np.array([0.0, 0.0, self.g])
            a_norm = self.g
        z_b = a_total / a_norm

        # Desired body x-axis (from yaw)
        x_c = np.array([np.cos(psi_ref), np.sin(psi_ref), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b_norm = np.linalg.norm(y_b)
        if y_b_norm < 1e-6:
            y_b = np.array([0.0, 1.0, 0.0])
        else:
            y_b /= y_b_norm
        x_b = np.cross(y_b, z_b)

        R_des = np.column_stack([x_b, y_b, z_b])

        # Convert rotation matrix to quaternion
        q = self._rot_to_quat(R_des)
        self.state[6:10] = q

        # Approximate ω from R and its derivative
        # ω ≈ R^T * Ṙ
        # We compute Ṙ numerically or from jerk
        # For initialization, set ω=0 (simplification)
        self.state[10:13] = np.zeros(3)

        self.state[0:3] = np.asarray(p_ref)
        self.state[3:6] = np.asarray(v_ref)
        self.f_actual = self.mass * a_norm
        self.tau_actual = np.zeros(3)

    @staticmethod
    def _rot_to_quat(R):
        """Rotation matrix → unit quaternion."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        q = np.array([qw, qx, qy, qz])
        return q / np.linalg.norm(q)

    def step(self, f_cmd, tau_cmd, wind_accel=None):
        """
        Advance one timestep with commanded thrust and torques.

        Args:
            f_cmd: commanded total thrust (N), scalar.
            tau_cmd: commanded body torques [τx, τy, τz] (Nm), (3,).
            wind_accel: external wind acceleration [ax, ay, az] (m/s²), (3,).

        Returns:
            state: full state vector (13,).
        """
        dt = self.dt

        # ---- Actuator dynamics (first-order lag) ----
        alpha_f = dt / self.tau_f
        self.f_actual += alpha_f * (f_cmd - self.f_actual)

        alpha_tau = dt / self.tau_tau
        self.tau_actual += alpha_tau * (np.asarray(tau_cmd) - self.tau_actual)

        # ---- Extract current state ----
        p = self.state[0:3]
        v = self.state[3:6]
        q = self.state[6:10]
        omega = self.state[10:13]
        R = quat_to_rotation(q)

        # ---- External disturbance ----
        wind = np.zeros(3) if wind_accel is None else np.asarray(wind_accel)

        # ---- Translational dynamics ----
        # v̇ = (1/m) * R * [0, 0, f]^T - g*e₃ - D*v + wind
        thrust_body = np.array([0.0, 0.0, self.f_actual])
        thrust_world = R @ thrust_body
        gravity = np.array([0.0, 0.0, -self.g])

        # Aerodynamic drag: F_drag = -D * v
        drag = -self.D @ v

        a = (thrust_world) / self.mass + gravity + drag + wind

        # Euler integration (position + velocity)
        p_new = p + v * dt
        v_new = v + a * dt

        # ---- Rotational dynamics ----
        # q̇ = 0.5 * q ⊗ [0, ω]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quat_multiply(q, omega_quat)
        q_new = quat_normalize(q + q_dot * dt)

        # ω̇ = J⁻¹ * (τ - ω × J ω)
        omega_dot = self.J_inv @ (self.tau_actual - np.cross(omega, self.J @ omega))
        omega_new = omega + omega_dot * dt

        # ---- Update state ----
        self.state[0:3] = p_new
        self.state[3:6] = v_new
        self.state[6:10] = q_new
        self.state[10:13] = omega_new

        return self.state.copy()
