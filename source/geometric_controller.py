"""
SO(3) Geometric Attitude Controller for Quadrotors.

Implements the almost-globally stable geometric tracking controller from:
  Lee, Leok, McClamroch. "Geometric tracking control of a quadrotor UAV on SE(3)"
  CDC 2010.

Takes a desired thrust direction + yaw from the MPC, outputs torque commands
to track the desired attitude, plus the thrust magnitude command.
"""

import numpy as np
from quadrotor_dynamics import quat_multiply, quat_rotate, quat_normalize, skew, vee


class GeometricAttitudeController:
    """
    SO(3) attitude tracking controller.

    Given:
      - a_des: desired acceleration from MPC (in world frame)
      - ψ_des: desired yaw angle
      - ω_des: desired angular velocity (from differential flatness)

    Produces:
      - f_cmd: total thrust command (N)
      - τ_cmd: torque command [τx, τy, τz] (Nm)
    """

    def __init__(self, mass=0.75, k_R=4.0, k_omega=1.5):
        """
        Args:
            mass: vehicle mass (kg).
            k_R: attitude proportional gain (rotation error → torque).
            k_omega: angular velocity damping gain.
        """
        self.mass = mass
        self.g = 9.81
        self.k_R = k_R      # proportional gain on SO(3) error
        self.k_omega = k_omega

    def compute_control(self, a_des, psi_des, q_current, omega_current,
                        a_des_dot=None):
        """
        Compute thrust and torque commands.

        Args:
            a_des: desired acceleration vector in world frame (3,).
                   From MPC: a_des = u_mpc + a_ff.
            psi_des: desired yaw angle (rad).
            q_current: current attitude quaternion (4,).
            omega_current: current angular velocity (3,).
            a_des_dot: time derivative of a_des for ω_des computation (3,).
                       If None, uses ω_des ≈ 0 (quasi-static assumption).

        Returns:
            f_cmd: thrust command (N).
            tau_cmd: torque command (3,) in body frame.
            R_des: desired rotation matrix (3,3) (for logging).
        """
        # ---- Desired thrust direction ----
        a_total = a_des + np.array([0.0, 0.0, self.g])
        a_norm = np.linalg.norm(a_total)
        if a_norm < 1e-6:
            a_total = np.array([0.0, 0.0, self.g])
            a_norm = self.g

        R = self._quat_to_rot(q_current)
        # Use the realizable component along the current body z-axis. This
        # avoids a vertical thrust spike while attitude is still catching up.
        thrust_cmd = self.mass * max(float(np.dot(a_total, R[:, 2])), 0.0)
        z_b_des = a_total / a_norm  # desired body z-axis

        # ---- Desired attitude from thrust direction + yaw ----
        x_c = np.array([np.cos(psi_des), np.sin(psi_des), 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        y_norm = np.linalg.norm(y_b_des)
        if y_norm < 1e-6:
            y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            y_b_des /= y_norm
        x_b_des = np.cross(y_b_des, z_b_des)
        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])

        # ---- Attitude error on SO(3) ----
        # e_R = 0.5 * vee(R_des^T * R - R^T * R_des)
        R_err = R_des.T @ R
        e_R = 0.5 * vee(R_err - R_err.T)

        # ---- Angular velocity error ----
        # For quasi-static ω_des, approximate as:
        # ω_des ≈ R_des^T * Ṙ_des (from differential flatness)
        # Simplified: ω_des derived from desired angular velocity
        if a_des_dot is not None:
            # ω_des from differential flatness:
            # ω_x = -h_ν · y_b,  ω_y = h_ν · x_b,  ω_z = ψ̇_des · z_b
            # where h_ν = (ȧ × z_b) / ||a_total||
            a_dot = a_des_dot
            h_nu = np.cross(a_dot, z_b_des) / a_norm
            omega_des = np.array([
                -np.dot(h_nu, y_b_des),
                 np.dot(h_nu, x_b_des),
                 0.0  # assuming yaw rate ≈ 0
            ])
        else:
            omega_des = np.zeros(3)

        omega_err = omega_current - R.T @ R_des @ omega_des
        # Simplified: use current omega as error since ω_des is small
        # omega_err = omega_current (when ω_des ≈ 0)

        # ---- Torque command (Lee et al. 2010) ----
        tau_cmd = -self.k_R * e_R - self.k_omega * omega_err
        # Project to body frame (already in body frame since e_R and ω are)

        return thrust_cmd, tau_cmd, R_des

    @staticmethod
    def _quat_to_rot(q):
        """Unit quaternion → rotation matrix."""
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
        ])
