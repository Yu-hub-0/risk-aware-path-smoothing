"""Full-dynamics nonlinear MPC for the quadrotor plant.

The prediction state is
    [position, velocity, quaternion, body_rate, thrust_actual, torque_actual]
and the control is
    [thrust_command, torque_command].

The model intentionally mirrors ``quadrotor_dynamics.py``: it includes rigid-body
attitude dynamics, first-order actuator lag, gravity, and linear aerodynamic drag.
"""

from time import perf_counter

import casadi as ca
import numpy as np


def _vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def _rotation_to_quaternion(rotation):
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        quat = np.array([
            0.25 / s,
            (rotation[2, 1] - rotation[1, 2]) * s,
            (rotation[0, 2] - rotation[2, 0]) * s,
            (rotation[1, 0] - rotation[0, 1]) * s,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quat = np.array([
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            ])
        elif index == 1:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quat = np.array([
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            ])
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quat = np.array([
                (rotation[1, 0] - rotation[0, 1]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            ])
    return quat / np.linalg.norm(quat)


class FullDynamicsNMPC:
    """Multiple-shooting NMPC with the same 17-state model as the plant."""

    NX = 17
    NU = 4

    def __init__(self, dt=0.05, horizon=20, mass=0.75, inertia=None,
                 drag_coeff=None, tau_f=0.05, tau_tau=0.03,
                 max_tilt_deg=45.0, thrust_max_ratio=2.5,
                 torque_max=(0.08, 0.08, 0.04), max_iterations=40):
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.mass = float(mass)
        self.J = (np.diag([0.0025, 0.0025, 0.005]) if inertia is None
                  else np.asarray(inertia, dtype=float))
        self.J_inv = np.linalg.inv(self.J)
        self.drag = np.diag([0.3, 0.3, 0.2]) if drag_coeff is None else np.diag(drag_coeff)
        self.tau_f = float(tau_f)
        self.tau_tau = float(tau_tau)
        self.g = 9.81
        self.max_tilt_rad = np.deg2rad(max_tilt_deg)
        self.thrust_min = 0.0
        self.thrust_max = thrust_max_ratio * self.mass * self.g
        self.torque_max = np.asarray(torque_max, dtype=float)
        self.max_iterations = int(max_iterations)

        self._u_previous = np.array([self.mass * self.g, 0.0, 0.0, 0.0])
        self._u_warm = None
        self._x_warm = None
        self._lam_g_warm = None
        self._build_problem()

    @staticmethod
    def _quat_multiply(q, r):
        return ca.vertcat(
            q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
            q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
            q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
            q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
        )

    @staticmethod
    def _rotation_matrix(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return ca.vertcat(
            ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            ca.horzcat(2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)),
            ca.horzcat(2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)),
        )

    def _continuous_dynamics(self, x, u):
        velocity = x[3:6]
        quaternion = x[6:10]
        omega = x[10:13]
        thrust_actual = x[13]
        torque_actual = x[14:17]

        rotation = self._rotation_matrix(quaternion)
        acceleration = (
            rotation @ ca.vertcat(0.0, 0.0, thrust_actual) / self.mass
            + ca.vertcat(0.0, 0.0, -self.g)
            - ca.DM(self.drag) @ velocity
        )
        q_dot = 0.5 * self._quat_multiply(
            quaternion, ca.vertcat(0.0, omega[0], omega[1], omega[2]))
        angular_momentum = ca.DM(self.J) @ omega
        gyroscopic = ca.cross(omega, angular_momentum)
        omega_dot = ca.DM(self.J_inv) @ (torque_actual - gyroscopic)
        thrust_dot = (u[0] - thrust_actual) / self.tau_f
        torque_dot = (u[1:4] - torque_actual) / self.tau_tau
        return ca.vertcat(velocity, acceleration, q_dot, omega_dot, thrust_dot, torque_dot)

    def _discrete_dynamics(self, x, u):
        dt = self.dt
        k1 = self._continuous_dynamics(x, u)
        k2 = self._continuous_dynamics(x + 0.5 * dt * k1, u)
        k3 = self._continuous_dynamics(x + 0.5 * dt * k2, u)
        k4 = self._continuous_dynamics(x + dt * k3, u)
        x_next = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        q_next = x_next[6:10]
        q_next = q_next / ca.sqrt(ca.dot(q_next, q_next) + 1e-12)
        return ca.vertcat(x_next[0:6], q_next, x_next[10:17])

    def _build_problem(self):
        n = self.horizon
        opti = ca.Opti()
        x_var = opti.variable(self.NX, n + 1)
        u_var = opti.variable(self.NU, n)

        x_initial = opti.parameter(self.NX)
        p_reference = opti.parameter(3, n + 1)
        v_reference = opti.parameter(3, n + 1)
        q_reference = opti.parameter(4, n + 1)
        omega_reference = opti.parameter(3, n + 1)
        u_reference = opti.parameter(self.NU, n)
        u_previous = opti.parameter(self.NU)
        position_lower = opti.parameter(3, n)
        position_upper = opti.parameter(3, n)
        position_error_increment = opti.parameter(3, n + 1)
        velocity_error_increment = opti.parameter(3, n + 1)

        q_position = ca.DM([30.0, 30.0, 45.0])
        q_velocity = ca.DM([6.0, 6.0, 10.0])
        q_omega = ca.DM([0.08, 0.08, 0.04])
        terminal_scale = 6.0
        objective = 0

        opti.subject_to(x_var[:, 0] == x_initial)
        for k in range(n):
            corrected_position = x_var[0:3, k] + position_error_increment[:, k]
            corrected_velocity = x_var[3:6, k] + velocity_error_increment[:, k]
            p_error = corrected_position - p_reference[:, k]
            v_error = corrected_velocity - v_reference[:, k]
            q_dot = ca.dot(x_var[6:10, k], q_reference[:, k])
            omega_error = x_var[10:13, k] - omega_reference[:, k]
            u_error = u_var[:, k] - u_reference[:, k]
            delta_u = u_var[:, k] - (u_previous if k == 0 else u_var[:, k - 1])

            objective += ca.dot(q_position * p_error, p_error)
            objective += ca.dot(q_velocity * v_error, v_error)
            objective += 12.0 * (1.0 - q_dot ** 2)
            objective += ca.dot(q_omega * omega_error, omega_error)
            objective += 0.15 * (u_error[0] / (self.mass * self.g)) ** 2
            objective += ca.sumsqr(u_error[1:4] / ca.DM(self.torque_max)) * 0.01
            objective += 0.35 * (delta_u[0] / (self.mass * self.g)) ** 2
            objective += ca.sumsqr(delta_u[1:4] / ca.DM(self.torque_max)) * 0.08

            x_next = self._discrete_dynamics(x_var[:, k], u_var[:, k])
            opti.subject_to(x_var[:, k + 1] == x_next)
            opti.subject_to(opti.bounded(
                position_lower[:, k],
                x_var[0:3, k + 1] + position_error_increment[:, k + 1],
                position_upper[:, k]))

        p_terminal = (
            x_var[0:3, n] + position_error_increment[:, n]
            - p_reference[:, n]
        )
        v_terminal = (
            x_var[3:6, n] + velocity_error_increment[:, n]
            - v_reference[:, n]
        )
        q_terminal_dot = ca.dot(x_var[6:10, n], q_reference[:, n])
        omega_terminal = x_var[10:13, n] - omega_reference[:, n]
        objective += terminal_scale * ca.dot(q_position * p_terminal, p_terminal)
        objective += terminal_scale * ca.dot(q_velocity * v_terminal, v_terminal)
        objective += terminal_scale * 12.0 * (1.0 - q_terminal_dot ** 2)
        objective += terminal_scale * ca.dot(q_omega * omega_terminal, omega_terminal)

        # R_33 is the vertical component of body-z and directly limits tilt.
        for k in range(1, n + 1):
            rotation = self._rotation_matrix(x_var[6:10, k])
            opti.subject_to(rotation[2, 2] >= np.cos(self.max_tilt_rad))
            opti.subject_to(opti.bounded(self.thrust_min, x_var[13, k], self.thrust_max))

        opti.subject_to(opti.bounded(self.thrust_min, u_var[0, :], self.thrust_max))
        for axis in range(3):
            limit = self.torque_max[axis]
            opti.subject_to(opti.bounded(-limit, u_var[axis + 1, :], limit))

        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": self.max_iterations,
                "tol": 1e-4,
                "acceptable_tol": 5e-3,
                "acceptable_iter": 3,
                "warm_start_init_point": "yes",
            },
        )

        self.opti = opti
        self.x_var = x_var
        self.u_var = u_var
        self.x_initial = x_initial
        self.p_reference = p_reference
        self.v_reference = v_reference
        self.q_reference = q_reference
        self.omega_reference = omega_reference
        self.u_reference = u_reference
        self.u_previous_parameter = u_previous
        self.position_lower = position_lower
        self.position_upper = position_upper
        self.position_error_increment = position_error_increment
        self.velocity_error_increment = velocity_error_increment

    def _build_attitude_and_input_reference(self, velocity, acceleration):
        n_points = len(velocity)
        rotations = np.zeros((n_points, 3, 3))
        quaternions = np.zeros((n_points, 4))
        thrust_actual = np.zeros(n_points)

        for k in range(n_points):
            required_acceleration = (
                acceleration[k] + np.array([0.0, 0.0, self.g])
                + self.drag @ velocity[k]
            )
            magnitude = max(np.linalg.norm(required_acceleration), 1e-8)
            body_z = required_acceleration / magnitude
            heading = np.array([1.0, 0.0, 0.0])
            body_y = np.cross(body_z, heading)
            if np.linalg.norm(body_y) < 1e-8:
                heading = np.array([0.0, 1.0, 0.0])
                body_y = np.cross(body_z, heading)
            body_y /= np.linalg.norm(body_y)
            body_x = np.cross(body_y, body_z)
            rotations[k] = np.column_stack((body_x, body_y, body_z))
            quaternions[k] = _rotation_to_quaternion(rotations[k])
            if k > 0 and np.dot(quaternions[k], quaternions[k - 1]) < 0.0:
                quaternions[k] *= -1.0
            thrust_actual[k] = self.mass * magnitude

        omega = np.zeros((n_points, 3))
        for k in range(n_points - 1):
            relative = rotations[k].T @ rotations[k + 1]
            omega[k] = _vee(0.5 * (relative - relative.T)) / self.dt
        if n_points > 1:
            omega[-1] = omega[-2]

        alpha = np.gradient(omega, self.dt, axis=0, edge_order=1)
        torque_actual = np.zeros_like(omega)
        for k in range(n_points):
            torque_actual[k] = self.J @ alpha[k] + np.cross(omega[k], self.J @ omega[k])

        thrust_command = thrust_actual[:-1] + self.tau_f * np.diff(thrust_actual) / self.dt
        torque_rate = np.gradient(torque_actual, self.dt, axis=0, edge_order=1)
        torque_command = torque_actual[:-1] + self.tau_tau * torque_rate[:-1]
        u_reference = np.column_stack((thrust_command, torque_command))
        u_reference[:, 0] = np.clip(u_reference[:, 0], self.thrust_min, self.thrust_max)
        u_reference[:, 1:4] = np.clip(
            u_reference[:, 1:4], -self.torque_max, self.torque_max)
        return quaternions, omega, u_reference

    def _rollout(self, x0, controls):
        if not hasattr(self, "_rollout_function"):
            x_symbol = ca.MX.sym("rollout_x", self.NX)
            u_symbol = ca.MX.sym("rollout_u", self.NU)
            self._rollout_function = ca.Function(
                "nmpc_rollout_step_cached", [x_symbol, u_symbol],
                [self._discrete_dynamics(x_symbol, u_symbol)])
        states = np.zeros((self.horizon + 1, self.NX))
        states[0] = x0
        for k in range(self.horizon):
            states[k + 1] = np.asarray(
                self._rollout_function(states[k], controls[k])).reshape(-1)
        return states

    def compute_control(self, x0, p_reference, v_reference, a_reference,
                        position_lower=None, position_upper=None,
                        position_error_increment=None,
                        velocity_error_increment=None):
        """Solve one NMPC step and return ``(thrust, torque, info)``."""
        x0 = np.asarray(x0, dtype=float).reshape(self.NX)
        p_reference = np.asarray(p_reference, dtype=float).reshape(self.horizon + 1, 3)
        v_reference = np.asarray(v_reference, dtype=float).reshape(self.horizon + 1, 3)
        a_reference = np.asarray(a_reference, dtype=float).reshape(self.horizon + 1, 3)
        q_reference, omega_reference, u_reference = (
            self._build_attitude_and_input_reference(v_reference, a_reference))

        self.opti.set_value(self.x_initial, x0)
        self.opti.set_value(self.p_reference, p_reference.T)
        self.opti.set_value(self.v_reference, v_reference.T)
        self.opti.set_value(self.q_reference, q_reference.T)
        self.opti.set_value(self.omega_reference, omega_reference.T)
        self.opti.set_value(self.u_reference, u_reference.T)
        self.opti.set_value(self.u_previous_parameter, self._u_previous)
        if position_error_increment is None:
            position_error_increment = np.zeros((self.horizon + 1, 3))
        if velocity_error_increment is None:
            velocity_error_increment = np.zeros((self.horizon + 1, 3))
        position_error_increment = np.asarray(
            position_error_increment, dtype=float
        ).reshape(self.horizon + 1, 3)
        velocity_error_increment = np.asarray(
            velocity_error_increment, dtype=float
        ).reshape(self.horizon + 1, 3)
        self.opti.set_value(
            self.position_error_increment, position_error_increment.T
        )
        self.opti.set_value(
            self.velocity_error_increment, velocity_error_increment.T
        )
        if position_lower is None or position_upper is None:
            position_lower = np.full((self.horizon, 3), -1e9)
            position_upper = np.full((self.horizon, 3), 1e9)
        self.opti.set_value(self.position_lower, np.asarray(position_lower).T)
        self.opti.set_value(self.position_upper, np.asarray(position_upper).T)

        if self._u_warm is None:
            controls_initial = u_reference.copy()
        else:
            controls_initial = np.vstack((self._u_warm[1:], self._u_warm[-1]))
        if self._x_warm is None:
            states_initial = self._rollout(x0, controls_initial)
        else:
            states_initial = np.vstack((x0, self._x_warm[2:], self._x_warm[-1]))
        self.opti.set_initial(self.u_var, controls_initial.T)
        self.opti.set_initial(self.x_var, states_initial.T)
        if self._lam_g_warm is not None:
            self.opti.set_initial(self.opti.lam_g, self._lam_g_warm)

        started = perf_counter()
        try:
            solution = self.opti.solve()
            controls = np.asarray(solution.value(self.u_var)).T
            states = np.asarray(solution.value(self.x_var)).T
            self._lam_g_warm = solution.value(self.opti.lam_g)
            success = True
            status = self.opti.stats().get("return_status", "Solve_Succeeded")
        except RuntimeError as error:
            controls = controls_initial
            states = states_initial
            success = False
            status = self.opti.stats().get("return_status", str(error).splitlines()[0])
            self._lam_g_warm = None

        elapsed = perf_counter() - started
        command = controls[0].copy()
        command[0] = np.clip(command[0], self.thrust_min, self.thrust_max)
        command[1:4] = np.clip(command[1:4], -self.torque_max, self.torque_max)
        self._u_previous = command
        self._u_warm = controls
        self._x_warm = states
        return float(command[0]), command[1:4], {
            "success": success,
            "status": status,
            "solve_time": elapsed,
            "predicted_states": states,
        }

    def reset(self):
        self._u_previous = np.array([self.mass * self.g, 0.0, 0.0, 0.0])
        self._u_warm = None
        self._x_warm = None
        self._lam_g_warm = None
