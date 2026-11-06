"""
Translational MPC for Quadrotor Trajectory Tracking.

Cascaded architecture (standard in the literature):
    MPC (translational) → a_des → Attitude Controller → [f, τ] → Quadrotor

The MPC supports a double-integrator model and an optional first-order
acceleration-response model. Both use integral-error augmentation.

Uses OSQP for efficient QP solving (50 Hz capable).
"""

import numpy as np

try:
    import osqp
    from scipy import sparse
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

from scipy.optimize import LinearConstraint, minimize


class TranslationalMPC:
    """
    Linear MPC for translational dynamics of a quadrotor.

    Model:  ṗ = v,  v̇ = a_cmd   (acceleration as control input)

    The MPC outputs a_des (desired acceleration in world frame), which is
    passed to the attitude controller along with a desired yaw.

    Integral error augmentation rejects constant disturbances.
    """

    def __init__(self, dt=0.02, horizon=25,
                 Q_pos=40.0, Q_vel=10.0, Q_int=4.0,
                 R_ctrl=0.05, a_max=12.0,
                 accel_time_constant=None, Q_accel=0.0, R_delta=0.0):
        """
        Args:
            dt: MPC timestep (s). Paper-typical: 0.02s (50 Hz).
            horizon: prediction horizon steps. 25 steps @ 50Hz = 0.5s.
            Q_pos: position error weight (higher → tighter tracking).
            Q_vel: velocity error weight.
            Q_int: integral error weight (robustness).
            R_ctrl: control effort weight.
            accel_time_constant: optional first-order acceleration response
                time constant. None keeps the double-integrator model.
            Q_accel: acceleration-error weight for the lag-aware model.
            R_delta: acceleration-command increment weight.
            a_max: acceleration command limit per axis (m/s²).
                   ~12 m/s² ≈ 1.2g, typical for aggressive quadrotor flight.
        """
        self.dt = dt
        self.horizon = horizon
        self.a_max = a_max

        self.Q_pos = Q_pos
        self.Q_vel = Q_vel
        self.Q_int = Q_int
        self.R_ctrl = R_ctrl
        self.accel_time_constant = accel_time_constant
        self.Q_accel = Q_accel
        self.R_delta = R_delta

        self.nx = 9    # [px,py,pz, vx,vy,vz, eix,eiy,eiz]
        self.nu = 3    # [ax_cmd, ay_cmd, az_cmd]

        self._build_model()
        self._u_prev = None  # warm-start
        self._u_applied_prev = np.zeros(self.nu)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self):
        dt = self.dt
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        self.A = np.block([
            [I3, dt * I3, Z3],
            [Z3, I3,      Z3],
            [dt * I3, Z3, I3],
        ])
        self.B = np.block([
            [Z3],
            [dt * I3],
            [Z3],
        ])

    # ------------------------------------------------------------------
    # Condensed QP matrices
    # ------------------------------------------------------------------
    def _build_qp(self, x0, pref_seq, vref_seq):
        """
        Build condensed QP:  min  0.5 U^T H U + f^T U
                             s.t. lb ≤ U ≤ ub
        """
        N, nx, nu = self.horizon, self.nx, self.nu

        # Block-diagonal weight matrices
        Q_blk = np.diag(np.concatenate([
            np.full(3, self.Q_pos), np.full(3, self.Q_vel), np.full(3, self.Q_int)
        ]))
        Q_bar = np.kron(np.eye(N), Q_blk)
        R_bar = np.kron(np.eye(N), self.R_ctrl * np.eye(nu))

        # Prediction matrices: X = F @ x0 + G @ U
        F = np.zeros((N * nx, nx))
        G = np.zeros((N * nx, N * nu))

        A_pow = np.eye(nx)
        for k in range(N):
            F[k * nx:(k + 1) * nx, :] = self.A @ A_pow
            A_pow = self.A @ A_pow

            Ap = np.eye(nx)
            for j in range(k, -1, -1):
                G[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = Ap @ self.B
                Ap = self.A @ Ap

        # Reference state sequence
        x_ref = np.zeros((N, nx))
        for k in range(N):
            x_ref[k, 0:3] = pref_seq[k]
            x_ref[k, 3:6] = vref_seq[k]
        x_ref_flat = x_ref.reshape(N * nx)

        # Condensed Hessian and gradient
        H = G.T @ Q_bar @ G + R_bar
        f = G.T @ Q_bar @ (F @ x0 - x_ref_flat)

        return H, f, F, G

    def _build_tracking_qp(self, x0, p_ref, v_ref, a_ref,
                           pref_seq, vref_seq, aref_seq):
        """Build an error-state QP whose input is acceleration correction.

        The executed acceleration is ``a_ref + u``. Reference discretization
        residuals are retained as affine terms, so the formulation also works
        when the supplied samples are not an exact double-integrator rollout.
        """
        N, nx, nu = self.horizon, self.nx, self.nu
        dt = self.dt
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        A = np.block([
            [I3, dt * I3, Z3],
            [Z3, I3, Z3],
            [dt * I3, Z3, I3],
        ])
        B = np.block([
            [0.5 * dt * dt * I3],
            [dt * I3],
            [Z3],
        ])

        p_nodes = np.vstack([np.asarray(p_ref), np.asarray(pref_seq)])
        v_nodes = np.vstack([np.asarray(v_ref), np.asarray(vref_seq)])
        a_intervals = np.vstack([np.asarray(a_ref), np.asarray(aref_seq)[:-1]])

        c_seq = np.zeros((N, nx))
        for k in range(N):
            c_seq[k, 0:3] = (
                p_nodes[k] + dt * v_nodes[k]
                + 0.5 * dt * dt * a_intervals[k] - p_nodes[k + 1]
            )
            c_seq[k, 3:6] = (
                v_nodes[k] + dt * a_intervals[k] - v_nodes[k + 1]
            )

        F = np.zeros((N * nx, nx))
        G = np.zeros((N * nx, N * nu))
        affine = np.zeros(N * nx)
        A_pow = np.eye(nx)
        affine_state = np.zeros(nx)
        for k in range(N):
            F[k * nx:(k + 1) * nx, :] = A @ A_pow
            A_pow = A @ A_pow

            Ap = np.eye(nx)
            for j in range(k, -1, -1):
                G[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = Ap @ B
                Ap = A @ Ap

            affine_state = A @ affine_state + c_seq[k]
            affine[k * nx:(k + 1) * nx] = affine_state

        e0 = np.concatenate([
            np.asarray(x0[0:3]) - np.asarray(p_ref),
            np.asarray(x0[3:6]) - np.asarray(v_ref),
            np.asarray(x0[6:9]),
        ])
        error_offset = F @ e0 + affine

        Q_blk = np.diag(np.concatenate([
            np.full(3, self.Q_pos), np.full(3, self.Q_vel),
            np.full(3, self.Q_int),
        ]))
        Q_bar = np.kron(np.eye(N), Q_blk)
        R_bar = np.kron(np.eye(N), self.R_ctrl * np.eye(nu))
        H = G.T @ Q_bar @ G + R_bar
        f = G.T @ Q_bar @ error_offset

        # Position constraints are expressed in absolute coordinates.
        absolute_offset = error_offset.copy()
        for k in range(N):
            absolute_offset[k * nx:k * nx + 3] += pref_seq[k]
        return H, f, G, absolute_offset

    def _build_lag_tracking_qp(self, x0, a_actual, p_ref, v_ref, a_ref,
                               pref_seq, vref_seq, aref_seq):
        """Build an error-state QP with first-order acceleration dynamics."""
        N, nx, nu = self.horizon, 12, self.nu
        dt = self.dt
        tau = float(self.accel_time_constant)
        if tau <= 0:
            raise ValueError("accel_time_constant must be positive")

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        alpha = np.exp(-dt / tau)
        vel_from_accel = tau * (1.0 - alpha)
        vel_from_command = dt - vel_from_accel
        pos_from_accel = tau * dt - tau * tau * (1.0 - alpha)
        pos_from_command = 0.5 * dt * dt - pos_from_accel

        # State: [position error, velocity error, acceleration error, integral error].
        # Input: acceleration correction added to the reference acceleration.
        A = np.block([
            [I3, dt * I3, pos_from_accel * I3, Z3],
            [Z3, I3, vel_from_accel * I3, Z3],
            [Z3, Z3, alpha * I3, Z3],
            [dt * I3, Z3, Z3, I3],
        ])
        B = np.block([
            [pos_from_command * I3],
            [vel_from_command * I3],
            [(1.0 - alpha) * I3],
            [Z3],
        ])

        p_nodes = np.vstack([np.asarray(p_ref), np.asarray(pref_seq)])
        v_nodes = np.vstack([np.asarray(v_ref), np.asarray(vref_seq)])
        a_nodes = np.vstack([np.asarray(a_ref), np.asarray(aref_seq)])

        c_seq = np.zeros((N, nx))
        for k in range(N):
            c_seq[k, 0:3] = (
                p_nodes[k] + dt * v_nodes[k]
                + 0.5 * dt * dt * a_nodes[k] - p_nodes[k + 1]
            )
            c_seq[k, 3:6] = (
                v_nodes[k] + dt * a_nodes[k] - v_nodes[k + 1]
            )
            c_seq[k, 6:9] = a_nodes[k] - a_nodes[k + 1]

        F = np.zeros((N * nx, nx))
        G = np.zeros((N * nx, N * nu))
        affine = np.zeros(N * nx)
        A_pow = np.eye(nx)
        affine_state = np.zeros(nx)
        for k in range(N):
            F[k * nx:(k + 1) * nx, :] = A @ A_pow
            A_pow = A @ A_pow

            Ap = np.eye(nx)
            for j in range(k, -1, -1):
                G[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = Ap @ B
                Ap = A @ Ap

            affine_state = A @ affine_state + c_seq[k]
            affine[k * nx:(k + 1) * nx] = affine_state

        e0 = np.concatenate([
            np.asarray(x0[0:3]) - np.asarray(p_ref),
            np.asarray(x0[3:6]) - np.asarray(v_ref),
            np.asarray(a_actual) - np.asarray(a_ref),
            np.asarray(x0[6:9]),
        ])
        error_offset = F @ e0 + affine

        Q_blk = np.diag(np.concatenate([
            np.full(3, self.Q_pos), np.full(3, self.Q_vel),
            np.full(3, self.Q_accel), np.full(3, self.Q_int),
        ]))
        Q_bar = np.kron(np.eye(N), Q_blk)
        R_bar = np.kron(np.eye(N), self.R_ctrl * np.eye(nu))
        H = G.T @ Q_bar @ G + R_bar
        f = G.T @ Q_bar @ error_offset

        if self.R_delta > 0:
            D = np.eye(N * nu)
            for k in range(1, N):
                row = slice(k * nu, (k + 1) * nu)
                prev = slice((k - 1) * nu, k * nu)
                D[row, prev] = -np.eye(nu)
            previous = np.zeros(N * nu)
            previous[:nu] = self._u_applied_prev
            H += self.R_delta * (D.T @ D)
            f -= self.R_delta * (D.T @ previous)

        absolute_offset = error_offset.copy()
        for k in range(N):
            absolute_offset[k * nx:k * nx + 3] += pref_seq[k]
        return H, f, G, absolute_offset, nx

    # ------------------------------------------------------------------
    # Solve MPC
    # ------------------------------------------------------------------
    def compute_control(self, x0, pref_seq, vref_seq,
                        pos_lower_seq=None, pos_upper_seq=None,
                        p_ref=None, v_ref=None, a_ref=None, aref_seq=None,
                        a_actual=None):
        """
        Args:
            x0: current augmented state [p, v, e_int] (9,).
            pref_seq: position reference over horizon (horizon, 3).
            vref_seq: velocity reference over horizon (horizon, 3).
            pos_lower_seq: optional lower position bounds over horizon (horizon, 3).
            pos_upper_seq: optional upper position bounds over horizon (horizon, 3).

        Returns:
            a_des: desired acceleration [ax, ay, az] (3,) in world frame.
            info: solver info dict.
        """
        use_tracking_error_model = all(
            value is not None for value in (p_ref, v_ref, a_ref, aref_seq)
        )
        use_accel_lag_model = (
            use_tracking_error_model
            and self.accel_time_constant is not None
            and a_actual is not None
        )
        if use_accel_lag_model:
            H, f, G, prediction_offset, prediction_nx = self._build_lag_tracking_qp(
                x0, a_actual, p_ref, v_ref, a_ref,
                pref_seq, vref_seq, aref_seq
            )
        elif use_tracking_error_model:
            H, f, G, prediction_offset = self._build_tracking_qp(
                x0, p_ref, v_ref, a_ref, pref_seq, vref_seq, aref_seq
            )
            prediction_nx = self.nx
        else:
            H, f, F, G = self._build_qp(x0, pref_seq, vref_seq)
            prediction_offset = F @ x0
            prediction_nx = self.nx
        N, nu = self.horizon, self.nu
        n_u = N * nu

        # Box constraints
        lb = np.full(n_u, -self.a_max)
        ub = np.full(n_u, self.a_max)

        has_pos_bounds = pos_lower_seq is not None or pos_upper_seq is not None
        if has_pos_bounds:
            if pos_lower_seq is None or pos_upper_seq is None:
                raise ValueError("pos_lower_seq and pos_upper_seq must be provided together")
            pos_lower_seq = np.asarray(pos_lower_seq, dtype=np.float64)
            pos_upper_seq = np.asarray(pos_upper_seq, dtype=np.float64)
            if pos_lower_seq.shape != (N, 3) or pos_upper_seq.shape != (N, 3):
                raise ValueError("position bounds must have shape (horizon, 3)")

            pos_rows = []
            for k in range(N):
                base = k * prediction_nx
                pos_rows.extend([base, base + 1, base + 2])
            pos_rows = np.asarray(pos_rows)

            A_pos = G[pos_rows, :]
            pos_offset = prediction_offset[pos_rows]
            pos_lb = pos_lower_seq.reshape(N * 3) - pos_offset
            pos_ub = pos_upper_seq.reshape(N * 3) - pos_offset

        # Warm-start: shift previous solution
        if self._u_prev is not None:
            u_warm = np.zeros(n_u)
            u_warm[:(N - 1) * nu] = self._u_prev[nu:]
            u_warm[(N - 1) * nu:] = self._u_prev[-nu:]
        else:
            u_warm = np.zeros(n_u)

        if HAS_OSQP:
            # --- OSQP solver (fast, paper-grade) ---
            H_sparse = sparse.csc_matrix(H)
            if has_pos_bounds:
                A = sparse.vstack([
                    sparse.eye(n_u, format='csc'),
                    sparse.csc_matrix(A_pos),
                ], format='csc')
                l = np.concatenate([lb, pos_lb])
                u = np.concatenate([ub, pos_ub])
            else:
                A = sparse.eye(n_u, format='csc')
                l = lb
                u = ub
            solver = osqp.OSQP()
            solver.setup(P=H_sparse, q=f, A=A,
                         l=l, u=u, verbose=False,
                         eps_abs=1e-4, eps_rel=1e-4,
                         max_iter=2000, polish=False)
            solver.warm_start(x=u_warm)
            result = solver.solve()
            u_opt = result.x if result.info.status_val in (1, 2) else u_warm
            success = result.info.status_val in (1, 2)
            status = result.info.status
        else:
            # --- Fallback: L-BFGS-B ---
            constraints = []
            if has_pos_bounds:
                constraints.append(LinearConstraint(A_pos, pos_lb, pos_ub))
            res = minimize(
                lambda u: 0.5 * u @ H @ u + f @ u,
                u_warm,
                method='SLSQP' if has_pos_bounds else 'L-BFGS-B',
                jac=lambda u: H @ u + f,
                bounds=[(lb[i], ub[i]) for i in range(n_u)],
                constraints=constraints,
                options={'maxiter': 300, 'disp': False},
            )
            u_opt = res.x
            success = res.success
            status = res.message

        self._u_prev = u_opt
        a_des = u_opt[:nu]
        self._u_applied_prev = a_des.copy()

        info = {'success': success, 'status': status}
        return a_des, info

    # ------------------------------------------------------------------
    # Integral error update (called outside MPC each timestep)
    # ------------------------------------------------------------------
    def update_integral(self, e_int, p_current, p_ref):
        """e_int += (p - p_ref) * dt"""
        return e_int + (p_current - p_ref) * self.dt

    def reset(self):
        self._u_prev = None
        self._u_applied_prev = np.zeros(self.nu)
