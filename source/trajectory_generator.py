"""
Minimum Snap Trajectory Generator for Quadrotor UAVs.

Converts discrete waypoints into a smooth, time-parameterized reference trajectory
suitable for MPC tracking. Uses piecewise 7th-order polynomials minimizing the
integral of squared snap (4th derivative) to guarantee smoothness up to jerk.
"""

import numpy as np
from scipy.linalg import block_diag


class MinimumSnapTrajectory:
    """
    Generates a minimum-snap trajectory through a sequence of 3D waypoints.

    The trajectory consists of M-1 polynomial segments (where M = number of
    waypoints). Each segment is a 7th-order polynomial minimizing ∫(snap)²dt,
    subject to waypoint position constraints and continuity of derivatives up
    to jerk at internal waypoints.

    Usage:
        waypoints = np.array([[0,0,1], [3,2,2], [6,-1,3], [8,3,1.5], [10,0,1]])
        traj = MinimumSnapTrajectory(waypoints, v_avg=2.0)
        t = np.linspace(0, traj.total_time, 1000)
        p, v, a = traj.get_reference(t)
    """

    def __init__(self, waypoints, v_avg=2.0, time_scale=1.0):
        """
        Args:
            waypoints: (M, 3) array of waypoint positions [x, y, z] in meters.
            v_avg: average flight speed for time allocation (m/s).
            time_scale: global time scaling factor (>1 = slower, <1 = faster).
        """
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (M, 3)")
        if len(self.waypoints) < 2:
            raise ValueError("need at least 2 waypoints")

        self.n_waypoints = len(self.waypoints)
        self.n_segments = self.n_waypoints - 1
        self.v_avg = v_avg
        self.time_scale = time_scale

        # Allocate time per segment, then solve for polynomial coefficients
        self.T = self._allocate_times()
        self.coeffs = self._solve_minimum_snap()  # shape: (n_segments, 3, 8)

    # ------------------------------------------------------------------
    # Time allocation
    # ------------------------------------------------------------------
    def _allocate_times(self):
        """Allocate segment durations proportional to Euclidean distance."""
        T = np.zeros(self.n_segments)
        for i in range(self.n_segments):
            dist = np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i])
            T[i] = max(dist / self.v_avg, 0.5)  # minimum 0.5s per segment
        T *= self.time_scale
        return T

    @property
    def total_time(self):
        return np.sum(self.T)

    def segment_start_times(self):
        """Return the global start time of each segment."""
        return np.concatenate([[0.0], np.cumsum(self.T)[:-1]])

    # ------------------------------------------------------------------
    # Minimum Snap QP construction
    # ------------------------------------------------------------------
    def _segment_Q(self, T):
        """
        Build the 8x8 Hessian for one segment of duration T.
        Q[i,j] = αᵢαⱼ · T^{i+j-7} / (i+j-7)  for i,j >= 4, else 0
        where α = [0, 0, 0, 0, 24, 120, 360, 840]
        """
        alpha = np.array([0, 0, 0, 0, 24, 120, 360, 840], dtype=np.float64)
        Q = np.zeros((8, 8), dtype=np.float64)
        for i in range(4, 8):
            for j in range(4, 8):
                exp = i + j - 7
                Q[i, j] = alpha[i] * alpha[j] * (T ** exp) / exp
        return Q

    def _segment_endpoint_matrix(self, T):
        """
        Returns a 8x8 matrix M such that:
            M @ c = [p(0), p'(0), p''(0), p'''(0),
                     p(T), p'(T), p''(T), p'''(T)]^T
        for a polynomial with coefficients c[0]..c[7].
        """
        M = np.zeros((8, 8), dtype=np.float64)
        # Derivatives at t=0  (rows 0-3)
        M[0, 0] = 1.0              # p(0)
        M[1, 1] = 1.0              # p'(0)
        M[2, 2] = 2.0              # p''(0)
        M[3, 3] = 6.0              # p'''(0)
        # Derivatives at t=T  (rows 4-7)
        for j in range(8):
            M[4, j] = T ** j                         # p(T)
            if j >= 1:
                M[5, j] = j * T ** (j - 1)           # p'(T)
            if j >= 2:
                M[6, j] = j * (j - 1) * T ** (j - 2)  # p''(T)
            if j >= 3:
                M[7, j] = j * (j - 1) * (j - 2) * T ** (j - 3)  # p'''(T)
        return M

    def _solve_minimum_snap(self):
        """
        Solve the equality-constrained QP for all segments and all dimensions.

        Formulation per dimension:
            min  c^T Q_total c
            s.t. A c = b

        Solved via KKT system:
            [Q_total  A^T] [c]   [0]
            [A        0  ] [λ] = [b]
        """
        n_vars = self.n_segments * 8  # total number of coefficients per dim

        # --- Build block-diagonal Q ---
        Q_blocks = [self._segment_Q(self.T[i]) for i in range(self.n_segments)]
        Q_total = block_diag(*Q_blocks)

        # --- Build constraint matrices ---
        A_rows = []
        b_rows = []

        # Helper to add a constraint row (sparse, length n_vars)
        def add_constraint(row, rhs):
            A_rows.append(row)
            b_rows.append(rhs)

        for seg in range(self.n_segments):
            T_i = self.T[seg]
            offset = seg * 8

            # Position at segment start = waypoint[seg]
            row = np.zeros(n_vars)
            row[offset + 0] = 1.0
            add_constraint(row, 0.0)  # RHS set per dimension later

            # Position at segment end = waypoint[seg+1]
            row = np.zeros(n_vars)
            for j in range(8):
                row[offset + j] = T_i ** j
            add_constraint(row, 0.0)

        # Continuity constraints at internal waypoints
        for seg in range(self.n_segments - 1):
            T_i = self.T[seg]
            offset_curr = seg * 8
            offset_next = (seg + 1) * 8

            for deriv in range(1, 4):  # velocity (1), acceleration (2), jerk (3)
                row = np.zeros(n_vars)
                # Current segment at t=T_i
                for j in range(deriv, 8):
                    coeff = 1.0
                    for k in range(deriv):
                        coeff *= (j - k)
                    row[offset_curr + j] = coeff * T_i ** (j - deriv)
                # Next segment at t=0
                coeff0 = 1.0
                for k in range(deriv):
                    coeff0 *= (deriv - k)  # = deriv!
                row[offset_next + deriv] = -coeff0
                add_constraint(row, 0.0)

        # Boundary conditions: zero velocity, acceleration, jerk at start & end
        # Start (segment 0, t=0)
        for deriv in range(1, 4):
            row = np.zeros(n_vars)
            coeff = 1.0
            for k in range(deriv):
                coeff *= (deriv - k)
            row[deriv] = coeff  # offset = 0 for segment 0
            add_constraint(row, 0.0)

        # End (last segment, t=T_last)
        last_seg = self.n_segments - 1
        T_last = self.T[last_seg]
        offset_last = last_seg * 8
        for deriv in range(1, 4):
            row = np.zeros(n_vars)
            for j in range(deriv, 8):
                coeff = 1.0
                for k in range(deriv):
                    coeff *= (j - k)
                row[offset_last + j] = coeff * T_last ** (j - deriv)
            add_constraint(row, 0.0)

        A = np.array(A_rows)
        n_constraints = len(A_rows)

        # --- Solve for each dimension ---
        coeffs = np.zeros((self.n_segments, 3, 8))

        for dim in range(3):
            b = np.zeros(n_constraints)
            # Set position constraints for this dimension
            con_idx = 0
            for seg in range(self.n_segments):
                b[con_idx] = self.waypoints[seg, dim]
                con_idx += 1
                b[con_idx] = self.waypoints[seg + 1, dim]
                con_idx += 1
            # Remaining constraints are zero (continuity + boundary)

            # Solve KKT system: [Q  A^T; A  0] [c; λ] = [0; b]
            KKT = np.block([
                [Q_total,      A.T],
                [A,      np.zeros((n_constraints, n_constraints))]
            ])
            rhs = np.concatenate([np.zeros(n_vars), b])
            solution = np.linalg.solve(KKT, rhs)
            c = solution[:n_vars]

            # Reshape into (n_segments, 8)
            coeffs[:, dim, :] = c.reshape(self.n_segments, 8)

        return coeffs

    # ------------------------------------------------------------------
    # Reference signal evaluation
    # ------------------------------------------------------------------
    def _find_segment(self, t):
        """Find which segment contains global time t."""
        t = np.clip(t, 0.0, self.total_time)
        cumsum = np.cumsum(self.T)
        seg = np.searchsorted(cumsum, t, side='right')
        return min(seg, self.n_segments - 1)

    def _segment_time(self, t_global):
        """Convert global time to (segment_index, local_time_in_segment)."""
        seg = self._find_segment(t_global)
        t0 = np.sum(self.T[:seg]) if seg > 0 else 0.0
        t_local = t_global - t0
        return seg, t_local

    def _eval_poly(self, c, t, deriv=0):
        """
        Evaluate polynomial with coefficients c at time t.
        deriv=0: position, 1: velocity, 2: acceleration, 3: jerk, 4: snap.
        """
        if deriv >= len(c):
            return 0.0
        result = 0.0
        n = len(c)
        for j in range(deriv, n):
            coeff = c[j]
            for k in range(deriv):
                coeff *= (j - k)
            result += coeff * t ** (j - deriv)
        return result

    def get_reference(self, t):
        """
        Evaluate reference trajectory at time(s) t.

        Args:
            t: scalar or array of query times.

        Returns:
            p: (..., 3) position
            v: (..., 3) velocity
            a: (..., 3) acceleration
            j: (..., 3) jerk
        """
        scalar = np.ndim(t) == 0
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        n = len(t_arr)

        p = np.zeros((n, 3)); v = np.zeros((n, 3))
        a = np.zeros((n, 3)); j = np.zeros((n, 3))

        for i, ti in enumerate(t_arr):
            seg, t_local = self._segment_time(ti)
            for dim in range(3):
                c = self.coeffs[seg, dim, :]
                p[i, dim] = self._eval_poly(c, t_local, deriv=0)
                v[i, dim] = self._eval_poly(c, t_local, deriv=1)
                a[i, dim] = self._eval_poly(c, t_local, deriv=2)
                j[i, dim] = self._eval_poly(c, t_local, deriv=3)

        if scalar:
            return p[0], v[0], a[0], j[0]
        return p, v, a, j

    def get_reference_sequence(self, t0, horizon, dt):
        """
        Get the MPC reference sequence over the prediction horizon.

        Returns:
            pref: (horizon, 3) position reference
            vref: (horizon, 3) velocity reference
            aref: (horizon, 3) acceleration reference
            jref: (horizon, 3) jerk reference
        """
        ts = t0 + np.arange(1, horizon + 1) * dt
        return self.get_reference(ts)

    def get_total_reference(self, dt):
        """
        Full reference trajectory sampled at dt.

        Returns: ts, pref, vref, aref, jref
        """
        N = int(self.total_time / dt) + 1
        ts = np.linspace(0, self.total_time, N)
        pref, vref, aref, jref = self.get_reference(ts)
        return ts, pref, vref, aref, jref

    # ------------------------------------------------------------------
    # Differential flatness feed-forward
    # ------------------------------------------------------------------
    def get_flatness_ff(self, t, mass=0.75, psi_ref=0.0):
        """
        Compute feed-forward control from differential flatness.

        Given the flat outputs [x(t), y(t), z(t), ψ(t)] and their derivatives,
        the quadrotor's thrust and angular velocity are uniquely determined.

        Args:
            t: query time (scalar).
            mass: vehicle mass (kg).
            psi_ref: desired yaw angle (rad).

        Returns:
            dict with keys:
                f_ff:     feed-forward total thrust (N)
                z_b:      desired body z-axis (unit vector)
                omega_ff: feed-forward angular velocity [ωx, ωy, ωz] (rad/s)
                a_total:  p̈ + g*e₃ (world frame)
        """
        p, v, a, j = self.get_reference(t)
        g = np.array([0.0, 0.0, 9.81])

        # Total acceleration (including gravity compensation)
        a_total = a + g
        a_norm = np.linalg.norm(a_total)
        if a_norm < 1e-6:
            a_total = g
            a_norm = 9.81

        # Feed-forward thrust
        f_ff = mass * a_norm

        # Desired body z-axis (thrust direction)
        z_b = a_total / a_norm

        # Desired body x, y axes from yaw
        x_c = np.array([np.cos(psi_ref), np.sin(psi_ref), 0.0])
        y_b = np.cross(z_b, x_c)
        y_norm = np.linalg.norm(y_b)
        if y_norm < 1e-6:
            y_b = np.array([0.0, 1.0, 0.0])
        else:
            y_b /= y_norm
        x_b = np.cross(y_b, z_b)

        # Feed-forward angular velocity (from jerk):
        # ω_x = -(j · y_b) / ||a_total||
        # ω_y =  (j · x_b) / ||a_total||
        # ω_z =  ψ̇ · (z_b · e₃) ≈ 0 (for constant yaw)
        h_omega = np.cross(j, z_b) / a_norm
        omega_ff = np.array([
            -np.dot(h_omega, y_b),
             np.dot(h_omega, x_b),
             0.0,
        ])

        return {
            'f_ff': f_ff,
            'z_b': z_b,
            'x_b': x_b,
            'y_b': y_b,
            'omega_ff': omega_ff,
            'a_total': a_total,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def print_info(self):
        """Print trajectory summary."""
        print(f"Waypoints: {self.n_waypoints}, Segments: {self.n_segments}")
        print(f"Total duration: {self.total_time:.2f}s")
        print(f"Segment durations: {self.T}")
        for i in range(self.n_segments):
            print(f"  Seg {i}: {self.waypoints[i]} -> {self.waypoints[i+1]}  "
                  f"({self.T[i]:.2f}s)")


# ------------------------------------------------------------------
# Demo / test
# ------------------------------------------------------------------
if __name__ == "__main__":
    waypoints = np.array([
        [0.0, 0.0, 1.0],
        [3.0, 2.0, 2.0],
        [6.0, -1.0, 3.0],
        [8.0, 3.0, 1.5],
        [10.0, 0.0, 1.0],
    ])
    traj = MinimumSnapTrajectory(waypoints, v_avg=2.0)
    traj.print_info()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    dt = 0.02
    ts, p, v, a, j = traj.get_total_reference(dt)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(2, 3, (1, 3), projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', linewidth=1.5, label='Reference')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
               c='r', s=60, marker='o', label='Waypoints')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(); ax.set_title('3D Reference Trajectory')

    for i, (label, data) in enumerate([('Position (m)', p), ('Velocity (m/s)', v),
                                        ('Acceleration (m/s²)', a)]):
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.plot(ts, data[:, 0], label='x')
        ax.plot(ts, data[:, 1], label='y')
        ax.plot(ts, data[:, 2], label='z')
        ax.set_xlabel('Time (s)'); ax.set_ylabel(label); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()
