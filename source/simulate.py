"""
Quadrotor Trajectory Tracking --- Paper-Grade Simulation.

Architecture (standard cascaded, matches literature):
    Waypoints -> Minimum Snap Trajectory
                    |
    +--- Differential Flatness Feed-Forward ---+
    |                                          |
    MPC (translational, 50Hz) --> Attitude Controller (250Hz) --> Quadrotor (full 6-DOF)
    output: a_des_mpc           output: [f, tau]                13D state
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

from trajectory_generator import MinimumSnapTrajectory
from mpc_controller import TranslationalMPC
from geometric_controller import GeometricAttitudeController
from quadrotor_dynamics import (
    QuadrotorDynamics,
    quat_multiply,
    quat_normalize,
    quat_to_rotation,
)
from disturbance_config import DisturbanceConfig


def _noisy_state_measurement(state, config, rng):
    measured = np.asarray(state, dtype=float).copy()
    if not config.measurement_noise:
        return measured

    measured[0:3] += rng.normal(0.0, config.position_noise_std, 3)
    measured[3:6] += rng.normal(0.0, config.velocity_noise_std, 3)
    measured[10:13] += rng.normal(0.0, config.body_rate_noise_std, 3)

    rotation_vector = rng.normal(
        0.0, np.deg2rad(config.attitude_noise_std_deg), 3)
    angle = np.linalg.norm(rotation_vector)
    if angle > 1e-12:
        axis = rotation_vector / angle
        delta_quaternion = np.r_[np.cos(0.5 * angle), axis * np.sin(0.5 * angle)]
        measured[6:10] = quat_normalize(
            quat_multiply(measured[6:10], delta_quaternion))
    return measured


def _apply_motor_mixer_limits(f_command, torque_command, mass, config):
    """Allocate wrench to four X-layout rotors, clip, and reconstruct it."""
    arm_xy = config.arm_length / np.sqrt(2.0)
    rotor_x = np.array([arm_xy, -arm_xy, -arm_xy, arm_xy])
    rotor_y = np.array([arm_xy, arm_xy, -arm_xy, -arm_xy])
    yaw_sign = np.array([1.0, -1.0, 1.0, -1.0])
    allocation = np.vstack((
        np.ones(4),
        rotor_y,
        -rotor_x,
        config.yaw_moment_ratio * yaw_sign,
    ))
    requested_wrench = np.r_[float(f_command), np.asarray(torque_command)]
    rotor_thrusts = np.linalg.solve(allocation, requested_wrench)
    max_rotor_thrust = config.max_rotor_thrust_ratio * mass * 9.81 / 4.0
    rotor_thrusts = np.clip(rotor_thrusts, 0.0, max_rotor_thrust)
    applied_wrench = allocation @ rotor_thrusts
    return float(applied_wrench[0]), applied_wrench[1:4], rotor_thrusts


# ======================================================================
def run_simulation(waypoints=None, v_avg=2.0,
                   mass=0.75, dt_mpc=0.02, dt_ctrl=0.004,
                   mpc_horizon=25,
                   Q_pos=40.0, Q_vel=10.0, Q_int=4.0, R_ctrl=0.05, a_max=12.0,
                   k_R=4.0, k_omega=1.5,
                   tau_f=0.05, tau_tau=0.03,
                   wind_const=None, gust_std=0.0,
                   sim_duration=None, seed=42,
                   use_ov_constraints=False,
                   ov_model_path=None,
                   ov_confidence_scale=1.96,
                   ov_min_radius=0.05,
                   ov_max_radius=None,
                   trajectory=None,
                   use_reference_error_mpc=False,
                   compensate_drag=False,
                   accel_time_constant=None,
                   Q_accel=0.0,
                   R_delta=0.0,
                   control_mode="cascaded_mpc",
                   max_tilt_deg=45.0,
                   thrust_max_ratio=2.5,
                   torque_max=(0.08, 0.08, 0.04),
                   nmpc_max_iterations=40,
                   use_imm_error_prediction=False,
                   imm_compensation_gain=1.0,
                   imm_max_position_increment=1.0,
                   imm_max_velocity_increment=1.0,
                   disturbances=None):
    """
    Full cascaded MPC + geometric attitude control simulation.

    Args:
        waypoints: (M,3) array of waypoints.
        v_avg: average flight speed (m/s).
        mass: vehicle mass (kg).
        dt_mpc: MPC timestep (s). 0.02 = 50 Hz.
        dt_ctrl: inner-loop timestep (s). 0.004 = 250 Hz.
        mpc_horizon: MPC prediction horizon (steps).
        Q_pos, Q_vel, Q_int, R_ctrl: MPC cost weights.
        a_max: acceleration command limit per axis (m/s^2).
        k_R, k_omega: attitude controller gains.
        tau_f, tau_tau: actuator time constants (s).
        wind_const: constant wind acceleration [wx, wy, wz] (m/s^2).
        gust_std: random gust std (m/s^2).
        sim_duration: override simulation duration (s).
        seed: random seed.
        use_ov_constraints: if True, predict a 3D OV tube and constrain MPC
            predicted positions to stay inside it.
        ov_model_path: optional path to plan_confidence best_model.pt.
        ov_confidence_scale: sigma multiplier. 1.96 gives a nominal 95% tube.
        ov_min_radius: minimum per-axis OV half-width in meters.
        ov_max_radius: optional per-axis OV half-width cap in meters.
        use_reference_error_mpc: solve for acceleration correction around the
            supplied time-parameterized reference.
        compensate_drag: cancel the drag term used by the simulation model.
        accel_time_constant: optional acceleration-response time constant used
            by the lag-aware translational MPC.
        Q_accel: acceleration-error weight in the lag-aware MPC.
        R_delta: command-increment weight in the lag-aware MPC.
        control_mode: ``"cascaded_mpc"`` for the translational MPC plus
            geometric attitude loop, or ``"full_nmpc"`` for direct thrust and
            torque optimization with the complete 17-state dynamics.
        max_tilt_deg: hard predicted tilt constraint in full NMPC mode.
        thrust_max_ratio: maximum commanded thrust divided by hover thrust.
        torque_max: per-axis torque-command bounds (Nm).
        nmpc_max_iterations: IPOPT iteration cap per NMPC update.
        use_imm_error_prediction: use an online IMM forecast of tracking-error
            trends to correct the full-NMPC predicted position and velocity.
        imm_compensation_gain: multiplier applied to the IMM trend forecast.
        imm_max_position_increment: per-axis clipping limit for the position
            trend correction passed to NMPC, in meters.
        imm_max_velocity_increment: per-axis clipping limit for the velocity
            trend correction passed to NMPC, in meters per second.
        disturbances: optional :class:`DisturbanceConfig` with independent
            model mismatch, wind, sensor noise, delay, initial offset, and
            motor saturation switches.

    Returns:
        dict with logged data and metrics.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    disturbances = DisturbanceConfig() if disturbances is None else disturbances
    enabled_disturbances = disturbances.enabled_names()
    effective_wind_const = (
        disturbances.constant_wind_accel if disturbances.wind else wind_const)
    effective_gust_std = disturbances.gust_std if disturbances.wind else gust_std
    g = 9.81
    psi_ref = 0.0

    # ---- Waypoints ----
    if waypoints is None:
        waypoints = np.array([
            [0.0, 0.0, 1.0],
            [3.0, 2.0, 2.0],
            [6.0, -1.0, 3.0],
            [8.0, 3.0, 1.5],
            [10.0, 0.0, 1.0],
        ])

    # ---- Reference trajectory ----
    traj = trajectory if trajectory is not None else MinimumSnapTrajectory(waypoints, v_avg=v_avg)
    traj.print_info()

    # ---- Controllers ----
    if control_mode == "full_nmpc":
        from full_nmpc_controller import FullDynamicsNMPC

        mpc = FullDynamicsNMPC(
            dt=dt_mpc,
            horizon=mpc_horizon,
            mass=mass,
            tau_f=tau_f,
            tau_tau=tau_tau,
            max_tilt_deg=max_tilt_deg,
            thrust_max_ratio=thrust_max_ratio,
            torque_max=torque_max,
            max_iterations=nmpc_max_iterations,
        )
        att = None
        print("Full-dynamics NMPC")
        print(f"  state: [p, v, quaternion, omega, thrust_actual, torque_actual]")
        print(f"  input: [thrust_command, torque_command]")
        print(f"  horizon: {mpc_horizon} steps ({mpc_horizon * dt_mpc:.2f} s)")
        print(f"  maximum tilt: {max_tilt_deg:.1f} deg")
        print(f"  thrust range: [0, {mpc.thrust_max:.2f}] N")
        print(f"  torque limits: {np.asarray(torque_max)} Nm")
    elif control_mode == "cascaded_mpc":
        mpc = TranslationalMPC(dt=dt_mpc, horizon=mpc_horizon,
                               Q_pos=Q_pos, Q_vel=Q_vel, Q_int=Q_int,
                               R_ctrl=R_ctrl, a_max=a_max,
                               accel_time_constant=accel_time_constant,
                               Q_accel=Q_accel, R_delta=R_delta)
        if accel_time_constant is not None:
            print("Lag-aware translational MPC")
            print(f"  acceleration time constant: {accel_time_constant:.3f} s")
            print(f"  acceleration error weight: {Q_accel:.3f}")
            print(f"  command increment weight: {R_delta:.3f}")
        att = GeometricAttitudeController(mass=mass, k_R=k_R, k_omega=k_omega)
    else:
        raise ValueError(f"Unknown control_mode: {control_mode}")

    print("Simulation disturbances")
    print(f"  enabled: {', '.join(enabled_disturbances) if enabled_disturbances else 'none (ideal)'}")

    # ---- Quadrotor model (full 6-DOF) ----
    ratio = max(1, int(np.ceil(dt_mpc / dt_ctrl)))
    dt_inner = dt_mpc / ratio
    plant_mass = mass
    plant_inertia = None
    plant_drag = None
    if disturbances.model_mismatch:
        plant_mass = mass * disturbances.actual_mass_scale
        plant_inertia = np.diag([0.0025, 0.0025, 0.005]) * disturbances.actual_inertia_scale
        plant_drag = np.array([0.3, 0.3, 0.2]) * disturbances.actual_drag_scale
        print(f"  nominal/actual mass: {mass:.3f}/{plant_mass:.3f} kg")
        print(f"  inertia scale: {disturbances.actual_inertia_scale:.3f}")
        print(f"  drag scale: {disturbances.actual_drag_scale:.3f}")
    quad = QuadrotorDynamics(
        mass=plant_mass,
        inertia=plant_inertia,
        drag_coeff=plant_drag,
        dt=dt_inner,
        tau_f=tau_f,
        tau_tau=tau_tau,
    )

    # ---- Initialise at reference attitude via differential flatness ----
    p0, v0, a0, j0 = traj.get_reference(0.0)
    # Full NMPC always predicts drag, so its initial attitude must start from
    # the same force balance even when the cascaded-controller flag is off.
    nominal_drag = np.diag([0.3, 0.3, 0.2])
    a0_init = (a0 + nominal_drag @ v0
               if (compensate_drag or control_mode == "full_nmpc") else a0)
    quad.set_state_from_reference(p0, v0, a0_init, psi_ref=psi_ref)
    if disturbances.initial_state_offset:
        quad.state[0:3] += np.asarray(disturbances.initial_position_offset)
        quad.state[3:6] += np.asarray(disturbances.initial_velocity_offset)
        print(f"  initial position offset: {disturbances.initial_position_offset} m")
        print(f"  initial velocity offset: {disturbances.initial_velocity_offset} m/s")
    e_int = np.zeros(3)
    imm_error_predictor = None
    if use_imm_error_prediction:
        if control_mode != "full_nmpc":
            raise ValueError(
                "IMM error prediction is currently supported only in full_nmpc mode"
            )
        from imm_error_predictor import IMMTrackingErrorPredictor

        imm_error_predictor = IMMTrackingErrorPredictor(dt=dt_mpc)
        print("Online IMM tracking-error prediction enabled")
        print(f"  models: {', '.join(imm_error_predictor.MODEL_NAMES)}")
        print(f"  compensation gain: {imm_compensation_gain:.3f}")
        print(
            "  increment limits: "
            f"position={imm_max_position_increment:.3f} m, "
            f"velocity={imm_max_velocity_increment:.3f} m/s"
        )

    delay_steps = (max(1, int(round(disturbances.delay_seconds / dt_inner)))
                   if disturbances.control_delay else 0)
    initial_command = (float(quad.f_actual), quad.tau_actual.copy())
    command_delay = deque(
        [(initial_command[0], initial_command[1].copy()) for _ in range(delay_steps)])
    if disturbances.control_delay:
        print(f"  command delay: {delay_steps * dt_inner:.3f} s ({delay_steps} inner steps)")
    if disturbances.motor_limits:
        rotor_limit = disturbances.max_rotor_thrust_ratio * plant_mass * g / 4.0
        print(f"  rotor thrust range: [0, {rotor_limit:.3f}] N per motor")

    total_time = sim_duration if sim_duration is not None else traj.total_time
    n_mpc = int(total_time / dt_mpc) + 1

    ov_sigma = None
    ov_lower = None
    ov_upper = None
    if use_ov_constraints:
        import sys
        from pathlib import Path

        root_dir = Path(__file__).resolve().parents[1]
        multiple_ov_dir = root_dir / "Multiple_OV"
        if str(multiple_ov_dir) not in sys.path:
            sys.path.insert(0, str(multiple_ov_dir))

        from ov_predictor import build_ov_bounds

        ov_ts = np.arange(n_mpc) * dt_mpc
        center_path, _, _, _ = traj.get_reference(ov_ts)
        ov_sigma, ov_lower, ov_upper = build_ov_bounds(
            center_path,
            model_path=ov_model_path or None,
            confidence_scale=ov_confidence_scale,
            min_radius=ov_min_radius,
            max_radius=ov_max_radius,
        )
        print("\nOV constraints enabled")
        print(f"  confidence scale: {ov_confidence_scale:.2f}")
        print(f"  mean sigma: {ov_sigma.mean(axis=0)} m")
        print(f"  minimum radius: {ov_min_radius} m")
        if ov_max_radius is not None:
            print(f"  maximum radius cap: {ov_max_radius} m")

    # ---- Logging ----
    log = {
        't':         np.zeros(n_mpc),
        'pos':       np.zeros((n_mpc, 3)),
        'vel':       np.zeros((n_mpc, 3)),
        'pos_measured': np.zeros((n_mpc, 3)),
        'vel_measured': np.zeros((n_mpc, 3)),
        'euler':     np.zeros((n_mpc, 3)),
        'omega':     np.zeros((n_mpc, 3)),
        'pref':      np.zeros((n_mpc, 3)),
        'vref':      np.zeros((n_mpc, 3)),
        'aref':      np.zeros((n_mpc, 3)),
        'accel_actual': np.zeros((n_mpc, 3)),
        'a_des_mpc': np.zeros((n_mpc, 3)),
        'f_cmd':     np.zeros(n_mpc),
        'tau_cmd':   np.zeros((n_mpc, 3)),
        'f_cmd_requested': np.zeros(n_mpc),
        'tau_cmd_requested': np.zeros((n_mpc, 3)),
        'rotor_thrusts': np.full((n_mpc, 4), np.nan),
        'f_actual':  np.zeros(n_mpc),
        'e_int':     np.zeros((n_mpc, 3)),
        'solver_time': np.zeros(n_mpc),
        'solver_success': np.zeros(n_mpc, dtype=bool),
        'imm_position_error': np.zeros((n_mpc, 3)),
        'imm_velocity_error': np.zeros((n_mpc, 3)),
        'imm_position_increment_1': np.zeros((n_mpc, 3)),
        'imm_position_increment_end': np.zeros((n_mpc, 3)),
        'imm_position_sigma_end': np.zeros((n_mpc, 3)),
        'imm_model_probabilities': np.zeros((n_mpc, 3)),
    }

    # ---- Main loop (outer: MPC rate) ----
    n_logged = 0
    for k in range(n_mpc):
        t = k * dt_mpc
        if t > total_time:
            break

        # Reference at current time
        p_ref, v_ref, a_ref, j_ref = traj.get_reference(t)
        # The plant state at loop entry is also at time t. Keep this snapshot
        # aligned with the current reference for logging and integral update.
        position_now = quad.position.copy()
        velocity_now = quad.velocity.copy()
        euler_now = quad.euler.copy()
        omega_now = quad.omega.copy()
        f_actual_now = float(quad.f_actual)
        e_int_now = e_int.copy()
        measured_state = _noisy_state_measurement(quad.state, disturbances, rng)
        a_actual_now = (
            quad.rotation_matrix @ np.array([0.0, 0.0, f_actual_now]) / quad.mass
            + np.array([0.0, 0.0, -g])
            - quad.D @ velocity_now
        )
        if effective_wind_const is not None:
            a_actual_now = a_actual_now + np.asarray(effective_wind_const)

        # MPC horizon reference sequence
        pref_seq, vref_seq, aref_seq, jref_seq = traj.get_reference_sequence(
            t, mpc_horizon, dt_mpc)

        imm_forecast = None
        position_error_increment = None
        velocity_error_increment = None
        if imm_error_predictor is not None:
            position_error = measured_state[0:3] - p_ref
            velocity_error = measured_state[3:6] - v_ref
            imm_error_predictor.update(position_error, velocity_error)
            imm_forecast = imm_error_predictor.forecast(mpc_horizon)
            position_trend = np.clip(
                imm_compensation_gain * imm_forecast.position_increment,
                -float(imm_max_position_increment),
                float(imm_max_position_increment),
            )
            velocity_trend = np.clip(
                imm_compensation_gain * imm_forecast.velocity_increment,
                -float(imm_max_velocity_increment),
                float(imm_max_velocity_increment),
            )
            # The current state already contains the measured error.  Only
            # future trend increments are supplied to the nonlinear predictor.
            position_error_increment = np.vstack((np.zeros(3), position_trend))
            velocity_error_increment = np.vstack((np.zeros(3), velocity_trend))

        pos_lower_seq = None
        pos_upper_seq = None
        if use_ov_constraints:
            idx = np.minimum(k + 1 + np.arange(mpc_horizon), len(ov_lower) - 1)
            pos_lower_seq = ov_lower[idx]
            pos_upper_seq = ov_upper[idx]

        if control_mode == "full_nmpc":
            full_state = np.concatenate([
                measured_state, [quad.f_actual], quad.tau_actual])
            p_horizon = np.vstack((p_ref, pref_seq))
            v_horizon = np.vstack((v_ref, vref_seq))
            a_horizon = np.vstack((a_ref, aref_seq))
            f_requested, tau_requested, mpc_info = mpc.compute_control(
                full_state,
                p_horizon,
                v_horizon,
                a_horizon,
                position_lower=pos_lower_seq,
                position_upper=pos_upper_seq,
                position_error_increment=position_error_increment,
                velocity_error_increment=velocity_error_increment,
            )
            a_des_mpc = a_actual_now
        else:
            # ---- Translational MPC: desired acceleration correction ----
            x0 = np.concatenate([measured_state[0:3], measured_state[3:6], e_int])
            tracking_kwargs = {}
            if use_reference_error_mpc:
                tracking_kwargs = {
                    'p_ref': p_ref,
                    'v_ref': v_ref,
                    'a_ref': a_ref,
                    'aref_seq': aref_seq,
                    'a_actual': a_actual_now,
                }
            a_des_mpc, mpc_info = mpc.compute_control(
                x0, pref_seq, vref_seq,
                pos_lower_seq=pos_lower_seq,
                pos_upper_seq=pos_upper_seq,
                **tracking_kwargs,
            )

            # Desired translational acceleration; attitude control adds gravity.
            if getattr(traj, "use_accel_feedforward", True):
                a_total_des = a_ref + a_des_mpc
            else:
                a_total_des = a_des_mpc
            if compensate_drag:
                a_total_des = a_total_des + quad.D @ quad.velocity

        if not mpc_info.get('success', False) and (k < 5 or k % 100 == 0):
            print(f"Warning: MPC solve failed at t={t:.2f}s: {mpc_info.get('status')}")

        if imm_forecast is not None:
            log['imm_position_error'][k] = position_error
            log['imm_velocity_error'][k] = velocity_error
            log['imm_position_increment_1'][k] = position_error_increment[1]
            log['imm_position_increment_end'][k] = position_error_increment[-1]
            log['imm_position_sigma_end'][k] = imm_forecast.position_sigma[-1]
            log['imm_model_probabilities'][k] = (
                imm_forecast.model_probabilities
            )

        # ---- Inner loop: attitude control at dt_ctrl ----
        rotor_thrusts = np.full(4, np.nan)
        for _ in range(ratio):
            if control_mode == "cascaded_mpc":
                attitude_measurement = _noisy_state_measurement(
                    quad.state, disturbances, rng)
                f_requested, tau_requested, _ = att.compute_control(
                    a_total_des, psi_des=psi_ref,
                    q_current=attitude_measurement[6:10],
                    omega_current=attitude_measurement[10:13],
                    a_des_dot=j_ref)

            if delay_steps > 0:
                command_delay.append((float(f_requested), np.asarray(tau_requested).copy()))
                f_applied, tau_applied = command_delay.popleft()
            else:
                f_applied = float(f_requested)
                tau_applied = np.asarray(tau_requested).copy()

            if disturbances.motor_limits:
                f_applied, tau_applied, rotor_thrusts = _apply_motor_mixer_limits(
                    f_applied, tau_applied, quad.mass, disturbances)

            # Wind disturbance
            wind = np.zeros(3)
            if effective_wind_const is not None:
                wind += np.asarray(effective_wind_const)
            if effective_gust_std > 0:
                wind += rng.normal(0.0, effective_gust_std, 3)

            quad.step(f_applied, tau_applied, wind_accel=wind)

        f_cmd = f_applied
        tau_cmd = tau_applied

        # ---- Update integral error ----
        if control_mode == "cascaded_mpc":
            e_int = mpc.update_integral(e_int, measured_state[0:3], p_ref)

        # ---- Log ----
        log['t'][k]         = t
        log['pos'][k]       = position_now
        log['vel'][k]       = velocity_now
        log['pos_measured'][k] = measured_state[0:3]
        log['vel_measured'][k] = measured_state[3:6]
        log['euler'][k]     = euler_now
        log['omega'][k]     = omega_now
        log['pref'][k]      = p_ref
        log['vref'][k]      = v_ref
        log['aref'][k]      = a_ref
        log['accel_actual'][k] = a_actual_now
        log['a_des_mpc'][k] = a_des_mpc
        log['f_cmd'][k]     = f_cmd
        log['tau_cmd'][k]   = tau_cmd
        log['f_cmd_requested'][k] = f_requested
        log['tau_cmd_requested'][k] = tau_requested
        log['rotor_thrusts'][k] = rotor_thrusts
        log['f_actual'][k]  = f_actual_now
        log['e_int'][k]     = e_int_now
        log['solver_time'][k] = mpc_info.get('solve_time', 0.0)
        log['solver_success'][k] = mpc_info.get('success', False)
        n_logged = k + 1

        if control_mode == "full_nmpc" and (k == 0 or (k + 1) % 200 == 0):
            print(f"  NMPC step {k + 1}/{n_mpc}, t={t:.2f}s, "
                  f"solve={log['solver_time'][k] * 1000:.1f} ms")

    # ---- Trim logs ----
    for key in log:
        log[key] = log[key][:n_logged]

    # ---- Metrics ----
    pos_error = log['pos'] - log['pref']
    error_norm = np.sqrt(np.sum(pos_error ** 2, axis=1))
    rmse = np.sqrt(np.mean(error_norm ** 2))
    max_e = np.max(error_norm)
    mean_e = np.mean(error_norm)

    print(f"\n{'='*55}")
    print(f"Tracking Performance")
    print(f"  RMSE = {rmse*100:.2f} cm    Mean = {mean_e*100:.2f} cm    Max = {max_e*100:.2f} cm")
    if control_mode == "full_nmpc":
        print(f"  NMPC solve mean = {np.mean(log['solver_time'])*1000:.1f} ms, "
              f"max = {np.max(log['solver_time'])*1000:.1f} ms, "
              f"success = {100*np.mean(log['solver_success']):.1f}%")
    if imm_error_predictor is not None and n_logged:
        final_probabilities = log['imm_model_probabilities'][-1]
        print(
            "  IMM final probabilities "
            f"(CV/CA/DM) = {np.round(final_probabilities, 3)}"
        )
    print(f"{'='*55}")

    log['pos_error']  = pos_error
    log['error_norm'] = error_norm
    log['rmse']       = rmse
    log['max_error']  = max_e
    log['mean_error'] = mean_e
    log['traj']       = traj
    log['ov_sigma']   = ov_sigma
    log['ov_lower']   = ov_lower
    log['ov_upper']   = ov_upper
    log['control_mode'] = control_mode
    log['disturbances'] = disturbances
    log['nominal_mass'] = mass
    log['plant_mass'] = plant_mass
    log['use_imm_error_prediction'] = imm_error_predictor is not None
    log['imm_compensation_gain'] = float(imm_compensation_gain)
    return log


# ======================================================================
# Plotting
# ======================================================================
def plot_results(log, title_prefix="", save_prefix=None, show=True):
    """Comprehensive visualisation of tracking performance."""
    traj = log['traj']
    wp = traj.waypoints

    # ---- Figure 1: 3D + time-domain ----
    fig1 = plt.figure(figsize=(16, 12))

    ax = fig1.add_subplot(2, 3, (1, 3), projection='3d')
    ax.plot(log['pref'][:, 0], log['pref'][:, 1], log['pref'][:, 2],
            'b-', lw=2.0, alpha=0.7, label='Reference')
    tracked_label = ('NMPC Tracked' if log.get('control_mode') == 'full_nmpc'
                     else 'MPC Tracked')
    ax.plot(log['pos'][:, 0], log['pos'][:, 1], log['pos'][:, 2],
            'r--', lw=1.5, label=tracked_label)
    if log.get('ov_lower') is not None and log.get('ov_upper') is not None:
        ov_lower = log['ov_lower'][:len(log['t'])]
        ov_upper = log['ov_upper'][:len(log['t'])]
        ax.plot(ov_lower[:, 0], ov_lower[:, 1], ov_lower[:, 2],
                color='gray', lw=0.7, alpha=0.4, label='OV lower')
        ax.plot(ov_upper[:, 0], ov_upper[:, 1], ov_upper[:, 2],
                color='gray', lw=0.7, alpha=0.4, label='OV upper')
    ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2],
               c='k', s=80, marker='o', zorder=5,
               label='Path markers (visual only)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    z_all = np.concatenate((log['pref'][:, 2], log['pos'][:, 2]))
    if np.ptp(z_all) < 1.0:
        z_center = 0.5 * (np.min(z_all) + np.max(z_all))
        ax.set_zlim(z_center - 0.5, z_center + 0.5)
    ax.set_title(f'{title_prefix}3D Trajectory'); ax.legend(loc='upper left')

    for idx, (lbl, ref, act) in enumerate([
        ('Position (m)', log['pref'], log['pos']),
        ('Velocity (m/s)', log['vref'], log['vel']),
        ('Attitude (rad)', np.zeros_like(log['euler']), log['euler']),
    ]):
        ax = fig1.add_subplot(2, 3, 4 + idx)
        colors = ['r', 'g', 'b']
        for d in range(3):
            ax.plot(log['t'], act[:, d], color=colors[d], lw=1.0)
            if idx < 2:
                ax.plot(log['t'], ref[:, d], color=colors[d], ls='--', lw=0.6, alpha=0.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)

    suptitle = (f'{title_prefix}'
                f'RMSE={log["rmse"]*100:.2f}cm  '
                f'Max={log["max_error"]*100:.2f}cm  '
                f'Mean={log["mean_error"]*100:.2f}cm')
    plt.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()

    # ---- Figure 2: Error analysis ----
    fig2, axes = plt.subplots(2, 2, figsize=(14, 9))

    pe = log['pos_error']
    for d, c, lb in [(0, 'r', 'ex'), (1, 'g', 'ey'), (2, 'b', 'ez')]:
        axes[0, 0].plot(log['t'], pe[:, d], color=c, lw=0.8, label=lb)
    axes[0, 0].axhline(y=0, color='k', lw=0.5)
    axes[0, 0].set_title('Position Error per Axis'); axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Error (m)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(log['t'], log['error_norm'], 'k-', lw=0.8)
    axes[0, 1].axhline(y=log['mean_error'], color='b', ls='--',
                       label=f'Mean={log["mean_error"]*100:.2f}cm')
    axes[0, 1].axhline(y=log['rmse'], color='r', ls='--',
                       label=f'RMSE={log["rmse"]*100:.2f}cm')
    axes[0, 1].set_title('3D Position Error Norm'); axes[0, 1].legend()
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('||Error|| (m)')
    axes[0, 1].grid(True, alpha=0.3)

    config = log.get('disturbances', DisturbanceConfig())
    if config.measurement_noise:
        measurement_error = log['pos_measured'] - log['pos']
        for d, c, lb in [(0, 'r', 'nx'), (1, 'g', 'ny'), (2, 'b', 'nz')]:
            axes[1, 0].plot(
                log['t'], measurement_error[:, d], color=c, lw=0.7, label=lb)
        axes[1, 0].set_title('Position Measurement Noise')
        axes[1, 0].set_ylabel('Measured - true (m)')
    elif config.motor_limits:
        for motor in range(4):
            axes[1, 0].plot(
                log['t'], log['rotor_thrusts'][:, motor], lw=0.7,
                label=f'rotor {motor + 1}')
        rotor_limit = config.max_rotor_thrust_ratio * log['plant_mass'] * 9.81 / 4.0
        axes[1, 0].axhline(rotor_limit, color='k', ls='--', lw=0.8, label='limit')
        axes[1, 0].set_title('Allocated Rotor Thrusts')
        axes[1, 0].set_ylabel('Thrust (N)')
    elif config.control_delay:
        thrust_gap = log['f_cmd_requested'] - log['f_cmd']
        torque_gap = np.linalg.norm(
            log['tau_cmd_requested'] - log['tau_cmd'], axis=1)
        axes[1, 0].plot(log['t'], thrust_gap, lw=0.8, label='thrust gap (N)')
        axes[1, 0].plot(log['t'], torque_gap, lw=0.8, label='torque gap (Nm)')
        axes[1, 0].set_title('Requested vs Delayed Command Gap')
        axes[1, 0].set_ylabel('Command difference')
    elif log.get('control_mode') == 'full_nmpc':
        velocity_error = np.linalg.norm(log['vel'] - log['vref'], axis=1)
        axes[1, 0].plot(log['t'], velocity_error, color='tab:purple', lw=0.8)
        axes[1, 0].set_title('3D Velocity Error Norm')
        axes[1, 0].set_ylabel('||Velocity error|| (m/s)')
    else:
        for d, c, lb in [(0, 'r', 'eix'), (1, 'g', 'eiy'), (2, 'b', 'eiz')]:
            axes[1, 0].plot(
                log['t'], log['e_int'][:, d], color=c, lw=0.8, label=lb)
        axes[1, 0].set_title('Integral Error State')
        axes[1, 0].set_ylabel('Integral')
    axes[1, 0].axhline(y=0, color='k', lw=0.5)
    handles, _ = axes[1, 0].get_legend_handles_labels()
    if handles:
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].grid(True, alpha=0.3)

    if not np.allclose(log['f_cmd_requested'], log['f_cmd']):
        axes[1, 1].plot(
            log['t'], log['f_cmd_requested'], color='tab:orange', lw=0.8,
            label='f_requested')
    axes[1, 1].plot(log['t'], log['f_cmd'], 'b-', lw=1.0, label='f_applied')
    axes[1, 1].plot(log['t'], log['f_actual'], 'r--', lw=0.8, label='f_actual')
    hover = log.get('plant_mass', 0.75) * 9.81
    axes[1, 1].axhline(y=hover, color='gray', ls=':', lw=0.8, label=f'Hover ({hover:.1f}N)')
    axes[1, 1].set_title('Thrust Command vs Actual'); axes[1, 1].legend()
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Thrust (N)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix}Error & Control Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig3 = None
    if log.get('use_imm_error_prediction', False):
        fig3, imm_axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        colors = ['r', 'g', 'b']
        labels = ['x', 'y', 'z']
        for axis, color, label in zip(range(3), colors, labels):
            imm_axes[0].plot(
                log['t'], log['imm_position_error'][:, axis],
                color=color, lw=0.9, label=f'e_{label}')
            imm_axes[1].plot(
                log['t'], log['imm_position_increment_end'][:, axis],
                color=color, lw=0.9, label=f'forecast increment {label}')
        imm_axes[0].axhline(0.0, color='k', lw=0.5)
        imm_axes[0].set_ylabel('Position error (m)')
        imm_axes[0].set_title('Measured Online Tracking Error')
        imm_axes[0].legend(ncol=3)
        imm_axes[1].axhline(0.0, color='k', lw=0.5)
        imm_axes[1].set_ylabel('End-horizon increment (m)')
        imm_axes[1].set_title(
            'IMM Trend Increment Applied to NMPC Prediction')
        imm_axes[1].legend(ncol=3)
        for model_index, model_name in enumerate(('CV', 'CA', 'DM')):
            imm_axes[2].plot(
                log['t'], log['imm_model_probabilities'][:, model_index],
                lw=1.0, label=model_name)
        imm_axes[2].set_ylim(-0.02, 1.02)
        imm_axes[2].set_ylabel('Probability')
        imm_axes[2].set_xlabel('Time (s)')
        imm_axes[2].set_title('IMM Model Probabilities')
        imm_axes[2].legend(ncol=3)
        for imm_axis in imm_axes:
            imm_axis.grid(True, alpha=0.3)
        fig3.suptitle(
            f'{title_prefix}Online IMM Error-Trend Prediction',
            fontsize=13, fontweight='bold')
        fig3.tight_layout()

    if save_prefix is not None:
        fig1.savefig(f"{save_prefix}_trajectory.png", dpi=150)
        fig2.savefig(f"{save_prefix}_analysis.png", dpi=150)
        if fig3 is not None:
            fig3.savefig(f"{save_prefix}_imm_prediction.png", dpi=150)
    if show:
        plt.show()
    return fig1, fig2


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    # ---- Waypoints (edit here to change your flight path) ----
    waypoints = np.array([
        [0.0, 0.0, 1.0],
        [3.0, 2.0, 2.0],
        [6.0, -1.0, 3.0],
        [8.0, 3.0, 1.5],
        [10.0, 0.0, 1.0],
    ])

    # Common parameters
    kw = dict(waypoints=waypoints, v_avg=2.0, mass=0.75,
              dt_mpc=0.02, dt_ctrl=0.004, mpc_horizon=25,
              Q_pos=40.0, Q_vel=10.0, Q_int=4.0, R_ctrl=0.05, a_max=12.0,
              k_R=4.0, k_omega=1.5, seed=42)

    # ================================================================
    print("=" * 60)
    print("Scenario 1: No disturbance (ideal)")
    print("=" * 60)
    log1 = run_simulation(**kw, wind_const=None, gust_std=0.0)

    # ================================================================
    print("\n" + "=" * 60)
    print("Scenario 2: Disturbance + integral action")
    print("=" * 60)
    log2 = run_simulation(**kw, wind_const=[0.6, -0.4, 0.2], gust_std=0.3)

    # ================================================================
    print("\n" + "=" * 60)
    print("Scenario 3: Disturbance, NO integral (Q_int=0)")
    print("=" * 60)
    log3 = run_simulation(**{**kw, 'Q_int': 0.0},
                          wind_const=[0.6, -0.4, 0.2], gust_std=0.3)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<48} {'RMSE':<10} {'Max':<10} {'Mean':<10}")
    print("-" * 78)
    for name, L in [("1. No disturbance", log1),
                     ("2. Disturbance + integral", log2),
                     ("3. Disturbance, NO integral", log3)]:
        print(f"  {name:<46} {L['rmse']*100:>5.1f} cm  "
              f"{L['max_error']*100:>5.1f} cm  {L['mean_error']*100:>5.1f} cm")

    # ---- Plots (comment out to reduce figures) ----
    plot_results(log1, title_prefix="[No Disturbance] ")
    plot_results(log2, title_prefix="[With Disturbance] ")
    plot_results(log3, title_prefix="[Disturb, No Integral] ")
