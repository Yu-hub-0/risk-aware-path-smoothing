"""Closed-loop E1 evaluation with the copied 17-state full NMPC simulator.

Run only accepted geometric references. All attempted case/method pairs stay in
the output, including rejected paths and failed solver runs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import make_interp_spline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e1 import (ROOT, METHODS, RISK_LIMIT, grid_path_xy, make_case,
                    retime, risk_at, solve_one)

sys.path.insert(0, str(ROOT / "source"))
from B_Astar_Algorithm import run_b_astar  # noqa: E402
from simulate import run_simulation  # noqa: E402


class SplineReference:
    def __init__(self, curve, altitude: float = 1.0):
        from run_e1 import V_MAX, A_T_MAX, A_LAT_MAX
        _, (times, _, params, _, _) = retime(curve)
        self.curve = curve
        self.altitude = float(altitude)
        self.base_total = float(times[-1])
        self.clock = make_interp_spline(
            times, params, k=5,
            bc_type=([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)]))
        probe = np.linspace(0, self.base_total, max(3000, 5 * len(times)))
        u = self.clock(probe)
        du = self.clock.derivative(1)(probe)
        ddu = self.clock.derivative(2)(probe)
        if np.min(du) < -1e-6 or np.min(u) < -1e-6 or np.max(u) > 1 + 1e-6:
            raise ValueError("Non-monotone continuous time law")
        d1, d2 = curve.derivative(1)(u), curve.derivative(2)(u)
        vel = d1 * du[:, None]
        acc = d2 * (du ** 2)[:, None] + d1 * ddu[:, None]
        speed = np.linalg.norm(vel, axis=1)
        tangent = d1 / np.linalg.norm(d1, axis=1)[:, None]
        a_t = np.abs(np.sum(acc * tangent, axis=1))
        a_lat = np.abs(vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / np.maximum(speed, 1e-9)
        self.scale = max(1.0, float(np.max(speed) / V_MAX),
                         float(np.sqrt(np.max(a_t) / A_T_MAX)),
                         float(np.sqrt(np.max(a_lat) / A_LAT_MAX))) * 1.02
        self.total_time = self.base_total * self.scale
        self.kinematic_check = dict(max_speed_mps=float(np.max(speed) / self.scale),
                                    max_tangential_acc_mps2=float(np.max(a_t) / self.scale ** 2),
                                    max_lateral_acc_mps2=float(np.max(a_lat) / self.scale ** 2))
        sampled = self.get_reference(np.linspace(0, self.total_time, 16))[0]
        self.waypoints = sampled
        self.p = sampled

    def print_info(self):
        print(f"C2 risk-screened B-spline reference: {self.total_time:.2f} s")

    def get_reference(self, t):
        scalar = np.ndim(t) == 0
        ts = np.atleast_1d(np.asarray(t, float))
        base_t = np.clip(ts / self.scale, 0, self.base_total)
        u = self.clock(base_t)
        du = self.clock.derivative(1)(base_t) / self.scale
        ddu = self.clock.derivative(2)(base_t) / self.scale ** 2
        dddu = self.clock.derivative(3)(base_t) / self.scale ** 3
        s0, s1 = self.curve(u), self.curve.derivative(1)(u)
        s2, s3 = self.curve.derivative(2)(u), self.curve.derivative(3)(u)
        p = np.column_stack((s0, np.full(len(ts), self.altitude)))
        v = np.column_stack((s1 * du[:, None], np.zeros(len(ts))))
        a = np.column_stack((s2 * (du ** 2)[:, None] + s1 * ddu[:, None],
                             np.zeros(len(ts))))
        j = np.column_stack((s3 * (du ** 3)[:, None] +
                             3 * s2 * (du * ddu)[:, None] + s1 * dddu[:, None],
                             np.zeros(len(ts))))
        v[ts > self.total_time] = 0.0
        a[ts > self.total_time] = 0.0
        j[ts > self.total_time] = 0.0
        return tuple(arr[0] if scalar else arr for arr in (p, v, a, j))

    def get_reference_sequence(self, t0, horizon, dt):
        return self.get_reference(t0 + np.arange(1, horizon + 1) * dt)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    rows = []
    for scenario_index, kind in enumerate(("corner", "s_bend", "narrow")):
        for index in range(args.n_per_scenario):
            if ((scenario_index * args.n_per_scenario + index) % args.num_shards
                    != args.shard_id):
                continue
            risk, start, goal = make_case(kind, index, args.seed)
            try:
                dense, sparse = run_b_astar(risk, start, goal,
                                            risk_threshold=RISK_LIMIT, target_nodes=None)
            except Exception as exc:
                for method in METHODS[1:]:
                    rows.append(dict(scenario=kind, case=index, method=method,
                                     status=f"search_failed:{type(exc).__name__}"))
                continue
            path = grid_path_xy(sparse)
            for method in METHODS[1:]:
                geom, _, curve = solve_one(method, path, grid_path_xy(dense), risk)
                item = dict(scenario=kind, case=index, method=method,
                            geometry_status=geom["status"])
                if geom["status"] != "accepted" or curve is None:
                    item["status"] = "not_executed_reference_rejected"
                    rows.append(item)
                    continue
                try:
                    reference = SplineReference(curve)
                except (ValueError, FloatingPointError) as exc:
                    item["status"] = f"reference_time_law_failed:{type(exc).__name__}"
                    rows.append(item)
                    continue
                if args.max_duration_s and reference.total_time > args.max_duration_s:
                    item["status"] = "not_executed_duration_cap"
                    rows.append(item)
                    continue
                stdout_path = log_dir / f"{kind}_{index:02d}_{method}.txt"
                t0 = perf_counter()
                try:
                    with stdout_path.open("w", encoding="utf-8") as stream:
                        with contextlib.redirect_stdout(stream):
                            log = run_simulation(trajectory=reference,
                                                 dt_mpc=0.05, dt_ctrl=0.005,
                                                 mpc_horizon=20,
                                                 nmpc_max_iterations=40,
                                                 control_mode="full_nmpc",
                                                 use_ov_constraints=False,
                                                 use_imm_error_prediction=False,
                                                 seed=args.seed + index)
                    pos = log["pos"]
                    outside = ((pos[:, :2] < 0) | (pos[:, :2] > 39)).any(axis=1)
                    actual_risk = risk_at(pos[:, :2], risk)
                    item.update(status="executed", duration_s=reference.total_time,
                                wall_s=perf_counter() - t0, rmse_m=float(log["rmse"]),
                                **reference.kinematic_check,
                                max_error_m=float(log["max_error"]),
                                actual_risk_violation_fraction=float(np.mean(actual_risk > RISK_LIMIT)),
                                out_of_map_fraction=float(np.mean(outside)),
                                solver_success_fraction=float(np.mean(log["solver_success"])),
                                solver_deadline_fraction=float(np.mean(log["solver_time"] > 0.05)),
                                solver_p95_ms=float(np.percentile(log["solver_time"], 95) * 1000),
                                terminal_error_m=float(np.linalg.norm(pos[-1] - reference.p[-1])))
                    np.savez_compressed(log_dir / f"{kind}_{index:02d}_{method}.npz",
                                        time_s=log["t"], actual_pos=pos, ref_pos=log["pref"],
                                        actual_vel=log["vel"], ref_vel=log["vref"],
                                        solver_time_s=log["solver_time"],
                                        solver_success=log["solver_success"])
                except Exception as exc:
                    item.update(status=f"simulation_failed:{type(exc).__name__}",
                                wall_s=perf_counter() - t0)
                    with stdout_path.open("a", encoding="utf-8") as stream:
                        stream.write(f"\nEXCEPTION: {exc!r}\n")
                rows.append(item)
                print(kind, index, method, item["status"],
                      f"{item.get('wall_s', 0):.1f}s", flush=True)
            # Save after every complete case to retain work after interruption.
            fields = sorted({key for row in rows for key in row})
            with (args.out / "control_results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    (args.out / "control_config.json").write_text(json.dumps(
        dict(seed=args.seed, n_per_scenario=args.n_per_scenario,
             dt_mpc_s=0.05, dt_ctrl_s=0.005, horizon=20,
             nmpc_max_iterations=40, max_duration_s=args.max_duration_s,
             shard_id=args.shard_id, num_shards=args.num_shards,
             notes="No learned OV or IMM; full 17-state NMPC C0, ideal plant."), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-scenario", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--out", type=Path, default=ROOT / "results_005")
    parser.add_argument("--max-duration-s", type=float, default=0.0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    run(parser.parse_args())
