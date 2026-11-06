"""E1 path-planning comparison using frozen copies of the supplied B-A* code.

The original B-A* and spline modules are in ``source/``. This runner adds the
paper's posterior acceptance/repair and records failures explicitly. Synthetic
maps use 1 m cells; the supplied 100 m/cell map is inappropriate for the
quadrotor tracking experiment without an explicit physical-scale conversion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import make_interp_spline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from B_Astar_Algorithm import run_b_astar  # noqa: E402
from astar_risk_path import densify_polyline  # noqa: E402

METHODS = ("P0", "P1", "P2", "P3")
RISK_LIMIT = 0.5
CELL_M = 1.0
V_MAX = 2.0
A_T_MAX = 2.0
A_LAT_MAX = 2.5
OPT_SAMPLES = 120
CHECK_SPACING_M = 0.005


def make_case(kind: str, index: int, seed: int):
    rng = np.random.default_rng(seed + 100_003 * index + 700_001 *
                                ("corner", "s_bend", "narrow").index(kind))
    n = 40
    yy, xx = np.mgrid[:n, :n]
    risk = 0.04 + 0.025 * (np.sin(xx / 4.0 + index) ** 2)
    risk = np.broadcast_to(risk, (n, n)).copy()
    jx, jy = rng.integers(-2, 3, size=2)
    if kind == "corner":
        risk[13 + jy:27 + jy, 14 + jx:27 + jx] = 0.9
        start, goal = (5, 5), (34, 34)
    elif kind == "s_bend":
        risk[:, 12 + jx:15 + jx] = 0.9
        risk[7 + jy:14 + jy, 12 + jx:15 + jx] = 0.04
        risk[:, 25 + jx:28 + jx] = 0.9
        risk[25 + jy:33 + jy, 25 + jx:28 + jx] = 0.04
        start, goal = (19, 4), (19, 35)
    elif kind == "narrow":
        risk[12 + jy:16 + jy, :] = 0.9
        risk[12 + jy:16 + jy, 5 + jx:13 + jx] = 0.04
        risk[24 + jy:28 + jy, :] = 0.9
        risk[24 + jy:28 + jy, 26 + jx:35 + jx] = 0.04
        start, goal = (4, 19), (35, 19)
    else:
        raise ValueError(kind)
    return risk, start, goal


def grid_path_xy(path):
    return np.asarray([(col * CELL_M, row * CELL_M) for row, col in path], float)


def controls(polyline: np.ndarray, count: int,
             extra: np.ndarray | None = None) -> np.ndarray:
    points = densify_polyline(polyline, max(50, count * 3), None)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    chosen = np.linspace(0.0, distances[-1], count)
    if extra is not None and len(extra):
        # Reinsert nearby nodes of the original B-A* route at violating bends.
        nearby = np.argmin(np.linalg.norm(points[None, :, :] - extra[:, None, :], axis=2), axis=1)
        chosen = np.unique(np.r_[chosen, distances[nearby]])
    cp = np.column_stack([np.interp(chosen, distances, points[:, k]) for k in range(2)])
    return cp[np.r_[True, np.linalg.norm(np.diff(cp, axis=0), axis=1) > 1e-9]]


def spline_from_controls(cp: np.ndarray):
    d = np.linalg.norm(np.diff(cp, axis=0), axis=1)
    if len(cp) < 4 or min(d) < 1e-7:
        raise ValueError("Too few or duplicate controls")
    u = np.r_[0.0, np.cumsum(d)]
    u /= u[-1]
    return make_interp_spline(u, cp, k=3, axis=0)


def risk_at(points: np.ndarray, risk: np.ndarray) -> np.ndarray:
    col = np.clip(np.rint(points[:, 0] / CELL_M).astype(int), 0, risk.shape[1] - 1)
    row = np.clip(np.rint(points[:, 1] / CELL_M).astype(int), 0, risk.shape[0] - 1)
    return risk[row, col]


def objective(cp: np.ndarray, cp0: np.ndarray, risk: np.ndarray,
              risk_weight: float) -> float:
    try:
        curve = spline_from_controls(cp)
    except ValueError:
        return math.inf
    p = curve(np.linspace(0, 1, OPT_SAMPLES))
    z = risk_at(p, risk)
    phi = np.maximum(0.0, z - RISK_LIMIT) ** 2
    dev = np.sum((cp - cp0) ** 2)
    bend = np.sum(np.diff(cp, n=2, axis=0) ** 2)
    return risk_weight * float(np.sum(phi)) + 0.25 * dev + 0.02 * bend


def optimize_controls(cp0: np.ndarray, risk: np.ndarray,
                      risk_weight: float, sweeps: int = 2) -> np.ndarray:
    """Bounded derivative-free coordinate search for nearest-cell risk.

    Nearest-cell lookup has zero gradient almost everywhere, so the copied
    gradient-based optimizer cannot reliably move away from violating cells.
    The objective still uses the exact thresholded lookup in main1.tex.
    """
    cp = cp0.copy()
    best = objective(cp, cp0, risk, risk_weight)
    directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)], float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for step in (0.8, 0.4)[:sweeps]:
        for j in range(1, len(cp) - 1):
            for direction in directions:
                trial = cp.copy()
                trial[j] += step * direction
                trial[j] = np.clip(trial[j], 0, risk.shape[0] - 1)
                score = objective(trial, cp0, risk, risk_weight)
                if score + 1e-9 < best:
                    cp, best = trial, score
    return cp


def sample_curve(curve, spacing: float = CHECK_SPACING_M):
    u0 = np.linspace(0, 1, 1200)
    p0 = curve(u0)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p0, axis=0), axis=1))]
    keep = np.r_[True, np.diff(cumulative) > 1e-10]
    cumulative, u0 = cumulative[keep], u0[keep]
    s = np.linspace(0, cumulative[-1], max(2, math.ceil(cumulative[-1] / spacing) + 1))
    return s, np.interp(s, cumulative, u0)


def assess(points: np.ndarray, risk: np.ndarray):
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mid = 0.5 * (points[1:] + points[:-1])
    z = risk_at(mid, risk)
    bad = z > RISK_LIMIT
    return dict(length_m=float(ds.sum()), violation_length_m=float(ds[bad].sum()),
                violation_fraction=float(ds[bad].sum() / max(ds.sum(), 1e-12)),
                max_risk=float(np.max(risk_at(points, risk))))


def retime(curve):
    """Arc-length speed profile with zero endpoint speed and acceleration caps."""
    s, u = sample_curve(curve, spacing=0.18)
    d1, d2 = curve.derivative(1)(u), curve.derivative(2)(u)
    norm = np.linalg.norm(d1, axis=1)
    if np.min(norm) < 1e-5:
        raise ValueError("Zero tangent")
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    kappa = np.abs(cross) / norm ** 3
    v = np.minimum(V_MAX, np.sqrt(A_LAT_MAX / np.maximum(kappa, 1e-9)))
    v[0] = v[-1] = 0.0
    for j in range(1, len(v)):
        v[j] = min(v[j], math.sqrt(v[j - 1] ** 2 + 2 * A_T_MAX * (s[j] - s[j - 1])))
    for j in range(len(v) - 2, -1, -1):
        v[j] = min(v[j], math.sqrt(v[j + 1] ** 2 + 2 * A_T_MAX * (s[j + 1] - s[j])))
    dt = 2 * np.diff(s) / np.maximum(v[1:] + v[:-1], 1e-9)
    t = np.r_[0.0, np.cumsum(dt)]
    tangential = np.abs(np.diff(v ** 2) / np.maximum(2 * np.diff(s), 1e-9))
    lateral = v ** 2 * kappa
    ok = (np.max(v) <= V_MAX + 1e-6 and np.max(tangential) <= A_T_MAX + 1e-6
          and np.max(lateral) <= A_LAT_MAX + 1e-6)
    return dict(duration_s=float(t[-1]), peak_speed_mps=float(v.max()),
                peak_tangential_acc_mps2=float(tangential.max()),
                peak_lateral_acc_mps2=float(lateral.max()), kinematic_ok=bool(ok)), (t, s, u, v, kappa)


def solve_one(method: str, sparse_xy: np.ndarray, dense_xy: np.ndarray,
              risk: np.ndarray):
    t0 = perf_counter()
    if method == "P0":
        points = densify_polyline(sparse_xy, None, CHECK_SPACING_M)
        result = assess(points, risk)
        result.update(status="geometric_only", kinematic_ok=False,
                      duration_s=None, repair_passes=0, optimized=False)
        return result, points, None
    accepted = None
    last = None
    extra = None
    for repair in range(3 if method == "P3" else 1):
        cp0 = controls(sparse_xy, 9, extra)
        cp = cp0 if method == "P1" else optimize_controls(cp0, risk, 80.0 * (2 ** repair))
        try:
            curve = spline_from_controls(cp)
            _, u = sample_curve(curve)
            points = curve(u)
            result = assess(points, risk)
            kin, _ = retime(curve)
            result.update(kin)
            result.update(repair_passes=repair, optimized=method != "P1")
            last = (result, points, curve)
            if result["violation_length_m"] <= 1e-8 and result["kinematic_ok"]:
                accepted = last
                break
            if method == "P3" and repair < 2:
                z = risk_at(points, risk)
                bad = points[z > RISK_LIMIT]
                if len(bad):
                    # Equally spaced violating samples prevent one long segment
                    # from consuming the entire repair budget.
                    selected = bad[np.linspace(0, len(bad) - 1,
                                               min(4, len(bad))).astype(int)]
                    nearest = np.argmin(np.linalg.norm(
                        dense_xy[None, :, :] - selected[:, None, :], axis=2), axis=1)
                    indices = np.unique(np.clip(np.r_[nearest - 3, nearest,
                                                      nearest + 3], 0, len(dense_xy) - 1))
                    extra = dense_xy[indices] if extra is None else np.vstack([extra, dense_xy[indices]])
        except (ValueError, FloatingPointError) as exc:
            last = (dict(status=f"invalid_spline:{type(exc).__name__}",
                         repair_passes=repair, optimized=method != "P1"),
                    sparse_xy, None)
    result, points, curve = accepted or last
    if "status" not in result:
        result["status"] = "accepted" if accepted else "risk_or_kinematic_failed"
    result["compute_s"] = perf_counter() - t0
    return result, points, curve


def run(n_per_scenario: int, seed: int, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    example = None
    for kind in ("corner", "s_bend", "narrow"):
        for index in range(n_per_scenario):
            risk, start, goal = make_case(kind, index, seed)
            t0 = perf_counter()
            try:
                dense, sparse = run_b_astar(risk, start, goal, risk_threshold=RISK_LIMIT,
                                            target_nodes=None)
            except Exception as exc:
                for method in METHODS:
                    rows.append(dict(scenario=kind, case=index, method=method,
                                     status=f"search_failed:{type(exc).__name__}"))
                continue
            path = grid_path_xy(sparse)
            planner_time = perf_counter() - t0
            curves = {}
            for method in METHODS:
                result, points, _ = solve_one(method, path, grid_path_xy(dense), risk)
                result.update(scenario=kind, case=index, method=method,
                              search_s=planner_time, dense_nodes=len(dense),
                              sparse_nodes=len(sparse))
                rows.append(result)
                curves[method] = points
            if example is None or (rows[-1].get("status") == "accepted" and
                                   rows[-3].get("violation_length_m", 0) > 0):
                example = (kind, index, risk.copy(), path.copy(),
                           {key: val.copy() for key, val in curves.items()})
    fields = sorted({key for row in rows for key in row})
    with (out / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if example:
        kind, index, risk, path, curves = example
        np.savez_compressed(out / "example.npz", scenario=kind, case=index,
                            risk=risk, polyline=path, **curves)
    metadata = dict(seed=seed, n_per_scenario=n_per_scenario,
                    total_cases=3 * n_per_scenario,
                    map="synthetic, 40 x 40 cells, 1 m/cell",
                    risk_limit=RISK_LIMIT, independent_check_spacing_m=CHECK_SPACING_M,
                    v_max_mps=V_MAX, a_t_max_mps2=A_T_MAX,
                    a_lat_max_mps2=A_LAT_MAX,
                    source_modules=["source/B_Astar_Algorithm.py",
                                    "source/astar_risk_path.py"],
                    scope="geometric and reference-kinematic E1 only; no NMPC execution")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "accepted": sum(r.get("status") == "accepted" for r in rows),
                      "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-scenario", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--out", type=Path, default=ROOT / "results_005")
    args = parser.parse_args()
    run(args.n_per_scenario, args.seed, args.out)
