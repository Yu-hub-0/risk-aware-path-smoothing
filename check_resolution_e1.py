"""Independent path-risk sampling refinement for all accepted E1 references."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e1 import ROOT, assess, grid_path_xy, make_case, sample_curve, solve_one
from source.B_Astar_Algorithm import run_b_astar


def main(n_per_scenario: int, seed: int, out: Path):
    rows = []
    for kind in ("corner", "s_bend", "narrow"):
        for index in range(n_per_scenario):
            risk, start, goal = make_case(kind, index, seed)
            dense, sparse = run_b_astar(risk, start, goal,
                                        risk_threshold=0.5, target_nodes=None)
            path, raw = grid_path_xy(sparse), grid_path_xy(dense)
            for method in ("P1", "P2", "P3"):
                result, _, curve = solve_one(method, path, raw, risk)
                if result["status"] != "accepted" or curve is None:
                    continue
                row = dict(scenario=kind, case=index, method=method,
                           original_violation_m=result["violation_length_m"])
                for spacing in (0.0025,):
                    _, u = sample_curve(curve, spacing=spacing)
                    metrics = assess(curve(u), risk)
                    row[f"violation_at_{spacing:g}_m"] = metrics["violation_length_m"]
                rows.append(row)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Checked {len(rows)} accepted references")
    print(f"New violations at 0.0025 m: {sum(r['violation_at_0.0025_m'] > 0 for r in rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-scenario", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--out", type=Path, default=ROOT / "results_005" / "resolution_check.csv")
    args = parser.parse_args()
    main(args.n_per_scenario, args.seed, args.out)
