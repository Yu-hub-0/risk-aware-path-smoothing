"""Merge E1 control shards and produce transparent case-level summary tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def main(data: Path, require_complete: bool):
    geometric = read_csv(data / "case_results.csv")
    control = []
    for path in sorted((data / "control_shards").glob("shard_*/control_results.csv")):
        control.extend(read_csv(path))
    if not control and (data / "control_results.csv").exists():
        control = read_csv(data / "control_results.csv")
    keys = [(r["scenario"], int(r["case"]), r["method"]) for r in control]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated scenario/case/method in control shards")
    expected = 3 * (len(geometric) // 4)
    if require_complete and len(control) != expected:
        raise RuntimeError(f"Incomplete control results: {len(control)}/{expected}")
    if control:
        control.sort(key=lambda r: (r["scenario"], int(r["case"]), r["method"]))
        write_csv(data / "control_results.csv", control)
    rows = []
    for method in ("P0", "P1", "P2", "P3"):
        subset = [r for r in geometric if r["method"] == method]
        n = len(subset)
        accepted = sum(r["status"] == "accepted" for r in subset)
        interval = binomtest(accepted, n).proportion_ci(method="exact") if method != "P0" else None
        executed = [r for r in control if r["method"] == method and r["status"] == "executed"]
        rows.append(dict(method=method, n_paths=n, accepted=accepted if method != "P0" else "",
                         acceptance_rate=accepted / n if method != "P0" else "",
                         acceptance_ci_low=interval.low if interval else "",
                         acceptance_ci_high=interval.high if interval else "",
                         mean_risk_violation_fraction=float(np.nanmean(
                             [value(r, "violation_fraction") for r in subset])),
                         mean_length_m=float(np.nanmean([value(r, "length_m") for r in subset])),
                         control_attempts=sum(r["method"] == method for r in control),
                         executed=len(executed),
                         mean_rmse_m=float(np.mean([value(r, "rmse_m") for r in executed]))
                         if executed else "",
                         mean_actual_risk_violation_fraction=float(np.mean(
                             [value(r, "actual_risk_violation_fraction") for r in executed]))
                         if executed else "",
                         mean_deadline_fraction=float(np.mean(
                             [value(r, "solver_deadline_fraction") for r in executed]))
                         if executed else ""))
    write_csv(data / "summary_by_method.csv", rows)
    by_key = {(r["scenario"], int(r["case"]), r["method"]): r for r in geometric}
    cases = sorted({(r["scenario"], int(r["case"])) for r in geometric})
    rng = np.random.default_rng(154)
    draws = rng.integers(0, len(cases), size=(5000, len(cases)))
    effects = []
    for comparator in ("P1", "P2"):
        for label, extractor, scale in (
            ("acceptance_difference_pp", lambda r: float(r["status"] == "accepted"), 100),
            ("risk_violation_difference_pp", lambda r: value(r, "violation_fraction"), 100),
            ("length_difference_m", lambda r: value(r, "length_m"), 1)):
            diff = np.array([extractor(by_key[(*case, "P3")]) -
                             extractor(by_key[(*case, comparator)]) for case in cases]) * scale
            means = np.mean(diff[draws], axis=1)
            lo, hi = np.quantile(means, [0.025, 0.975])
            effects.append(dict(comparison=f"P3 minus {comparator}", metric=label,
                                n_cases=len(cases), mean_difference=float(np.mean(diff)),
                                ci_95_low=float(lo), ci_95_high=float(hi),
                                method="paired case bootstrap; 5000 replicates"))
    write_csv(data / "paired_effects.csv", effects)
    (data / "completion.json").write_text(json.dumps(
        dict(geometry_rows=len(geometric), control_rows=len(control),
             expected_control_rows=expected, complete=len(control) == expected), indent=2),
        encoding="utf-8")
    print(json.dumps({"control_rows": len(control), "expected": expected,
                      "complete": len(control) == expected}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "results_005")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    main(args.data, not args.allow_partial)
