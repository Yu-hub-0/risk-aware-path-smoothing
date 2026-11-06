"""E1 closed-loop evidence figure from saved NMPC logs and case-level CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_e1 import make_case  # noqa: E402

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "svg.fonttype": "none", "pdf.fonttype": 42})


def plot(data_dir: Path, scenario: str, case: int, method: str, seed: int):
    name = f"{scenario}_{case:02d}_{method}.npz"
    candidates = list((data_dir / "control_shards").glob(f"shard_*/logs/{name}"))
    if not candidates:
        candidates = list((data_dir / "logs").glob(name))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one log for {name}, got {len(candidates)}")
    log = np.load(candidates[0])
    t = log["time_s"]
    p, ref = log["actual_pos"], log["ref_pos"]
    err = np.linalg.norm(p - ref, axis=1)
    solve_ms = 1000 * log["solver_time_s"]
    risk, _, _ = make_case(scenario, case, seed)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 1.05, 1.05]})
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.25, wspace=0.35)
    ax = axes[0]
    ax.imshow((risk > 0.5).astype(int),
              cmap=ListedColormap(["#F7F7F7", "#D8635D"]),
              origin="lower", vmin=0, vmax=1,
              extent=(-0.5, 39.5, -0.5, 39.5), interpolation="nearest")
    ax.plot(ref[:, 0], ref[:, 1], color="#176B62", lw=1.45, label="Reference")
    ax.plot(p[:, 0], p[:, 1], color="#364C55", lw=0.9, linestyle="--", label="Actual")
    ax.set_xlim(0, 39)
    ax.set_ylim(0, 39)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize=6, loc="upper left")

    axes[1].plot(t, err * 100, color="#176B62", lw=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position error (cm)")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(t, solve_ms, color="#176B62", lw=0.65, rasterized=True)
    axes[2].axhline(50, color="#B2504B", lw=0.9, linestyle="--", label="Control period (50 ms)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Solve time (ms)")
    axes[2].set_ylim(bottom=0)
    axes[2].legend(fontsize=6, loc="upper right")
    for axis in axes[1:]:
        axis.grid(axis="y", color="#E7E9EB", lw=0.55)
        axis.set_axisbelow(True)
    for letter, axis, description in zip(
            "abc", axes, ("Closed-loop path", "Position error", "NMPC solve time")):
        box = axis.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.035, f"({letter}) {description}",
                 ha="center", va="center", fontsize=7, fontweight="bold")
    stem = data_dir / f"Fig_E1_control_{scenario}_{case:02d}_{method}"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "results_005")
    parser.add_argument("--scenario", default="corner")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--method", default="P3")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(plot(args.data, args.scenario, args.case, args.method, args.seed))
