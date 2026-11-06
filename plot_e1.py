"""Publication-style E1 figures from case_results.csv, using matplotlib only."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
COLORS = {"P0": "#657078", "P1": "#A9B5BF", "P2": "#5B86AA", "P3": "#176B62"}
LABELS = {"P0": "B-A*", "P1": "Spline", "P2": "+ risk penalty", "P3": "+ repair"}
PATH_LABELS = {"P0": "P0  B-A*", "P1": "P1  B-spline",
               "P2": "P2  + risk penalty", "P3": "P3  + local repair"}
RISK_CMAP = ListedColormap(["#F7F7F7", "#D8635D"])
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "savefig.facecolor": "white"})


def save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key):
    try:
        return float(row[key])
    except (ValueError, TypeError, KeyError):
        return np.nan


def paired_bootstrap_ci(values: np.ndarray, seed: int = 315, repeats: int = 5000):
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(repeats, len(values)))]
    means = np.nanmean(sample, axis=1)
    return np.quantile(means, [0.025, 0.975])


def panel_marks(fig, axes, descriptions, y: float):
    """Put each panel's letter and short description below its axes."""
    for letter, axis, description in zip("abc", axes, descriptions):
        box = axis.get_position()
        fig.text((box.x0 + box.x1) / 2, y, f"({letter}) {description}",
                 ha="center", va="center", fontsize=7, fontweight="bold")


def map_figure(npz_path: Path, out: Path):
    data = np.load(npz_path)
    risk = data["risk"]
    paths = {method: data[method] for method in LABELS}
    fig = plt.figure(figsize=(7.2, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.075, right=0.98, top=0.98, bottom=0.21, wspace=0.28)
    for axis in (ax, ax_zoom):
        axis.imshow((risk > 0.5).astype(int), cmap=RISK_CMAP, vmin=0, vmax=1, origin="lower",
                    extent=(-0.5, 39.5, -0.5, 39.5), interpolation="nearest")
        for method in ("P0", "P1", "P2", "P3"):
            p = paths[method]
            axis.plot(p[:, 0], p[:, 1], color=COLORS[method],
                      lw=1.7 if method == "P3" else 1.0,
                      linestyle="--" if method == "P1" else "-",
                      label=PATH_LABELS[method], zorder=4 if method == "P3" else 3)
        axis.set_aspect("equal")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
    ax.set_xlim(0, 39)
    ax.set_ylim(0, 39)
    # Locate the largest disagreement between ordinary smoothing and repaired path.
    p1 = paths["P1"]
    z = risk[np.clip(np.rint(p1[:, 1]).astype(int), 0, 39),
             np.clip(np.rint(p1[:, 0]).astype(int), 0, 39)]
    bad = p1[z > 0.5]
    center = bad[len(bad) // 2] if len(bad) else p1[len(p1) // 2]
    ax_zoom.set_xlim(max(0, center[0] - 6), min(39, center[0] + 6))
    ax_zoom.set_ylim(max(0, center[1] - 6), min(39, center[1] + 6))
    ax_zoom.legend(loc="upper right", fontsize=6, handlelength=1.7)
    panel_marks(fig, (ax, ax_zoom),
                ("Global path comparison", "Enlarged risk boundary"), 0.035)
    save(fig, out / "Fig_E1_paths")


def quantitative_figure(rows, out: Path):
    cases = sorted({(r["scenario"], int(r["case"])) for r in rows})
    lookup = {(r["scenario"], int(r["case"]), r["method"]): r for r in rows}
    methods = list(LABELS)
    violations = np.array([[number(lookup[(case[0], case[1], m)], "violation_fraction")
                            for m in methods] for case in cases])
    lengths = np.array([[number(lookup[(case[0], case[1], m)], "length_m")
                        for m in methods] for case in cases])
    accepted = np.array([[lookup[(case[0], case[1], m)]["status"] == "accepted"
                          for m in methods] for case in cases])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.27, wspace=0.34)
    x = np.arange(4)
    for j, method in enumerate(methods):
        y = violations[:, j] * 100
        # Deterministic horizontal displacement only avoids mark overlap.
        jitter = 0.08 * np.sin(np.arange(len(y)) * 2.399)
        axes[0].scatter(j + jitter, y, color=COLORS[method],
                        s=5, alpha=0.24, linewidths=0, rasterized=True)
        ci = paired_bootstrap_ci(y)
        axes[0].errorbar(j, np.nanmean(y), yerr=np.array([[np.nanmean(y) - ci[0]],
                                                          [ci[1] - np.nanmean(y)]]),
                         fmt="o", color=COLORS[method], markersize=4,
                         capsize=2, lw=1.1)
        if j:
            n = int(accepted[:, j].sum())
            interval = binomtest(n, len(cases)).proportion_ci(confidence_level=0.95,
                                                               method="exact")
            axes[1].errorbar(j, 100 * n / len(cases),
                             yerr=np.array([[100 * (n / len(cases) - interval.low)],
                                            [100 * (interval.high - n / len(cases))]]),
                             fmt="o", color=COLORS[method], markersize=5,
                             capsize=2, lw=1.1)
            relative = 100 * (lengths[:, j] / lengths[:, 0] - 1)
            ci_len = paired_bootstrap_ci(relative)
            axes[2].errorbar(j, np.nanmean(relative),
                             yerr=np.array([[np.nanmean(relative) - ci_len[0]],
                                            [ci_len[1] - np.nanmean(relative)]]),
                             fmt="o", color=COLORS[method], markersize=5,
                             capsize=2, lw=1.1)
    axes[0].set_ylabel("Path length above threshold (%)")
    axes[1].set_ylabel("Accepted paths (%)")
    axes[1].set_ylim(0, 105)
    axes[2].set_ylabel("Length change (%)")
    axes[2].axhline(0, color="#C4C8CC", lw=0.7)
    for ax in axes:
        ax.set_xlim(-0.35, 3.35)
        ax.set_xticks(x, [LABELS[m] for m in methods], rotation=25,
                      rotation_mode="anchor", ha="right")
        ax.grid(axis="y", color="#E7E9EB", linewidth=0.55)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x[1:], [LABELS[m] for m in methods[1:]], rotation=25,
                       rotation_mode="anchor", ha="right")
    axes[2].set_xticks(x[1:], [LABELS[m] for m in methods[1:]], rotation=25,
                       rotation_mode="anchor", ha="right")
    panel_marks(fig, axes,
                ("Above-threshold length", "Accepted references", "Relative path length"),
                0.035)
    save(fig, out / "Fig_E1_quantitative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "results_005")
    args = parser.parse_args()
    rows = load_rows(args.data / "case_results.csv")
    quantitative_figure(rows, args.data)
    if (args.data / "example.npz").exists():
        map_figure(args.data / "example.npz", args.data)
