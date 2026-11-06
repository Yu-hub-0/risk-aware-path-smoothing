# -*- coding: utf-8 -*-
"""B-A* path planning on a 2D risk grid.

This module keeps the original B-A* idea from the script version:

- 8-neighbor A* search.
- Edge cost combines grid distance and local risk.
- Heuristic combines Euclidean distance and average straight-line risk.
- A Bresenham line-of-sight pass sparsifies the dense grid path into key nodes.

The file is intentionally import-safe: importing it no longer launches plotting
or reads absolute paths outside this project.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RISK_PATH = ROOT / "risk_matrix_2d.npy"
DEFAULT_X_PATH = ROOT / "x_unique.npy"
DEFAULT_Y_PATH = ROOT / "y_unique.npy"


def load_default_map():
    risk_map = np.load(DEFAULT_RISK_PATH).astype(float)
    x_unique = np.load(DEFAULT_X_PATH).astype(float)
    y_unique = np.load(DEFAULT_Y_PATH).astype(float)
    return risk_map, x_unique, y_unique


def bresenham(x1, y1, x2, y2):
    """Return integer grid cells on the line from (x1, y1) to (x2, y2)."""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points


def line_max_risk(risk_map, a, b):
    cells = bresenham(a[0], a[1], b[0], b[1])
    # Experimental guard: a centerline can cross a risk cell near a corner
    # even when ordinary Bresenham chooses the neighboring free cell.
    samples = max(2, 8 * max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1)
    rows = np.clip(np.rint(np.linspace(a[0], b[0], samples)).astype(int),
                   0, risk_map.shape[0] - 1)
    cols = np.clip(np.rint(np.linspace(a[1], b[1], samples)).astype(int),
                   0, risk_map.shape[1] - 1)
    return max(max(float(risk_map[cell]) for cell in cells),
               float(np.max(risk_map[rows, cols])))


def line_mean_risk(risk_map, a, b):
    cells = bresenham(a[0], a[1], b[0], b[1])
    return float(np.mean([risk_map[cell] for cell in cells]))


def g_combined(risk_map, current_node, neighbor, distance_weight=0.4, risk_weight=0.6):
    distance = np.linalg.norm(np.array(neighbor) - np.array(current_node))
    risk = float(risk_map[neighbor])
    return distance_weight * distance + risk_weight * risk


def h_combined(risk_map, neighbor, goal, distance_weight=0.4, risk_weight=0.6):
    distance = np.linalg.norm(np.array(neighbor) - np.array(goal))
    risk_average = line_mean_risk(risk_map, neighbor, goal)
    return distance_weight * distance + risk_weight * risk_average


def get_neighbors(position, rows, cols):
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    neighbors = []
    for dx, dy in directions:
        nx, ny = position[0] + dx, position[1] + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            neighbors.append((nx, ny))
    return neighbors


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star_search(risk_map, start, goal, risk_threshold=0.5):
    """Run risk-aware A* and forbid nodes whose risk exceeds threshold."""
    rows, cols = risk_map.shape
    start = tuple(start)
    goal = tuple(goal)
    if risk_map[start] > risk_threshold:
        raise ValueError(f"start {start} risk={risk_map[start]:.3f} exceeds threshold")
    if risk_map[goal] > risk_threshold:
        raise ValueError(f"goal {goal} risk={risk_map[goal]:.3f} exceeds threshold")

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    open_members = {start}
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        open_members.discard(current)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, rows, cols):
            if risk_map[neighbor] > risk_threshold:
                continue
            if neighbor[0] != current[0] and neighbor[1] != current[1]:
                # Do not squeeze a diagonal edge between blocked cells.
                if (risk_map[neighbor[0], current[1]] > risk_threshold or
                        risk_map[current[0], neighbor[1]] > risk_threshold):
                    continue
            tentative_g_score = g_score[current] + g_combined(risk_map, current, neighbor)
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + h_combined(risk_map, neighbor, goal)
                if neighbor not in open_members:
                    heapq.heappush(open_set, (f_score, neighbor))
                    open_members.add(neighbor)

    return None


def sparsify_path_with_risk(risk_map, path, risk_threshold=0.5, target_nodes=None):
    """Compress a dense A* path using risk-safe line-of-sight segments."""
    if not path:
        return []

    sparse = [path[0]]
    i = 0
    while i < len(path) - 1:
        best = i + 1
        for j in range(i + 1, len(path)):
            if line_max_risk(risk_map, path[i], path[j]) <= risk_threshold:
                best = j
            else:
                break
        sparse.append(path[best])
        i = best

    if target_nodes is not None and len(sparse) > target_nodes:
        keep = np.linspace(0, len(sparse) - 1, target_nodes).round().astype(int)
        keep[0] = 0
        keep[-1] = len(sparse) - 1
        sparse = [sparse[idx] for idx in np.unique(keep)]
    return sparse


def run_b_astar(risk_map, start, goal, risk_threshold=0.5, target_nodes=9):
    dense_path = a_star_search(risk_map, start, goal, risk_threshold=risk_threshold)
    if dense_path is None:
        raise RuntimeError(f"B-A* failed to find a path from {start} to {goal}")
    sparse_path = sparsify_path_with_risk(
        risk_map,
        dense_path,
        risk_threshold=risk_threshold,
        target_nodes=target_nodes,
    )
    if target_nodes is not None and len(sparse_path) < target_nodes <= len(dense_path):
        keep = np.linspace(0, len(dense_path) - 1, target_nodes).round().astype(int)
        keep[0] = 0
        keep[-1] = len(dense_path) - 1
        sparse_path = [dense_path[idx] for idx in np.unique(keep)]
    return dense_path, sparse_path


def plot_b_astar_result(risk_map, dense_path, sparse_path, out_path=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(risk_map, cmap="jet", interpolation="nearest", origin="lower", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Risk(0-1)")
    dense_r, dense_c = zip(*dense_path)
    sparse_r, sparse_c = zip(*sparse_path)
    ax.plot(dense_c, dense_r, color="pink", linewidth=1.5, label="B-A* dense path")
    ax.plot(sparse_c, sparse_r, "o-", color="lime", linewidth=2.5, label="B-A* key nodes")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
    else:
        plt.show()
    return fig


def main():
    risk_map, _, _ = load_default_map()
    start = (25, 25)
    goal = (140, 300)
    dense_path, sparse_path = run_b_astar(
        risk_map,
        start,
        goal,
        risk_threshold=0.5,
        target_nodes=9,
    )
    print(f"Dense B-A* nodes: {len(dense_path)}")
    print(f"Sparse B-A* nodes: {len(sparse_path)}")
    print(sparse_path)
    plot_b_astar_result(risk_map, dense_path, sparse_path)


if __name__ == "__main__":
    main()
