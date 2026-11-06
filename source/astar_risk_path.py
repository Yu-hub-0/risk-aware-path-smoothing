from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    path: list[tuple[int, int]]
    distance: float
    risk_cost: float
    total_cost: float


@dataclass(frozen=True)
class SmoothResult:
    control_points: np.ndarray
    smooth_points: np.ndarray
    degree: int
    distance: float
    risk_cost: float
    total_cost: float
    verification: dict | None = None   # 风险验证结果
    optimized: bool = False            # 控制点是否经过风险约束优化


NEIGHBORS_8 = (
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
)


def coord_to_index(value: float, axis: np.ndarray) -> int:
    """Return the nearest grid index for a projected coordinate."""
    return int(np.abs(axis - value).argmin())


def parse_grid_point(text: str) -> tuple[int, int]:
    row_text, col_text = text.split(",", maxsplit=1)
    return int(row_text), int(col_text)


def parse_coord_point(text: str, x_axis: np.ndarray, y_axis: np.ndarray) -> tuple[int, int]:
    x_text, y_text = text.split(",", maxsplit=1)
    col = coord_to_index(float(x_text), x_axis)
    row = coord_to_index(float(y_text), y_axis)
    return row, col


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_metrics(
    path: list[tuple[int, int]],
    risk: np.ndarray,
    cell_size: float,
    risk_weight: float,
) -> tuple[float, float, float]:
    distance = 0.0
    risk_cost = 0.0

    for (row_a, col_a), (row_b, col_b) in zip(path, path[1:]):
        step_cells = math.hypot(row_b - row_a, col_b - col_a)
        step_distance = step_cells * cell_size
        step_risk = (risk[row_a, col_a] + risk[row_b, col_b]) * 0.5
        distance += step_distance
        risk_cost += step_risk * step_distance

    total_cost = distance + risk_weight * risk_cost
    return distance, risk_cost, total_cost


def continuous_path_metrics(
    points_xy: np.ndarray,
    risk: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    risk_weight: float,
) -> tuple[float, float, float]:
    distance = 0.0
    risk_cost = 0.0

    for point_a, point_b in zip(points_xy, points_xy[1:]):
        step_distance = float(np.linalg.norm(point_b - point_a))
        col_a = coord_to_index(point_a[0], x_axis)
        row_a = coord_to_index(point_a[1], y_axis)
        col_b = coord_to_index(point_b[0], x_axis)
        row_b = coord_to_index(point_b[1], y_axis)
        step_risk = (risk[row_a, col_a] + risk[row_b, col_b]) * 0.5
        distance += step_distance
        risk_cost += step_risk * step_distance

    total_cost = distance + risk_weight * risk_cost
    return distance, risk_cost, total_cost


# ──────────────────────────────────────────────────────────────
# Point 1: 平滑后风险验证
# ──────────────────────────────────────────────────────────────

def _batch_risk_lookup(
    points_xy: np.ndarray,
    risk: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化的风险查询：给定一组投影坐标点，返回每个点对应的风险值和网格索引。"""
    cols = np.array([coord_to_index(float(p[0]), x_axis) for p in points_xy])
    rows = np.array([coord_to_index(float(p[1]), y_axis) for p in points_xy])
    cols = np.clip(cols, 0, risk.shape[1] - 1)
    rows = np.clip(rows, 0, risk.shape[0] - 1)
    return risk[rows, cols], rows, cols


def verify_smooth_risk(
    smooth_points: np.ndarray,
    risk: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    risk_threshold: float,
) -> dict:
    """验证平滑路径上每个采样点是否越入高风险网格。

    原理
    ----
    B 样条是连续曲线，而风险地图是离散网格。即使控制点都在低风险格子上，
    曲线在相邻两个低风险控制点之间仍可能"内切"进高风险区域。
    此函数将平滑曲线的每个采样点映射回网格坐标，检查风险值是否超限。

    参数
    ----
    smooth_points : (N, 2)  平滑曲线采样点（投影坐标系）
    risk          : 2D 风险矩阵
    x_axis, y_axis: 投影坐标轴
    risk_threshold: 超过此值视为违规

    返回
    ----
    {
        "violations":      [{index, risk, grid, coord}, ...],
        "violation_count": 违规点数量,
        "violation_ratio": 违规点占比,
        "max_risk":        路径上最大风险值,
        "mean_risk":       路径上平均风险值,
    }
    """
    risk_values, rows, cols = _batch_risk_lookup(smooth_points, risk, x_axis, y_axis)

    violations = []
    for i in range(len(smooth_points)):
        r = float(risk_values[i])
        if r > risk_threshold:
            violations.append({
                "index": i,
                "risk": r,
                "grid": (int(rows[i]), int(cols[i])),
                "coord": (float(smooth_points[i, 0]), float(smooth_points[i, 1])),
            })

    return {
        "violations": violations,
        "violation_count": len(violations),
        "violation_ratio": len(violations) / len(smooth_points) if len(smooth_points) > 0 else 0.0,
        "max_risk": float(risk_values.max()),
        "mean_risk": float(risk_values.mean()),
    }


# ──────────────────────────────────────────────────────────────
# Point 2: 风险约束优化
# ──────────────────────────────────────────────────────────────

def risk_constrained_optimize(
    initial_control_points: np.ndarray,
    risk: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    degree: int,
    sample_count: int,
    risk_weight: float,
    deviation_weight: float = 1.0,
    smoothness_weight: float = 0.01,
    max_iterations: int = 50,
) -> np.ndarray:
    """通过优化控制点位置，让平滑路径自动远离高风险区。

    原理
    ----
    标准 B 样条只保证曲线光滑，不保证避开高风险区。
    此函数将控制点坐标作为优化变量，最小化以下目标：

        J = risk_weight    × Σ risk(spline(t))        ← 路径风险
          + deviation_weight × Σ (cp - cp₀)²          ← 偏差惩罚（正则化）
          + smoothness_weight× Σ (cp[i-1]-2cp[i]+cp[i+1])²  ← 曲率惩罚

    - 第一项推动控制点远离高风险区
    - 第二项防止控制点偏离初始位置太远，保持路径整体形状
    - 第三项保持路径光滑，防止优化后出现锯齿

    使用 L-BFGS-B 优化器（支持边界约束）。
    起点和终点控制点被固定，不会移动。

    参数
    ----
    initial_control_points : (M, 2) 初始控制点
    risk, x_axis, y_axis   : 风险地图和坐标轴
    degree                 : B 样条阶数
    sample_count           : 采样点数
    risk_weight            : 风险项权重（越大越避风险）
    deviation_weight       : 偏差项权重（越大越保守）
    smoothness_weight      : 曲率项权重（越大越光滑）
    max_iterations         : 最大优化迭代次数

    返回
    ----
    优化后的控制点 (M, 2)
    """
    from scipy.interpolate import make_interp_spline
    from scipy.optimize import minimize

    n_cp = len(initial_control_points)
    x_min, x_max = float(x_axis.min()), float(x_axis.max())
    y_min, y_max = float(y_axis.min()), float(y_axis.max())

    # 边界约束：[x0, y0, x1, y1, ..., xN, yN]
    bounds = [(x_min, x_max), (y_min, y_max)] * n_cp

    # 固定起点和终点：设极窄边界使其无法移动
    bounds[0] = (float(initial_control_points[0, 0]),) * 2
    bounds[1] = (float(initial_control_points[0, 1]),) * 2
    bounds[-2] = (float(initial_control_points[-1, 0]),) * 2
    bounds[-1] = (float(initial_control_points[-1, 1]),) * 2

    def objective(flat_cp: np.ndarray) -> float:
        cp = flat_cp.reshape(n_cp, 2)

        # 重建弦长参数化
        diffs = np.diff(cp, axis=0)
        chord_lengths = np.linalg.norm(diffs, axis=1)
        if chord_lengths.sum() < 1e-9:
            return 1e10
        chord_cumsum = np.r_[0.0, np.cumsum(chord_lengths)]
        params = chord_cumsum / chord_cumsum[-1]

        # 参数必须严格递增
        if not np.all(np.diff(params) > 0):
            return 1e10

        try:
            actual_deg = min(degree, len(cp) - 1)
            spline = make_interp_spline(params, cp, k=actual_deg, axis=0)
        except Exception:
            return 1e10

        samples = np.unique(np.r_[np.linspace(0.0, 1.0, sample_count), params])
        try:
            smooth_pts = spline(samples)
        except Exception:
            return 1e10

        # 风险代价
        risk_vals, _, _ = _batch_risk_lookup(smooth_pts, risk, x_axis, y_axis)
        risk_cost = float(risk_vals.sum())

        # 偏差代价
        deviation = float(np.sum((cp - initial_control_points) ** 2))

        # 曲率代价（二阶差分近似）
        if len(cp) >= 3:
            second_diff = cp[2:] - 2 * cp[1:-1] + cp[:-2]
            curvature = float(np.sum(second_diff ** 2))
        else:
            curvature = 0.0

        return risk_weight * risk_cost + deviation_weight * deviation + smoothness_weight * curvature

    x0 = initial_control_points.flatten()
    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-8},
    )

    return result.x.reshape(n_cp, 2)


# ──────────────────────────────────────────────────────────────
# Point 3: 急转弯处插入额外控制点
# ──────────────────────────────────────────────────────────────

def detect_sharp_turns(
    path_xy: np.ndarray,
    angle_threshold_deg: float = 120.0,
) -> np.ndarray:
    """检测路径中的急转弯，返回需要作为控制点的索引。

    原理
    ----
    B 样条在急转弯处会产生"内切"偏离——曲线不经过控制点，
    而是被拉向邻近控制点的平均位置，偏离量取决于：
      1) 控制点间距（越密偏离越小）
      2) B 样条阶数（3 次比 2 次偏离更大）
      3) 拐角角度（越尖锐偏离越严重）

    此函数通过计算相邻两段路径的夹角，在角度小于阈值的位置
    标记该点及前后邻点为"强制控制点"，使 B 样条更贴合原始路径。

    参数
    ----
    path_xy            : 路径点 (N, 2)
    angle_threshold_deg: 角度阈值（度）
        - 180° = 直线（不添加任何点）
        - 120° = 中等弯（建议添加，默认值）
        - 90°  = 急弯（必须添加）
        - 越小越宽松，只有非常尖锐的弯才会被标记

    返回
    ----
    需要强制包含的控制点索引数组
    """
    if len(path_xy) < 3:
        return np.array([], dtype=int)

    angle_threshold = np.radians(angle_threshold_deg)
    turn_indices = []

    for i in range(1, len(path_xy) - 1):
        v1 = path_xy[i] - path_xy[i - 1]
        v2 = path_xy[i + 1] - path_xy[i]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-9 or len2 < 1e-9:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            # 急转弯：标记该点及前后邻点，确保 B 样条在此处有足够控制点
            turn_indices.extend([i - 1, i, i + 1])

    return np.unique(np.array(turn_indices, dtype=int))


def make_open_uniform_knots(control_count: int, degree: int) -> np.ndarray:
    internal_count = control_count - degree - 1
    if internal_count > 0:
        internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)
    return np.r_[np.zeros(degree + 1), internal, np.ones(degree + 1)]


def choose_control_points(
    path_xy: np.ndarray,
    control_point_step: int,
    control_point_count: int | None,
    mandatory_indices: np.ndarray | None = None,
) -> np.ndarray:
    if len(path_xy) < 2:
        raise ValueError("Need at least two path points for smoothing.")

    if control_point_count is not None:
        if control_point_count < 2:
            raise ValueError("--control-point-count must be at least 2.")
        indices = np.linspace(0, len(path_xy) - 1, control_point_count).round().astype(int)
    else:
        if control_point_step < 1:
            raise ValueError("--control-point-step must be at least 1.")
        indices = np.arange(0, len(path_xy), control_point_step, dtype=int)
        if indices[-1] != len(path_xy) - 1:
            indices = np.r_[indices, len(path_xy) - 1]

    # 合并急转弯处的强制控制点索引
    if mandatory_indices is not None and len(mandatory_indices) > 0:
        indices = np.unique(np.r_[indices, mandatory_indices])

    indices = np.unique(indices)
    if indices[0] != 0:
        indices = np.r_[0, indices]
    if indices[-1] != len(path_xy) - 1:
        indices = np.r_[indices, len(path_xy) - 1]
    return path_xy[indices]


def densify_polyline(
    path_xy: np.ndarray,
    target_count: int | None,
    spacing: float | None,
) -> np.ndarray:
    if len(path_xy) < 2:
        return path_xy

    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    keep = np.r_[True, segment_lengths > 1e-9]
    path_xy = path_xy[keep]
    if len(path_xy) < 2:
        return path_xy

    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    total_length = cumulative[-1]
    if total_length <= 0.0:
        return path_xy

    if spacing is not None:
        if spacing <= 0.0:
            raise ValueError("--densify-spacing must be greater than 0.")
        distances = np.arange(0.0, total_length, spacing)
        if len(distances) == 0 or not np.isclose(distances[-1], total_length):
            distances = np.r_[distances, total_length]
    elif target_count is not None and target_count > 0:
        if target_count < 2:
            raise ValueError("--densify-control-points must be 0 or at least 2.")
        distances = np.linspace(0.0, total_length, target_count)
    else:
        return path_xy

    x_dense = np.interp(distances, cumulative, path_xy[:, 0])
    y_dense = np.interp(distances, cumulative, path_xy[:, 1])
    return np.column_stack((x_dense, y_dense))


def smooth_with_bspline(
    path_xy: np.ndarray,
    risk: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    risk_weight: float,
    degree: int,
    control_point_step: int,
    control_point_count: int | None,
    densify_control_points: int | None,
    densify_spacing: float | None,
    sample_count: int,
    angle_threshold_deg: float = 120.0,
    risk_verify: bool = True,
    risk_verify_threshold: float = 0.5,
    risk_optimize: bool = True,
    risk_optimize_iterations: int = 50,
    deviation_weight: float = 1.0,
    smoothness_weight: float = 0.01,
) -> SmoothResult:
    from scipy.interpolate import make_interp_spline

    path_xy = densify_polyline(path_xy, densify_control_points, densify_spacing)

    # ── Point 3: 急转弯处插入额外控制点 ──
    mandatory = None
    if angle_threshold_deg < 180.0:
        mandatory = detect_sharp_turns(path_xy, angle_threshold_deg)

    control_points = choose_control_points(path_xy, control_point_step, control_point_count, mandatory)
    segment_lengths = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
    keep = np.r_[True, segment_lengths > 1e-9]
    control_points = control_points[keep]
    actual_degree = min(degree, len(control_points) - 1)
    if actual_degree < 1:
        raise ValueError("Need at least two control points for B-spline smoothing.")

    # ── 构建初始 B 样条 ──
    chord_lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(control_points, axis=0), axis=1))]
    parameters = chord_lengths / chord_lengths[-1]
    spline = make_interp_spline(parameters, control_points, k=actual_degree, axis=0)
    samples = np.unique(np.r_[np.linspace(0.0, 1.0, sample_count), parameters])
    smooth_points = spline(samples)
    smooth_points[:, 0] = np.clip(smooth_points[:, 0], x_axis.min(), x_axis.max())
    smooth_points[:, 1] = np.clip(smooth_points[:, 1], y_axis.min(), y_axis.max())

    # ── Point 1: 平滑后风险验证 ──
    verification = None
    if risk_verify:
        verification = verify_smooth_risk(smooth_points, risk, x_axis, y_axis, risk_verify_threshold)

    # ── Point 2: 风险约束优化（仅在发现违规时触发） ──
    optimized = False
    if risk_optimize and verification is not None and verification["violation_count"] > 0:
        optimized_cp = risk_constrained_optimize(
            initial_control_points=control_points,
            risk=risk,
            x_axis=x_axis,
            y_axis=y_axis,
            degree=actual_degree,
            sample_count=sample_count,
            risk_weight=risk_weight,
            deviation_weight=deviation_weight,
            smoothness_weight=smoothness_weight,
            max_iterations=risk_optimize_iterations,
        )
        # 用优化后的控制点重建 B 样条
        chord_lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(optimized_cp, axis=0), axis=1))]
        if chord_lengths[-1] > 1e-9:
            parameters = chord_lengths / chord_lengths[-1]
            spline = make_interp_spline(parameters, optimized_cp, k=actual_degree, axis=0)
            samples = np.unique(np.r_[np.linspace(0.0, 1.0, sample_count), parameters])
            smooth_points = spline(samples)
            smooth_points[:, 0] = np.clip(smooth_points[:, 0], x_axis.min(), x_axis.max())
            smooth_points[:, 1] = np.clip(smooth_points[:, 1], y_axis.min(), y_axis.max())
            control_points = optimized_cp
            optimized = True

        # 优化后重新验证
        if risk_verify:
            verification = verify_smooth_risk(smooth_points, risk, x_axis, y_axis, risk_verify_threshold)

    distance, risk_cost, total_cost = continuous_path_metrics(
        smooth_points, risk, x_axis, y_axis, risk_weight
    )
    return SmoothResult(
        control_points, smooth_points, actual_degree,
        distance, risk_cost, total_cost,
        verification, optimized,
    )


def astar(
    risk: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    cell_size: float,
    risk_weight: float,
    obstacle_threshold: float | None,
) -> SearchResult:
    from B_Astar_Algorithm import line_mean_risk, run_b_astar

    rows, cols = risk.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError(f"Start index {start} is outside the risk matrix.")
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError(f"Goal index {goal} is outside the risk matrix.")

    risk_threshold = 0.5 if obstacle_threshold is None else obstacle_threshold
    _, path = run_b_astar(
        risk,
        start,
        goal,
        risk_threshold=risk_threshold,
    )

    distance = 0.0
    risk_cost = 0.0
    for point_a, point_b in zip(path, path[1:]):
        step_cells = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
        step_distance = step_cells * cell_size
        step_risk = line_mean_risk(risk, point_a, point_b)
        distance += step_distance
        risk_cost += step_risk * step_distance

    total_cost = distance + risk_weight * risk_cost
    return SearchResult(path, distance, risk_cost, total_cost)


def plot_result(
    risk: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    result: SearchResult,
    smooth_result: SmoothResult | None,
    start: tuple[int, int],
    goal: tuple[int, int],
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    path_rows = np.array([point[0] for point in result.path])
    path_cols = np.array([point[1] for point in result.path])
    path_x = x_axis[path_cols]
    path_y = y_axis[path_rows]

    plt.figure(figsize=(11, 7))
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    image = plt.imshow(risk, origin="lower", extent=extent, cmap="YlOrRd", aspect="equal")
    plt.colorbar(image, label="Risk")

    plt.plot(path_x, path_y, color="dodgerblue", linewidth=1.2, alpha=0.55, label="B-A* path")
    plt.scatter(path_x, path_y, c="dodgerblue", s=22, alpha=0.9, zorder=3, label=f"B-A* key nodes ({len(result.path)})")
    if smooth_result is not None:
        plt.plot(
            smooth_result.control_points[:, 0],
            smooth_result.control_points[:, 1],
            color="seagreen",
            marker="o",
            markersize=3.5,
            linewidth=0.9,
            alpha=0.8,
            label="B-spline control polygon",
        )
        plt.plot(
            smooth_result.smooth_points[:, 0],
            smooth_result.smooth_points[:, 1],
            color="blue",
            linewidth=2.4,
            label=f"B-spline smooth path (degree {smooth_result.degree})",
        )
        # 绘制风险违规点（红色圆点）
        if smooth_result.verification is not None and smooth_result.verification["violation_count"] > 0:
            violations = smooth_result.verification["violations"]
            v_x = [v["coord"][0] for v in violations]
            v_y = [v["coord"][1] for v in violations]
            plt.scatter(
                v_x, v_y, c="red", s=14, alpha=0.7, zorder=4,
                label=f"Risk violations ({len(violations)})",
            )
    plt.scatter(x_axis[start[1]], y_axis[start[0]], c="lime", edgecolors="black", s=80, label="Start", zorder=3)
    plt.scatter(x_axis[goal[1]], y_axis[goal[0]], c="cyan", edgecolors="black", s=80, label="Goal", zorder=3)

    if smooth_result is None:
        title = (
            f"B-A* risk-aware path | distance={result.distance:.1f} m, "
            f"risk_cost={result.risk_cost:.2f}, total={result.total_cost:.2f}"
        )
    else:
        opt_tag = " [optimized]" if smooth_result.optimized else ""
        viol_tag = ""
        if smooth_result.verification is not None and smooth_result.verification["violation_count"] > 0:
            viol_tag = f", violations={smooth_result.verification['violation_count']}"
        title = (
            f"B-A* + B-spline{opt_tag} | raw={result.distance:.1f} m, "
            f"smooth={smooth_result.distance:.1f} m, "
            f"smooth risk={smooth_result.risk_cost:.2f}{viol_tag}"
        )
    plt.title(title)
    plt.xlabel("Projected X")
    plt.ylabel("Projected Y")
    plt.legend(loc="best")
    plt.tight_layout()
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk-aware B-A* path planning on risk_matrix_2d.npy.")
    parser.add_argument("--risk", default="risk_matrix_2d.npy", help="Path to the 2D risk matrix .npy file.")
    parser.add_argument("--x-axis", default="x_unique.npy", help="Path to the unique projected X coordinates.")
    parser.add_argument("--y-axis", default="y_unique.npy", help="Path to the unique projected Y coordinates.")
    parser.add_argument("--risk-weight", type=float, default=80.0, help="Higher values avoid risky cells more strongly.")
    parser.add_argument("--obstacle-threshold", type=float, default=None, help="Treat cells >= this risk as impassable.")
    parser.add_argument("--start-index", type=parse_grid_point, default=None, help="Grid start as row,col.")
    parser.add_argument("--goal-index", type=parse_grid_point, default=None, help="Grid goal as row,col.")
    parser.add_argument("--start-coord", default=None, help="Projected coordinate start as x,y.")
    parser.add_argument("--goal-coord", default=None, help="Projected coordinate goal as x,y.")
    parser.add_argument("--no-smooth", action="store_true", help="Disable B-spline path smoothing.")
    parser.add_argument("--bspline-degree", type=int, default=3, help="B-spline degree. 3 is cubic.")
    parser.add_argument(
        "--control-point-step",
        type=int,
        default=1,
        help="Use one A* point every N points as a B-spline control point.",
    )
    parser.add_argument(
        "--control-point-count",
        type=int,
        default=None,
        help="Use exactly this many sampled B-spline control points; overrides --control-point-step.",
    )
    parser.add_argument(
        "--densify-control-points",
        type=int,
        default=100,
        help="Resample the B-A* polyline to this many equally spaced points before B-spline smoothing; use 0 to disable.",
    )
    parser.add_argument(
        "--densify-spacing",
        type=float,
        default=None,
        help="Resample the B-A* polyline at this projected-distance interval before smoothing; overrides --densify-control-points.",
    )
    parser.add_argument("--smooth-samples", type=int, default=800, help="Number of points sampled on the smooth curve.")
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=120.0,
        help="Angle threshold (degrees) for detecting sharp turns; smaller = less aggressive. Use 180 to disable.",
    )
    parser.add_argument("--no-risk-verify", action="store_true", help="Disable post-smoothing risk verification.")
    parser.add_argument(
        "--risk-verify-threshold",
        type=float,
        default=0.5,
        help="Risk threshold for post-smoothing verification.",
    )
    parser.add_argument("--no-risk-optimize", action="store_true", help="Disable risk-constrained control-point optimization.")
    parser.add_argument(
        "--risk-optimize-iterations",
        type=int,
        default=50,
        help="Maximum iterations for risk-constrained optimization.",
    )
    parser.add_argument(
        "--deviation-weight",
        type=float,
        default=1.0,
        help="Penalty weight for control-point deviation during risk optimization.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.01,
        help="Penalty weight for curvature during risk optimization.",
    )
    parser.add_argument("--no-show", action="store_true", help="Only print the path metrics; do not open a plot window.")
    args = parser.parse_args()

    risk = np.load(args.risk).astype(float)
    x_axis = np.load(args.x_axis)
    y_axis = np.load(args.y_axis)

    if risk.shape != (len(y_axis), len(x_axis)):
        raise ValueError(
            f"Risk matrix shape {risk.shape} does not match y/x axes {(len(y_axis), len(x_axis))}."
        )

    cell_size_x = float(np.median(np.diff(x_axis)))
    cell_size_y = float(np.median(np.diff(y_axis)))
    if not np.isclose(cell_size_x, cell_size_y):
        raise ValueError(f"Expected square cells, got dx={cell_size_x}, dy={cell_size_y}.")
    cell_size = cell_size_x

    default_start = (0, 0)
    default_goal = (risk.shape[0] - 1, risk.shape[1] - 1)

    start = args.start_index or default_start
    goal = args.goal_index or default_goal
    if args.start_coord:
        start = parse_coord_point(args.start_coord, x_axis, y_axis)
    if args.goal_coord:
        goal = parse_coord_point(args.goal_coord, x_axis, y_axis)

    result = astar(
        risk=risk,
        start=start,
        goal=goal,
        cell_size=cell_size,
        risk_weight=args.risk_weight,
        obstacle_threshold=args.obstacle_threshold,
    )

    print(f"Start index: {start}, coordinate: ({x_axis[start[1]]}, {y_axis[start[0]]})")
    print(f"Goal index: {goal}, coordinate: ({x_axis[goal[1]]}, {y_axis[goal[0]]})")
    print(f"Path points: {len(result.path)}")
    print(f"Distance: {result.distance:.2f} m")
    print(f"Risk cost: {result.risk_cost:.4f}")
    print(f"Total cost: {result.total_cost:.4f}")

    path_rows = np.array([point[0] for point in result.path])
    path_cols = np.array([point[1] for point in result.path])
    path_xy = np.column_stack((x_axis[path_cols], y_axis[path_rows]))

    smooth_result = None
    if not args.no_smooth:
        smooth_result = smooth_with_bspline(
            path_xy=path_xy,
            risk=risk,
            x_axis=x_axis,
            y_axis=y_axis,
            risk_weight=args.risk_weight,
            degree=args.bspline_degree,
            control_point_step=args.control_point_step,
            control_point_count=args.control_point_count,
            densify_control_points=args.densify_control_points,
            densify_spacing=args.densify_spacing,
            sample_count=args.smooth_samples,
            angle_threshold_deg=args.angle_threshold,
            risk_verify=not args.no_risk_verify,
            risk_verify_threshold=args.risk_verify_threshold,
            risk_optimize=not args.no_risk_optimize,
            risk_optimize_iterations=args.risk_optimize_iterations,
            deviation_weight=args.deviation_weight,
            smoothness_weight=args.smoothness_weight,
        )
        print(f"B-spline degree: {smooth_result.degree}")
        print(f"Control points: {len(smooth_result.control_points)}")
        print(f"Smooth samples: {len(smooth_result.smooth_points)}")
        print(f"Smooth distance: {smooth_result.distance:.2f} m")
        print(f"Smooth risk cost: {smooth_result.risk_cost:.4f}")
        print(f"Smooth total cost: {smooth_result.total_cost:.4f}")
        if smooth_result.optimized:
            print("Control points were risk-optimized.")
        if smooth_result.verification is not None:
            v = smooth_result.verification
            print(f"Risk verification: {v['violation_count']} violations ({v['violation_ratio']:.2%})")
            print(f"  Max risk on path: {v['max_risk']:.4f}, Mean risk: {v['mean_risk']:.4f}")

    if not args.no_show:
        plot_result(risk, x_axis, y_axis, result, smooth_result, start, goal)


if __name__ == "__main__":
    main()
