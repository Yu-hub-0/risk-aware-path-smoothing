# E1 风险约束平滑实验（开发批次）

本目录是与原 `experiment/Path_平滑`、`experiment/MPC` 分开的实验副本。`source/` 保存所需的原始算法文件；原目录不改动。唯有实验副本中的 `B_Astar_Algorithm.py` 增加了对斜向穿角和 Bresenham 漏格的保守检查。`run_e1.py` 在现有 B-A* 搜索与路径重采样基础上，提供 P1/P2/P3 的可复现对照、后验风险验收与局部控制点补入；`run_control_e1.py` 使用复制的 17 状态全动力学 NMPC 模拟器。

## 实验对象与解释范围

- 三类 40×40 栅格合成场景，每格 1 m：转角、S 弯、窄通道；每类 30 个由固定种子生成的参数化实例。三类场景仍共用基本模板，本批次属于实现开发与数值核查，不能代替更广泛地图上的独立正式测试。
- P0：B-A* 折线，仅作为几何参照；P1：普通三次 B-spline；P2：增加与稿件一致的超阈值风险惩罚；P3：在 P2 上加入违规附近的 B-A* 节点补入和有限重试。
- 合格参考需要风险长度为零且通过速度、切向加速度、横向加速度限制。控制实验只运行合格参考，但所有拒绝路径仍保留在 `control_results.csv` 分母中。
- 风险验收沿曲线按 0.005 m 间距独立采样；`resolution_check.csv` 用 0.0025 m 复核被接受的曲线。本检查是数值证据，不构成连续曲线的严格数学证明。
- 连续时间参考由三次几何 B-spline 与五次时间样条复合，端点速度和加速度为零，控制器按链式法则接收位置、速度、加速度。若局部时间样条不单调，参考会被拒绝。所有合格参考再按时长整体放慢以满足运动学限制。
- C0 使用 17 状态 NMPC、0.05 s 控制周期、0.005 s 动力学积分、20 步预测、40 次最大求解迭代。当前批次是**暂停等待优化的离线闭环仿真**；求解超过 50 ms 的比例被明确报告。

原始风险地图间距为 100 m，因此不直接用它生成 2 m/s 四旋翼的闭环轨迹；本实验的米级合成地图用于控制比较。将实际地图转换为真实飞行尺度需要另外冻结明确的尺度规则。

## 复现

在论文目录 PowerShell 中使用用户指定的环境：

```powershell
$uavPython = 'C:\Users\yujin\.conda\envs\uav\python.exe'
& $uavPython experiment\paper_experiments\E1_path\run_e1.py
& $uavPython experiment\paper_experiments\E1_path\check_resolution_e1.py
& $uavPython experiment\paper_experiments\E1_path\plot_e1.py
& $uavPython experiment\paper_experiments\E1_path\run_control_e1.py
& $uavPython experiment\paper_experiments\E1_path\summarize_e1.py
& $uavPython experiment\paper_experiments\E1_path\plot_control_e1.py
```

控制任务可用 `--num-shards 4 --shard-id 0..3` 并行运行，各自使用不同 `--out`；四份 CSV 齐全后运行汇总脚本。

## 数据与图表

- `results_005/case_results.csv`：全部 90×4 条几何/参考记录。
- `results_005/resolution_check.csv`：被接受路径的更密风险检查。
- `results_005/control_shards/shard_*/control_results.csv`：全部 90×3 条控制尝试或拒绝原因；`logs/` 保存逐时刻状态与求解记录。
- `results_005/summary_by_method.csv`、`paired_effects.csv`：方法统计与按案例配对差值，95% 区间以案例为单位计算。
- `Fig_E1_paths`：相同路径的局部风险对照；`Fig_E1_quantitative`：风险、合格参考比例和航程；`Fig_E1_control_corner_00_P3`：单案例闭环轨迹、跟踪误差和求解时延。每图有 SVG、PDF、600 dpi TIFF 和预览 PNG。

图注建议明确：合格比例的误差线为二项精确 95% 区间；风险和长度的误差线为案例级 bootstrap 95% 区间，`n=90` 个参数化任务。控制图是一个预先固定的实现核查案例，不代表总体平均效果。所有求解超时结果保留，不可称为实时通过。

## 本次开发批次结果

90 个配对任务中，P1/P2/P3 分别生成 27/23/75 条通过风险与运动学筛查的参考；P0 仅作为几何参照。P3 相比 P1 的合格比例差为 +53.3 个百分点，案例级配对 bootstrap 95% 区间为 +42.2 至 +64.4 个百分点。P3 的平均超阈值路径长度比例比 P1 低 1.116 个百分点（95% 区间 −1.331 至 −0.907）。P2 的合格数低于 P1，说明当前局部风险惩罚不能单独可靠解决切角。全部 125 条被接受的参考在 0.0025 m 复核中没有新增越界。

270 条控制方法记录齐全：125 条参考被执行，145 条在参考生成阶段拒绝。理想且名义/真实模型一致的条件下，执行轨迹的平均位置 RMSE 为 0.00233 m；NMPC 采样时刻未观察到实际路径风险越界，优化求解成功率为 100%。但平均 55.7% 的求解步耗时超过 50 ms，因此这组数据只能说明离线闭环中的跟踪表现，不能说明实时控制能力。各方法的 RMSE 是在各自通过筛查的条件子集上计算，不能被直接当作同一批任务上的公平配对控制增益。

运行环境：用户指定的 `uav`（Python 3.11.15、NumPy 2.4.4、SciPy 1.17.1、Matplotlib 3.11.1、CasADi 3.7.2）。`figure_legends.md` 包含对应的英文图注草稿。

## 尚未解决的发表前限制

本批次的风险地图与任务来自有限的三类合成模板；应增加不参与开发的实际地图或更多结构变化的独立任务。当前 B-A* 复制件的保守格点检查比原文件严格，论文方法描述需要明确这个实现细节。P3 的节点补入是局部重采样加有限重试，若要逐字声称“原关键点插入”，还需核对插入节点与 B-A* 原始节点的对应关系。闭环风险检查目前基于 NMPC 采样时刻，实体动力学积分时刻的风险评价需另行加密。OV、IMM 及冲突重规划均未混入 E1 的 C0 比较。
