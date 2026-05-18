# 实验记录

本文档整理项目目前的主要实验路线、参数变化和阶段结论。完整提交文件、模型参数和临时日志不直接提交到 GitHub，只在本地实验目录中归档；本文件保留可用于复现和汇报的关键信息。

## 1. 官方 Baseline 参数

| 参数 | 当前基线设置 |
| --- | --- |
| model | CRAFT |
| num_neighbors | 30 或 40 |
| hidden_size | 64 |
| n_layers | 2 |
| n_heads | 2 |
| learning_rate | 0.0001 |
| batch_size | 200 |

## 2. 官方 Baseline 提交记录

| 版本 | 代码状态 | dataset1 参数 | dataset2 参数 | A 榜反馈 |
| --- | --- | --- | --- | --- |
| v1 | 官方 baseline | epochs=20, early_stop=5 | epochs=5, early_stop=3 | 约 1.042 |
| v2 | 官方 baseline | 不变 | epochs=20, early_stop=5 | 约 1.057 |
| v3 | 官方 baseline | epochs=20, early_stop=5 | epochs=20, early_stop=5 | 约 1.061 |
| v4 | 官方 baseline | epochs=30, early_stop=8 | 不变 | 约 1.064 |
| v5 | 官方 baseline | epochs=40, early_stop=10 | 不变 | 约 1.068 |
| v6 | 官方 baseline | epochs=50, early_stop=10 | 不变 | 约 1.0708 |
| v7 | 官方 baseline | epochs=60, early_stop=10 | 不变 | 约 1.0708 |
| v8 | 官方 baseline | batch_size=100 | 不变 | 约 1.0669 |
| v9 | 官方 baseline | epochs=70, early_stop=12, batch_size=200 | 不变 | 约 1.0878 |

阶段观察：

1. dataset1 对训练轮数和 early stopping 比较敏感，增加训练轮数后有明显提升。
2. batch_size=200 比 batch_size=100 更稳定。
3. 相同显式参数多次训练存在波动，因此后续实验需要固定随机种子、保存 checkpoint 和记录实际 early stopping 位置。

## 3. 基础排序与数据处理实验

围绕开题报告中“历史频次、近期频次等基础排序方法”的要求，已完成以下处理流程：

| 模块 | 内容 |
| --- | --- |
| 字段检查 | 检查 `src/source`、`dst/destination`、`time`、候选列 |
| 时间排序 | 确认 dataset1、dataset2 均已按时间顺序排列 |
| 时间泄漏检查 | 构造历史邻居时只使用预测时间之前的交互 |
| 历史频次 | 统计 source-destination 历史交互次数 |
| 近期频次 | 使用最近窗口或最近邻居序列统计偏好 |
| 时间衰减 | 对更近的历史交互给予更高权重 |
| 实验记录模板 | 记录配置、模型、指标、A 榜反馈和结论 |

这些模块主要用于解释模型结果、构建重排序特征和支持报告中的实验设计。

## 4. 双模型与多模型融合记录

| 阶段 | 实验说明 | A 榜反馈 |
| --- | --- | --- |
| round8/round9 | 历史最优结果与新 raw 结果固定比例融合 | 约 1.11996 到 1.12356 |
| round10 | 三路融合加入少量频次信号 | 约 1.12334，未超过双模型最优 |
| round13 | 引入多样性 CRAFT 结果，搜索新模型比例 | 约 1.12663 |
| round17 | best21.5% 与 seed7777 模型低比例融合 | 1.1271744963034764 |
| round18 | rank-space 融合 | 1.124651558251298，效果较差 |
| round19 | 置信度自适应融合 | 1.1273681134893112 |
| round20 | 提高 extra 融合权重 | 1.1273511801941303 |

当前最好：

```text
round19_adapt_best215_seed7777_b02_e06_t005
A 榜反馈：1.1273681134893112
```

## 5. 关键结论

1. 单纯调 CRAFT 训练轮数可以提升 baseline，但提升很快进入瓶颈。
2. 固定比例融合能显著超过单模型，但在 50% 附近出现饱和。
3. 不同随机种子和不同邻居配置的 CRAFT 模型具有互补性。
4. rank-space 融合在当前数据上表现较差，后续不作为主线。
5. 置信度自适应融合优于固定权重融合，是当前最值得继续优化的方向。

## 6. 后续实验计划

| 实验组 | 说明 | 目标 |
| --- | --- | --- |
| CRAFT + stronger diversity | 训练结构差异更明显的新 CRAFT 配置 | 获取更有互补性的预测 |
| Adaptive Ensemble Sweep | 围绕 round19 参数做小范围搜索 | 稳定提升当前最优 |
| Frequency Ablation | 单独测试历史频次、近期频次、时间衰减贡献 | 支撑实验报告 |
| Validation Correlation | 比较本地 MRR 和 A 榜反馈 | 降低盲目提交成本 |
| Final Repro Package | 保存 config、checkpoint、result.zip、日志 | 支撑开源复现 |
