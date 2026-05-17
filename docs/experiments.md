# 实验记录

本文件整理根目录原始文件 `调参信息记录` 中的 baseline 提交记录，便于中期汇报展示和后续复现实验。当前阶段主要结论是：官方 CRAFT baseline 已经跑通，dataset1 对训练轮数和 early stopping 较敏感，batch size 从 200 降到 100 后分数下降。

## 官方 Baseline 参数

| 参数 | 当前记录 |
| --- | --- |
| model | CRAFT |
| num_neighbors | 30 |
| hidden_size | 64 |
| n_layers | 2 |
| n_heads | 2 |
| learning_rate | 0.0001 |

## 提交记录

| 版本 | 代码状态 | dataset1 参数 | dataset2 参数 | A 榜分数 |
| --- | --- | --- | --- | --- |
| v1 | 官方 baseline | epochs=20, early_stop=5, batch_size=200 | epochs=5, early_stop=3, batch_size=200 | 1.042 |
| v2 | 官方 baseline，未改代码 | 不变 | epochs=20, early_stop=5, batch_size=200 | 1.057 |
| v3 | 官方 baseline，未改代码 | epochs=20, early_stop=5, batch_size=200 | epochs=20, early_stop=5, batch_size=200 | 1.061 |
| v4 | 官方 baseline，未改代码 | epochs=30, early_stop=8, batch_size=200 | 不变 | 1.064 |
| v5 | 官方 baseline，未改代码 | epochs=40, early_stop=10, batch_size=200 | 不变 | 1.068 |
| v6 | 官方 baseline，未改代码 | epochs=50, early_stop=10, batch_size=200 | 不变 | 1.070811 |
| v7 | 官方 baseline，未改代码 | epochs=60, early_stop=10, batch_size=200 | 不变 | 1.070847 |
| v8 | 官方 baseline，未改代码 | epochs=30, early_stop=8, batch_size=100 | 不变 | 1.0669 |
| v9 | 官方 baseline，未改代码 | epochs=60, early_stop=10, batch_size=100 | 不变 | 1.0653 |
| v10 | 官方 baseline，未改代码 | epochs=70, early_stop=12, batch_size=200 | 不变 | 1.0878 |
| v11 | 官方 baseline，未改代码 | epochs=70, early_stop=12, batch_size=200，实际 early stop 于第 48 epoch | 不变 | 1.068216648455683 |

## 阶段性观察

1. dataset1 增大 epochs 并放宽 early_stop 后，A 榜分数总体提升，说明 baseline 仍有训练不足或收敛不稳定问题。
2. batch_size=100 的两次提交低于 batch_size=200，后续默认保留 batch_size=200 作为基础设置。
3. v10 与 v11 使用相同显式参数但得分差异较大，说明需要固定随机种子、保存日志和 checkpoint，并记录实际 early stopping 位置。
4. 当前记录均为“未改官方 baseline 代码”，后续提升空间主要来自数据处理、时间感知模块、候选重排序和集成策略。

## 后续消融计划

| 实验组 | 说明 | 目标 |
| --- | --- | --- |
| Official CRAFT | 官方 baseline 参数复现 | 作为主对照 |
| CRAFT + time window | 按时间窗口构建动态图快照 | 验证局部时间结构贡献 |
| CRAFT + time decay | 对历史交互加入时间衰减权重 | 降低过早历史交互干扰 |
| CRAFT + recent neighbors | 调整近期邻居采样策略 | 分析 `num_neighbors` 与近期行为的影响 |
| CRAFT + rerank | baseline 得分与时间感知重排序融合 | 提升候选 destination 排序 |
| Full strategy | 时间窗口 + 时间衰减 + 近期邻居 + rerank | 形成最终提交方案 |
