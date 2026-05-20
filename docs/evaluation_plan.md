# 实验设计与评价指标

本文档说明本项目如何按照开题报告要求，评价动态图推荐模型和时间感知候选节点重排序策略。

## 1. 实验目标

实验目标不是只追求单次 A 榜分数，而是建立一套可复现、可解释的时序图候选节点排序流程：

1. 复现官方 JittorGeometric CRAFT baseline。
2. 构建历史频次、近期频次、时间衰减等基础排序基线。
3. 补充 BPR/矩阵分解轻量对照基线，并将 LightGCN 作为后续静态图推荐对照，比较传统协同过滤、静态图推荐和动态时序图模型的差异。
4. 比较不同训练轮数、随机种子、邻居数量、融合权重和自适应策略对 MRR 的影响。
5. 记录本地验证指标、A 榜反馈、运行时间和结果文件路径，形成实验报告和 PPT 可用表格。

## 2. 数据划分原则

官方 `dataset1` 和 `dataset2` 已按时间顺序排列。实验中保持官方顺序，不随机打乱。

本地验证采用时间切分：

```text
前 85% 交互 -> 训练
后 15% 交互 -> 验证
```

验证和测试阶段只允许使用预测时间之前的历史交互，避免时间泄漏。测试集的 100 个候选 destination 由官方给定，不自行替换候选集合。

## 3. 评价指标

| 指标 | 用途 | 状态 |
| --- | --- | --- |
| MRR | 比赛核心指标，真实节点排名越靠前越好 | 已实现 |
| Recall@K | 判断真实节点是否进入前 K 个候选 | 已实现 |
| HitRate@K | 单目标场景下与 Recall@K 等价，便于展示命中率 | 已实现 |
| NDCG@K | 考虑命中位置折损的排序指标 | 已实现 |
| AUC/AP | 用于 CRAFT 验证阶段的正负样本区分能力评估 | CRAFT 验证流程已使用 |
| 训练时间 | 衡量不同模型和参数配置的计算成本 | 实验记录中补充 |
| 推理时间 | 衡量生成提交文件的效率 | 实验记录中补充 |
| 显存/内存占用 | 评估大数据集上的可运行性 | 实验记录中补充 |
| 重复交互命中率 | 分析模型是否依赖历史重复连接 | 后续分析 |
| 新交互命中率 | 分析模型对未重复关系的泛化能力 | 后续分析 |
| 热门偏置 | 分析模型是否过度倾向热门 destination | 后续分析 |

指标实现位置：

```text
src/metrics.py
scripts/evaluate_predictions.py
craft_main.py
```

## 4. 对照实验组

| 实验组 | 说明 | 与开题报告关系 | 当前状态 |
| --- | --- | --- | --- |
| History Frequency | 根据 source-destination 历史交互次数排序 | 历史频次基线 | 已完成 |
| Recent Frequency | 根据最近窗口内交互次数排序 | 近期频次基线 | 已完成 |
| Time Decay | 对越近的历史交互赋予越高权重 | 时间感知重排序 | 已完成 |
| BPR/MF | 基于隐式反馈的 pairwise 排序或矩阵分解 | 传统推荐对照 | 已有轻量脚本 |
| LightGCN | 静态图协同过滤推荐模型 | 图推荐对照 | 计划补充 |
| Official CRAFT | 官方 JittorGeometric baseline | 主要深度模型 | 已完成 |
| CRAFT Tuned | 调整 epochs、early_stop、batch_size、num_neighbors | 深度模型调参 | 已完成 |
| Multi-seed CRAFT | 多随机种子训练并融合 | 降低训练波动 | 已完成 |
| Fixed-weight Ensemble | 固定比例融合多个预测文件 | 融合消融 | 已完成 |
| Adaptive Ensemble | 按样本置信度调整融合强度 | 当前最优重排序策略 | 已完成 |

## 5. 消融实验设计

固定官方 CRAFT baseline 后，逐步加入模块：

```text
CRAFT
CRAFT + 历史频次
CRAFT + 近期频次
CRAFT + 时间衰减
CRAFT + 多随机种子
CRAFT + 固定权重融合
CRAFT + 置信度自适应融合
```

后续补充对照：

```text
BPR/MF
LightGCN
CRAFT vs BPR/MF vs LightGCN
CRAFT + 重排序 vs CRAFT 原始输出
```

BPR/MF 对照实验入口：

```bash
python scripts/run_bpr_baseline.py --dataset dataset1 --data_dir data --output_dir results --epochs 5 --factors 64
python scripts/run_bpr_baseline.py --dataset dataset2 --data_dir data --output_dir results --epochs 5 --factors 64
```

每个实验至少记录：

```text
实验编号
数据集
随机种子
训练轮数
early_stop
batch_size
num_neighbors
融合比例或自适应参数
本地验证 MRR/Recall@K/NDCG@K/HitRate@K
AUC/AP（如适用）
训练时间与推理时间
A 榜反馈分数
结果文件路径
备注
```

## 6. 当前阶段结论

截至当前阶段，单纯调参提升有限；更有效的方向是构造具有差异性的 CRAFT 模型，再进行可复现的模型集成和置信度自适应重排序。

当前最好 A 榜反馈来自：

```text
round19_adapt_best215_seed7777_b02_e06_t005
score = 1.1273681134893112
```

该结果说明不同随机种子模型与已有最优结果存在互补性，自适应融合比简单固定比例融合更有效。

## 7. 评测脚本示例

使用内置 demo：

```bash
python scripts/evaluate_predictions.py --demo
```

评测验证集预测文件：

```bash
python scripts/evaluate_predictions.py \
  --input results/dataset1/validation_predictions.csv \
  --target_col destination \
  --ranked_col ranked_destinations
```

`validation_predictions.csv` 示例：

```csv
source,time,destination,ranked_destinations
u1,100,v3,"v9 v3 v4 v7"
u2,101,v8,"v8 v1 v2 v6"
```

脚本会输出 MRR、Recall@K、HitRate@K 和 NDCG@K 等指标。
