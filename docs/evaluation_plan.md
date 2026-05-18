# 实验设计与评价指标

本文档用于说明本项目如何评价动态图推荐模型，解决“缺少实验设计和评价展示”的问题。

## 1. 实验目标

本项目的实验目标不是只追求一次 A 榜提交分数，而是建立一套可复现的时序图候选节点排序实验流程：

1. 复现官方 CRAFT baseline。
2. 构建历史频次、近期频次、时间衰减等非深度排序基线。
3. 比较不同训练轮数、随机种子、邻居数量和融合权重对 MRR 的影响。
4. 记录本地验证指标和 A 榜反馈之间的关系。
5. 为最终报告和答辩提供可解释的实验表格。

## 2. 数据划分原则

官方提供的 `dataset1` 和 `dataset2` 已经按时间顺序排列。实验中保持官方数据顺序，不随机打乱。

本地验证时采用时间切分：

```text
前 85% 交互 -> 训练
后 15% 交互 -> 验证
```

验证和测试阶段只允许使用预测时间之前的历史交互，避免时间泄漏。

## 3. 评价指标

### MRR

MRR 是比赛核心指标。若真实目标节点在候选列表中排名为 `rank`，则该样本得分为：

```text
RR = 1 / rank
```

所有样本取平均：

```text
MRR = mean(RR)
```

真实节点越靠前，MRR 越高。

### Recall@K

判断真实节点是否出现在前 K 个候选中：

```text
Recall@K = hit_in_top_k / number_of_queries
```

### HitRate@K

本任务每个查询只有一个真实目标节点，因此 HitRate@K 与 Recall@K 等价，用于更直观展示命中率。

### NDCG@K

NDCG@K 考虑命中位置的折损。真实节点越靠前，贡献越大。

指标实现位置：

```text
src/metrics.py
scripts/evaluate_predictions.py
```

## 4. 对照实验组

| 实验组 | 说明 | 目的 |
| --- | --- | --- |
| Official CRAFT | 官方 baseline 原始结构 | 主对照 |
| History Frequency | 根据 source-destination 历史交互次数排序 | 验证历史偏好 |
| Recent Frequency | 根据最近窗口内交互次数排序 | 验证近期行为 |
| Time Decay | 对历史交互加入时间衰减权重 | 验证时间距离影响 |
| CRAFT Tuned | 调整 epochs、early_stop、batch_size、num_neighbors | 获得稳定深度模型 |
| Multi-seed CRAFT | 多随机种子训练并融合 | 降低训练波动 |
| Fixed-weight Ensemble | 固定比例融合多个预测文件 | 观察融合比例影响 |
| Adaptive Ensemble | 根据样本置信度调整融合比例 | 当前效果最好的方法 |

## 5. 消融实验设计

固定官方 CRAFT baseline 后，逐步加入模块：

```text
CRAFT
CRAFT + 历史频次
CRAFT + 近期频次
CRAFT + 时间衰减
CRAFT + 多随机种子
CRAFT + 固定权重融合
CRAFT + 自适应融合
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
融合比例
本地验证 MRR
A 榜反馈分数
结果文件路径
备注
```

## 6. 当前阶段结论

从已提交实验看，单纯调参提升有限。更有效的方向是构造具有差异性的 CRAFT 模型，再进行可复现的融合和自适应重排序。

当前最好 A 榜反馈来自：

```text
round19_adapt_best215_seed7777_b02_e06_t005
score = 1.1273681134893112
```

该结果说明：不同随机种子模型与已有最优结果存在互补性，自适应融合比简单固定比例融合更有效。

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

脚本会输出：

```json
{
  "MRR": 0.75,
  "Recall@1": 0.5,
  "HitRate@1": 0.5,
  "NDCG@1": 0.5
}
```
