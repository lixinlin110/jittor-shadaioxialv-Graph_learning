# 基于图学习的动态推荐任务

本仓库面向第六届“计图”人工智能算法挑战赛赛道一：基于图学习的动态推荐任务。项目研究对象是时序图中的未来链接预测：给定历史交互三元组 `(source, destination, time)`，在测试阶段对每个 `source`、`time` 和 100 个候选 `destination` 进行概率打分与重排序，并按照比赛要求生成 `dataset1_result.csv`、`dataset2_result.csv` 和 `result.zip`。

本项目不是多模态 RAG 项目。仓库内容、代码、实验记录和报告材料均围绕动态图推荐、时序图学习、候选节点重排序和 JittorGeometric CRAFT baseline 展开。

## 研究题目

基于图学习的动态推荐任务研究：面向时序图未来链接预测的时间感知候选节点重排序策略。

## 研究背景

推荐系统、社交网络、交易网络、论文引用和网页跳转等场景中，用户或节点之间的交互关系会随时间持续变化。与静态图不同，时序图不仅包含图结构本身，还包含交互发生的时间顺序。模型需要同时理解节点历史行为、近期行为、重复交互、热门节点偏置以及候选节点之间的相对排序关系。

本赛题要求参赛者基于 Jittor/JittorGeometric 框架完成动态图推荐任务。官方提供了基于 JittorGeometric 的 CRAFT baseline，本项目在其基础上开展数据检查、基础排序方法、模型训练、候选节点重排序、多模型集成和自适应融合实验。

## 研究目标

1. 跑通官方 CRAFT baseline，并形成可复现的训练、推理和提交文件生成流程。
2. 完成数据读取、字段检查、时间排序、时间泄漏检查和候选集格式检查。
3. 实现历史频次、近期频次、时间衰减、重复交互等基础排序与重排序方法。
4. 基于 CRAFT 输出结果进行多随机种子、多配置模型集成，提升候选节点排序质量。
5. 建立 MRR、Recall@K、HitRate@K、NDCG@K 等指标的本地评测代码和实验记录模板。
6. 形成可用于中期检查、最终报告和 PPT 汇报的实验材料。

## 任务定义

训练集每行表示一次历史交互：

```text
source, destination, time
```

测试集每行包含：

```text
source, time, candidate_1, candidate_2, ..., candidate_100
```

模型需要输出 100 个候选节点与当前 `source` 在给定 `time` 发生交互的概率。比赛核心评价指标为 MRR，即真实目标节点在候选列表中排名越靠前，得分越高。

## 技术路线

```text
官方数据
-> 字段检查、ID 检查、时间排序、时间泄漏检查
-> 训练/验证/测试流程整理
-> 官方 CRAFT baseline 训练
-> 历史频次、近期频次、时间衰减等基础排序
-> 多种子 CRAFT 与候选重排序结果生成
-> 双模型/多模型融合
-> 置信度自适应融合
-> MRR/Recall@K/NDCG@K/HitRate@K 评估
-> dataset1_result.csv、dataset2_result.csv、result.zip
```

## 方法设计

| 方法模块 | 作用 | 与课题关系 |
| --- | --- | --- |
| CRAFT baseline | 使用 source 历史邻居序列和候选 destination 进行匹配打分 | 官方基线与主要深度模型 |
| 历史频次排序 | 统计 source 与 destination 的历史交互强度 | 对照传统推荐和热门度方法 |
| 近期频次排序 | 强化最近交互对预测时刻的影响 | 对应时序图的时间局部性 |
| 时间衰减 | 距离预测时间越近的历史交互权重越高 | 降低过早历史行为干扰 |
| 多随机种子训练 | 训练多个 CRAFT 模型，获得互补预测 | 降低单次训练波动 |
| 候选节点重排序 | 在模型分数基础上融合频次、重复交互和时间信号 | 面向比赛指标的排序优化 |
| 自适应融合 | 对不同样本按置信度调整融合比例 | 当前 A 榜最有效的提升策略 |

## 当前阶段结果

截至 2026-05-15 的实验记录：

| 阶段 | 方法 | A 榜反馈 |
| --- | --- | --- |
| 官方 CRAFT baseline 调参 | 增加 dataset1 训练轮数、保持 batch_size=200 | 从约 `1.042` 提升到约 `1.0878` |
| CRAFT 双模型融合 | 最优结果与新 raw 结果按比例融合 | 约 `1.12356` |
| 多样性模型融合 | 引入不同随机种子和邻居配置 | 约 `1.12663` |
| 新种子模型少量融合 | `best21.5%` 与 seed7777 模型融合 | 约 `1.12717` |
| 置信度自适应融合 | 根据样本分数差距调整融合权重 | 当前最好约 `1.12737` |

> A 榜分数用于阶段性实验反馈。最终报告中会同时记录本地验证指标、提交时间、参数配置、模型文件和生成脚本，保证过程可解释、可复现。

## 评价指标

本仓库已在 `src/metrics.py` 中实现以下指标：

| 指标 | 含义 |
| --- | --- |
| MRR | 真实节点排名倒数的平均值，是比赛核心指标 |
| Recall@K | 前 K 个候选中是否覆盖真实节点 |
| HitRate@K | 前 K 个候选中命中真实节点的比例 |
| NDCG@K | 考虑排名位置折损的排序质量指标 |

可使用脚本直接评测预测文件：

```bash
python scripts/evaluate_predictions.py --demo
```

或评测自己的验证集预测文件：

```bash
python scripts/evaluate_predictions.py \
  --input results/dataset1/validation_predictions.csv \
  --target_col destination \
  --ranked_col ranked_destinations
```

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── data.py
│   ├── infer.py
│   ├── metrics.py
│   ├── models.py
│   └── train.py
├── scripts/
│   ├── run_baseline.sh
│   ├── run_baseline.ps1
│   └── evaluate_predictions.py
├── configs/
│   ├── baseline_dataset1.json
│   ├── baseline_dataset2.json
│   └── time_aware_rerank.json
├── docs/
│   ├── data.md
│   ├── experiments.md
│   ├── evaluation_plan.md
│   ├── reproducibility.md
│   ├── midterm_report.md
│   └── compute_request.md
├── results/
│   ├── README.md
│   └── experiment_summary.md
└── data/
    └── .gitkeep
```

## 环境配置

建议使用 Python 3.10。

```bash
conda create -n jittor-graph python=3.10
conda activate jittor-graph
pip install -r requirements.txt
```

JittorGeometric 建议按官方仓库源码安装：

```bash
pip install git+https://github.com/Jittor/jittor.git
git clone https://github.com/AlgRUC/JittorGeometric.git
cd JittorGeometric
pip install .
```

## 数据放置

比赛原始数据较大，不直接提交到 GitHub。推荐目录：

```text
data/
├── dataset1/
│   ├── train.csv
│   └── test.csv
└── dataset2/
    ├── train.csv
    └── test.csv
```

如果需要本地验证，也可以额外放置：

```text
valid.csv
```

字段格式和候选集要求见 [docs/data.md](docs/data.md)。

## 快速运行

不依赖真实比赛数据的演示流程：

```bash
python main.py --demo --dataset demo
```

使用真实数据训练轻量时间感知 reranker：

```bash
python main.py --dataset dataset1 --data_dir data --output_dir results
python main.py --dataset dataset2 --data_dir data --output_dir results
```

官方 CRAFT baseline 的实际比赛训练在本地 JittorGeometric 环境中执行，命令风格如下：

```bash
python main.py --dataset dataset1 --epochs 70 --early_stop 12 --batch_size 200
python main.py --dataset dataset2 --epochs 20 --early_stop 5 --batch_size 200
```

也可以运行脚本：

```bash
bash scripts/run_baseline.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
```

## 推理与结果生成

轻量 reranker 推理示例：

```bash
python -m src.infer \
  --dataset dataset1 \
  --model_path results/dataset1/time_aware_reranker.json \
  --candidate_file data/dataset1/test.csv \
  --output_file results/dataset1/submission.csv
```

比赛提交要求最终生成：

```text
result.zip
├── dataset1_result.csv
└── dataset2_result.csv
```

每个 csv 文件中每一行包含 100 个概率值，保留 8 位小数。

## 实验设计

详细实验计划见 [docs/evaluation_plan.md](docs/evaluation_plan.md)。当前实验分为：

1. 官方 CRAFT baseline 复现。
2. 历史频次、近期频次、时间衰减等基础排序对照。
3. CRAFT 单模型调参。
4. 多随机种子 CRAFT 集成。
5. 固定权重融合消融。
6. 置信度自适应融合消融。
7. 提交结果与本地验证结果对照记录。

## 可复现性

复现说明见 [docs/reproducibility.md](docs/reproducibility.md)。每次有效实验需要保存：

```text
config.json
run_metadata.json
metrics.json
dataset1_result.csv
dataset2_result.csv
result.zip
model checkpoint 或模型参数
实验日志
```

对于 A 榜提交结果，需要记录提交时间、文件路径、融合比例、随机种子、训练轮数、验证指标和平台反馈分数。

## 中期检查对应关系

| 检查项 | 仓库对应内容 |
| --- | --- |
| 研究题目和目标 | README 的研究题目、研究目标 |
| 技术路线 | README 技术路线与 docs/midterm_report.md |
| 数据说明 | docs/data.md |
| 实验设计 | docs/evaluation_plan.md |
| 评价指标代码 | src/metrics.py、scripts/evaluate_predictions.py |
| 运行说明 | README 环境配置、快速运行、推理与结果生成 |
| 阶段结果 | results/experiment_summary.md、docs/experiments.md |
| 可复现性 | docs/reproducibility.md |

## 小组分工

| 成员 | 主要工作 |
| --- | --- |
| 李鑫霖 | 环境搭建、JittorGeometric baseline、模型训练、重排序与融合实验、仓库维护 |
| 队友 | 文献阅读、研究现状整理、实验表格与 PPT 材料补充 |
| 协作任务 | 结果复核、报告撰写、答辩演示、参考文献整理 |

## 参考资料

- 第六届“计图”人工智能算法挑战赛：https://cg.cs.tsinghua.edu.cn/jittor/news/2026-4-9-13-44-00-00-JittorComp6th/
- Jittor：https://github.com/Jittor/jittor
- JittorGeometric：https://github.com/AlgRUC/JittorGeometric
- CRAFT paper：https://arxiv.org/abs/2505.19408
