# 基于图学习的动态推荐任务

本仓库面向第六届“计图”人工智能算法挑战赛赛道一：基于图学习的动态推荐任务。项目研究对象是时序图未来链接预测：给定历史交互三元组 `(source, destination, time)`，在测试阶段对每个 `source`、`time` 和 100 个候选 `destination` 进行概率打分与重排序，并生成 `dataset1_result.csv`、`dataset2_result.csv` 和 `result.zip`。

仓库内容、代码和实验记录均围绕 Jittor/JittorGeometric、CRAFT baseline、动态图推荐、候选节点重排序和模型集成展开。

## 研究题目

基于图学习的动态推荐任务研究：面向时序图未来链接预测的时间感知候选节点重排序策略。

## 研究背景与问题定义

推荐系统、社交网络、交易网络、论文引用和网页跳转等场景中，节点之间的交互关系会随时间持续变化。与静态图不同，时序图不仅包含图结构，还包含交互顺序。模型需要理解节点历史行为、近期行为、重复交互、热门节点偏置以及候选节点之间的相对排序关系。

训练集每行表示一次历史交互：

```text
source, destination, time
```

测试集每行包含：

```text
source, time, candidate_1, candidate_2, ..., candidate_100
```

模型输出 100 个候选节点的交互概率。比赛核心评价指标为 MRR，真实目标节点排名越靠前，得分越高。

## 研究目标

1. 跑通官方 JittorGeometric CRAFT baseline，形成可复现的训练、推理和提交文件生成流程。
2. 完成数据读取、字段检查、时间排序、时间泄漏检查和候选集格式检查。
3. 实现历史频次、近期频次、时间衰减、重复交互等基础排序与重排序方法。
4. 将 BPR/矩阵分解、LightGCN 作为后续对照基线，比较传统协同过滤、静态图推荐和动态时序图模型。
5. 基于 CRAFT 输出结果进行多随机种子、多配置模型集成，提升候选节点排序质量。
6. 建立 MRR、Recall@K、HitRate@K、NDCG@K、AUC/AP、训练时间、推理时间等实验记录和评价流程。
7. 形成可用于中期检查、最终报告和 PPT 汇报的实验材料。

## 技术路线

```text
官方数据
-> 字段检查、ID 检查、时间排序、时间泄漏检查
-> 按时间划分训练/验证/测试流程
-> 历史频次、近期频次、时间衰减等基础排序
-> BPR/矩阵分解、LightGCN 等计划对照基线
-> 官方 CRAFT/JittorGeometric baseline 训练
-> 多随机种子 CRAFT 与候选重排序结果生成
-> 固定权重融合与置信度自适应融合
-> MRR/Recall@K/NDCG@K/HitRate@K/AUC 评估
-> dataset1_result.csv、dataset2_result.csv、result.zip
```

## 方法设计

| 方法模块 | 作用 | 与课题关系 | 当前状态 |
| --- | --- | --- | --- |
| History Frequency | 统计 source-destination 历史交互次数 | 传统频次基线 | 已完成 |
| Recent Frequency | 强化最近窗口内的交互行为 | 近期行为建模 | 已完成 |
| Time Decay | 距预测时间越近的历史交互权重越高 | 时间感知重排序 | 已完成 |
| BPR/矩阵分解 | 基于隐式反馈的 pairwise 排序对照 | 传统推荐基线 | 已有轻量对照脚本 |
| LightGCN | 静态图协同过滤推荐模型 | 图推荐对照 | 计划补充 |
| CRAFT baseline | 使用 source 历史邻居序列和候选 destination 打分 | 官方主深度模型 | 已完成 |
| Multi-seed CRAFT | 训练多个 CRAFT 模型获得互补预测 | 降低单次训练波动 | 已完成 |
| Candidate Reranking | 融合模型分数、频次、时间和重复交互信号 | 面向 MRR 的排序优化 | 已完成 |
| Adaptive Ensemble | 按样本置信度调整融合比例 | 当前最有效策略 | 已完成 |

## 当前阶段结果

截至当前实验记录，最好 A 榜反馈为：

```text
round19_adapt_best215_seed7777_b02_e06_t005
score = 1.1273681134893112
```

阶段结果汇总：

| 阶段 | 方法 | A 榜反馈 |
| --- | --- | --- |
| 官方 CRAFT baseline 调参 | 增加 dataset1 训练轮数、保持 batch_size=200 | 从约 `1.042` 到约 `1.0878` |
| CRAFT 双模型融合 | 最优结果与新 raw 结果按比例融合 | 约 `1.12356` |
| 多样性模型融合 | 引入不同随机种子和邻居配置 | 约 `1.12663` |
| 新种子模型少量融合 | `best21.5%` 与 seed7777 模型融合 | 约 `1.12717` |
| 置信度自适应融合 | 根据样本分数差距调整融合权重 | 当前最好约 `1.12737` |

A 榜分数只作为阶段性反馈。最终报告会同时记录本地验证指标、提交时间、参数配置、模型文件和生成脚本，保证过程可解释、可复现。

## 评价指标

指标代码位于 `src/metrics.py` 和 `scripts/evaluate_predictions.py`。

| 指标 | 含义 | 用途 |
| --- | --- | --- |
| MRR | 真实节点排名倒数的平均值 | 比赛核心指标 |
| Recall@K | 前 K 个候选中是否覆盖真实节点 | 排序召回分析 |
| HitRate@K | 前 K 个候选中命中真实节点的比例 | 展示命中率 |
| NDCG@K | 考虑排名位置折损的排序质量指标 | 排序质量分析 |
| AUC/AP | 正负样本区分能力 | CRAFT 验证阶段辅助指标 |
| 训练时间/推理时间 | 计算成本 | 工程可复现分析 |
| 重复/新交互命中率 | 分析模型依赖历史重复连接的程度 | 错误分析和消融 |

运行指标 demo：

```bash
python scripts/evaluate_predictions.py --demo
```

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── main.py                    # 轻量时间感知 reranker demo
├── craft_main.py              # 官方 CRAFT/JittorGeometric 训练与推理入口
├── src/
│   ├── data.py
│   ├── infer.py
│   ├── metrics.py
│   ├── models.py
│   └── train.py
├── scripts/
│   ├── evaluate_predictions.py
│   ├── make_result_zip.py
│   ├── run_bpr_baseline.py
│   ├── run_baseline.sh
│   └── run_baseline.ps1
├── configs/
├── docs/
│   ├── data.md
│   ├── experiments.md
│   ├── evaluation_plan.md
│   ├── reproducibility.md
│   ├── midterm_report.md
│   └── opening_report_alignment.md
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

官方 CRAFT baseline 的实际比赛训练入口为 `craft_main.py`：

```bash
python craft_main.py --dataset dataset1 --epochs 70 --early_stop 12 --batch_size 200
python craft_main.py --dataset dataset2 --epochs 20 --early_stop 5 --batch_size 200
```

BPR/矩阵分解轻量对照基线：

```bash
python scripts/run_bpr_baseline.py --dataset dataset1 --data_dir data --output_dir results --epochs 5 --factors 64
python scripts/run_bpr_baseline.py --dataset dataset2 --data_dir data --output_dir results --epochs 5 --factors 64
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

打包提交文件：

```bash
python scripts/make_result_zip.py --input_dir results --output result.zip
```

## 实验设计与可复现性

详细实验计划见 [docs/evaluation_plan.md](docs/evaluation_plan.md)。复现说明见 [docs/reproducibility.md](docs/reproducibility.md)。快速复现指南见 [docs/quickstart_reproduce.md](docs/quickstart_reproduce.md)。开题报告逐项对照见 [docs/opening_report_alignment.md](docs/opening_report_alignment.md)。

每次有效实验需要保存：

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
| 开题报告一致性 | docs/opening_report_alignment.md |
| 技术路线 | README 技术路线、docs/midterm_report.md |
| 数据说明 | docs/data.md |
| 实验设计 | docs/evaluation_plan.md |
| 评价指标代码 | src/metrics.py、scripts/evaluate_predictions.py |
| 运行说明 | README 环境配置、快速运行、推理与结果生成、docs/quickstart_reproduce.md |
| 阶段结果 | results/experiment_summary.md、docs/experiments.md |
| 可复现性 | docs/reproducibility.md |
| 官方 CRAFT 入口 | craft_main.py |

## 小组分工

| 成员 | 主要工作 |
| --- | --- |
| 李鑫霖 | 环境搭建、JittorGeometric/CRAFT baseline、模型训练与推理、历史频次和近期频次基线、时间感知重排序、模型融合实验、仓库维护和 README/Demo |
| 涂东岳 | 文献阅读、BPR/LightGCN/JODIE/TGAT/TGN 等研究现状整理、数据字段与时间顺序复核、候选集检查、指标表格、PPT 材料、可视化说明和参考文献整理 |
| 协作任务 | 结果复核、报告撰写、答辩演示、参考文献整理 |

## 参考资料

- 第六届“计图”人工智能算法挑战赛：https://cg.cs.tsinghua.edu.cn/jittor/news/2026-4-9-13-44-00-00-JittorComp6th/
- Jittor：https://github.com/Jittor/jittor
- JittorGeometric：https://github.com/AlgRUC/JittorGeometric
- CRAFT paper：https://arxiv.org/abs/2505.19408
