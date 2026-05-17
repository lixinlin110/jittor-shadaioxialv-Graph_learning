# 基于图学习的动态推荐任务

本仓库用于第六届“计图”人工智能算法挑战赛图学习方向的中期汇报、实验归档和算力申请。项目围绕时序图未来链接预测展开：给定历史交互三元组 `(source, destination, time)`，在测试阶段对每个 `source`、`time` 和候选 `destination` 集合进行打分与重排序，并以 MRR 作为核心指标。

官方赛题页面说明，本届挑战赛的正式赛道包含“基于图学习的动态推荐任务”，要求使用 Jittor/JittorGeometric 完成；官方提供基于 JittorGeometric 实现的 CRAFT 方法作为 baseline。我们的中期目标是在官方 CRAFT baseline 的基础上，补充时间窗口、时间衰减和近期邻居采样模块，形成可复现的候选节点重排序实验流程。

## 当前进度

- 已完成开题报告，明确研究题目为“基于图学习的动态推荐任务研究：面向时序图未来链接预测的时间感知候选节点重排序策略”。
- 已跑通官方 CRAFT baseline，并保留多轮参数记录；当前记录中 A 榜最好成绩约为 `1.0878`。
- 已整理仓库结构、运行脚本、配置模板、实验记录、数据说明、中期汇报材料和算力申请说明。
- 下一步工作是把时间感知重排序策略接到官方 baseline 的训练/推理流程中，并完成消融实验。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── main.py
├── src/
│   ├── data.py
│   ├── infer.py
│   ├── metrics.py
│   ├── models.py
│   └── train.py
├── scripts/
│   ├── run_baseline.sh
│   └── run_baseline.ps1
├── configs/
│   ├── baseline_dataset1.json
│   ├── baseline_dataset2.json
│   └── time_aware_rerank.json
├── docs/
│   ├── compute_request.md
│   ├── data.md
│   ├── experiments.md
│   └── midterm_report.md
├── results/
│   └── README.md
└── data/
    └── .gitkeep
```

## 环境配置

```bash
conda create -n jittor-graph python=3.10
conda activate jittor-graph
pip install -r requirements.txt
```

JittorGeometric 建议按官方仓库安装：

```bash
pip install git+https://github.com/Jittor/jittor.git
git clone https://github.com/AlgRUC/JittorGeometric.git
cd JittorGeometric
pip install .
```

## 数据放置

推荐将比赛数据放在以下目录：

```text
data/
├── dataset1/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── dataset2/
    ├── train.csv
    ├── valid.csv
    └── test.csv
```

训练文件至少包含三列：`source`、`destination`、`time`。脚本也兼容常见别名，如 `src/dst/ts`、`u/i/t`、`user_id/item_id/timestamp`。候选集文件可以使用 `candidates` 字符串列，或使用 `candidate_0, candidate_1, ...` 多列形式。

## 运行方式

快速检查项目流程：

```bash
python main.py --demo --dataset demo
```

复现实验记录中的官方 baseline 参数接口：

```bash
python main.py --dataset dataset1 --epochs 70 --early_stop 12 --batch_size 200
python main.py --dataset dataset2 --epochs 20 --early_stop 5 --batch_size 200
```

也可以直接使用脚本：

```bash
bash scripts/run_baseline.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
```

推理示例：

```bash
python -m src.infer \
  --dataset dataset1 \
  --model_path results/dataset1/time_aware_reranker.json \
  --candidate_file data/dataset1/test.csv \
  --output_file results/dataset1/submission.csv
```

## 方法路线

1. 官方 CRAFT baseline：以可学习节点 ID 表示和 source 近期交互序列为基础，通过 target-aware matching 对候选 destination 打分。
2. 时间窗口动态图快照：按交互时间划分窗口，比较不同历史窗口对排序结果的影响。
3. 时间衰减边权重：预测时间越近的历史交互获得越高权重，降低过早历史交互的干扰。
4. 近期邻居采样：优先利用 source 的近邻交互序列，控制 `num_neighbors` 并提升推理效率。
5. 候选节点重排序：在 baseline 得分上融合历史频次、近期频次、重复连接、新连接和时间衰减信号。

## 中期汇报重点

- 研究问题：动态图未来链接预测与候选节点重排序。
- 已有基础：官方 CRAFT baseline 已跑通，调参记录完整。
- 创新方向：时间感知候选节点重排序策略。
- 评价指标：MRR 为核心，辅助使用 Recall@K、NDCG@K、Hit Rate。
- 算力需求：需要 GPU 支持多数据集、多参数、多消融实验重复训练。

## 参考链接

- 第六届“计图”人工智能算法挑战赛：https://cg.cs.tsinghua.edu.cn/jittor/news/2026-4-9-13-44-00-00-JittorComp6th/
- Jittor：https://github.com/Jittor/jittor
- JittorGeometric：https://github.com/AlgRUC/JittorGeometric
- CRAFT 论文：https://arxiv.org/abs/2505.19408
