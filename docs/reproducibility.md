# 可复现性说明

本项目的目标之一是让每一次有效实验都能被复查、复跑和解释。本文档说明从环境、数据、训练、推理、评估到提交文件的完整复现流程。

## 1. 环境复现

推荐环境：

```text
Python 3.10
Jittor >= 1.3.0
JittorGeometric source install
numpy, pandas, scikit-learn, tqdm
```

安装命令：

```bash
conda create -n jittor-graph python=3.10
conda activate jittor-graph
pip install -r requirements.txt
```

JittorGeometric：

```bash
pip install git+https://github.com/Jittor/jittor.git
git clone https://github.com/AlgRUC/JittorGeometric.git
cd JittorGeometric
pip install .
```

## 2. 数据复现

比赛数据不提交到 GitHub。复现实验时需要将数据放入：

```text
data/dataset1/train.csv
data/dataset1/test.csv
data/dataset2/train.csv
data/dataset2/test.csv
```

如果做本地验证，可以从训练集尾部按时间切出 `valid.csv`。切分时必须保持时间顺序，不能随机打乱。

## 3. 训练复现

轻量时间感知 reranker：

```bash
python main.py --dataset dataset1 --data_dir data --output_dir results
python main.py --dataset dataset2 --data_dir data --output_dir results
```

官方 CRAFT/JittorGeometric 训练入口：

```bash
python craft_main.py --dataset dataset1 --epochs 70 --early_stop 12 --batch_size 200
python craft_main.py --dataset dataset2 --epochs 20 --early_stop 5 --batch_size 200
```

每次训练建议固定并记录：

```text
random seed
epochs
early_stop
batch_size
num_neighbors
hidden_size
n_layers
n_heads
learning_rate
```

## 4. 推理复现

轻量 reranker 推理示例：

```bash
python -m src.infer \
  --dataset dataset1 \
  --model_path results/dataset1/time_aware_reranker.json \
  --candidate_file data/dataset1/test.csv \
  --output_file results/dataset1/submission.csv
```

CRAFT 入口会直接生成：

```text
data/{dataset}/{dataset}_result.csv
```

比赛提交文件需要整理为：

```text
result.zip
├── dataset1_result.csv
└── dataset2_result.csv
```

打包命令：

```bash
python scripts/make_result_zip.py --input_dir results --output result.zip
```

## 5. 指标复现

内置 demo：

```bash
python scripts/evaluate_predictions.py --demo
```

验证集预测文件：

```bash
python scripts/evaluate_predictions.py \
  --input results/dataset1/validation_predictions.csv \
  --target_col destination \
  --ranked_col ranked_destinations
```

## 6. 实验归档规范

每次有效实验建议单独建目录，例如：

```text
results/experiments/round19_adapt_best215_seed7777_b02_e06_t005/
├── config.json
├── run_metadata.json
├── metrics.json
├── dataset1_result.csv
├── dataset2_result.csv
├── result.zip
└── notes.md
```

`notes.md` 至少记录：

```text
实验目的
相对上一轮的变化
本地验证指标
A 榜反馈分数
是否保留为当前最优
下一步计划
```

## 7. 防止时间泄漏

1. 训练阶段只使用训练时间段内的交互。
2. 验证和测试阶段构造历史邻居时，只允许使用 `event_time <= predict_time` 的交互。
3. 全局热门度、历史频次、近期频次只能从允许的历史范围统计。
4. 不能用测试集真实答案参与训练或调参。

## 8. 当前最优记录

当前阶段最优 A 榜反馈：

```text
实验：round19_adapt_best215_seed7777_b02_e06_t005
方法：best21.5% 基线结果 + seed7777 CRAFT 结果 + 置信度自适应融合
A 榜反馈：1.1273681134893112
```

该记录会继续与后续本地验证、A 榜提交和最终实验报告同步维护。
