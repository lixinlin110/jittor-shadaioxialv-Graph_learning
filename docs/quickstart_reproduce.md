# 快速复现指南

本文档用于降低复现门槛，说明从获取数据、检查环境、运行 baseline、生成提交文件到记录实验结果的最短路径。

## 1. 获取代码

```bash
git clone https://github.com/lixinlin110/jittor-shadaioxialv-Graph_learning.git
cd jittor-shadaioxialv-Graph_learning
```

如果网络环境无法稳定使用 `git clone`，也可以在 GitHub 页面点击 `Code -> Download ZIP` 下载源码压缩包。

## 2. 配置环境

推荐使用 Python 3.10。

```bash
conda create -n jittor-graph python=3.10
conda activate jittor-graph
pip install -r requirements.txt
```

官方 CRAFT baseline 需要 JittorGeometric。建议按源码方式安装：

```bash
pip install git+https://github.com/Jittor/jittor.git
git clone https://github.com/AlgRUC/JittorGeometric.git
cd JittorGeometric
pip install .
```

## 3. 获取比赛数据

数据文件不放入 GitHub。复现者需要从第六届“计图”人工智能算法挑战赛赛道一页面下载数据包，或使用本地已有的 `data_A.zip`。

下载或解压后整理为：

```text
data/
├── dataset1/
│   ├── train.csv
│   └── test.csv
└── dataset2/
    ├── train.csv
    └── test.csv
```

字段要求：

```text
train.csv: src,dst,time
test.csv:  src,time,candidate_1,...,candidate_100
```

官方数据已经按时间顺序排列。复现实验时不要随机打乱数据。

## 4. 运行代码完整性检查

先确认核心脚本没有截断或语法错误：

```bash
python -m py_compile craft_main.py scripts/run_bpr_baseline.py scripts/evaluate_predictions.py scripts/make_result_zip.py
python scripts/evaluate_predictions.py --demo
```

如果 `craft_main.py` 在网页预览中显示被截断，以本地文件为准。当前本地完整文件约 390 行，末尾包含：

```python
if __name__ == "__main__":
    main()
```

## 5. 运行官方 CRAFT baseline

```bash
python craft_main.py --dataset dataset1 --data_dir data --output_dir results --epochs 70 --early_stop 12 --batch_size 200
python craft_main.py --dataset dataset2 --data_dir data --output_dir results --epochs 20 --early_stop 5 --batch_size 200
```

输出文件：

```text
results/dataset1/dataset1_result.csv
results/dataset2/dataset2_result.csv
```

## 6. 运行 BPR/矩阵分解轻量对照基线

该脚本用于补充开题报告中的传统推荐对照实验，不作为当前最高分提交方案。

```bash
python scripts/run_bpr_baseline.py --dataset dataset1 --data_dir data --output_dir results --epochs 5 --factors 64 --submission_copy
python scripts/run_bpr_baseline.py --dataset dataset2 --data_dir data --output_dir results --epochs 5 --factors 64 --submission_copy
```

输出文件：

```text
results/dataset1/dataset1_bpr_result.csv
results/dataset1/bpr_mf_model.npz
results/dataset1/bpr_mf_metadata.json
results/dataset2/dataset2_bpr_result.csv
results/dataset2/bpr_mf_model.npz
results/dataset2/bpr_mf_metadata.json
```

## 7. 生成提交文件

如果 `results/dataset1/dataset1_result.csv` 和 `results/dataset2/dataset2_result.csv` 已经存在：

```bash
python scripts/make_result_zip.py --input_dir results --output result.zip
```

最终结构必须是：

```text
result.zip
├── dataset1_result.csv
└── dataset2_result.csv
```

不要上传外层文件夹，只上传 `result.zip`。

## 8. 实验记录规范

每次实验建议保存：

```text
实验编号
数据集
模型类型
随机种子
训练轮数
batch_size
num_neighbors 或 factors
学习率
本地验证指标
A 榜反馈分数
结果文件路径
是否保留为最优
```

阶段性结果可以写入：

```text
results/experiment_summary.md
docs/experiments.md
```

## 9. 当前推荐复现顺序

1. 先运行 `python scripts/evaluate_predictions.py --demo` 确认基础环境。
2. 再运行 `scripts/run_bpr_baseline.py`，获得传统推荐对照结果。
3. 最后运行 `craft_main.py`，获得官方 CRAFT/JittorGeometric baseline 结果。
4. 将 CRAFT、BPR/频次基线和融合结果写入实验表格，用于最终报告和 PPT。
