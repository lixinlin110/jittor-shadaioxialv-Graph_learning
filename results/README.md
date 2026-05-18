# Results

本目录用于保存训练日志、模型指标、预测文件和中期汇报表格。大型 checkpoint、原始比赛数据和临时日志不建议提交到 GitHub。

推荐结构：

```text
results/
├── experiment_summary.md
├── dataset1/
│   ├── run_metadata.json
│   ├── metrics.json
│   └── submission.csv
├── dataset2/
│   ├── run_metadata.json
│   ├── metrics.json
│   └── submission.csv
└── experiments/
    └── round_xxx/
        ├── config.json
        ├── metrics.json
        ├── result.zip
        └── notes.md
```

当前阶段主要实验结果见 [experiment_summary.md](experiment_summary.md)。

比赛正式提交文件结构：

```text
result.zip
├── dataset1_result.csv
└── dataset2_result.csv
```

每个 csv 每行对应一个测试样本，包含 100 个候选节点概率，使用英文逗号分隔，保留 8 位小数。
