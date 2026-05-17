# 算力申请说明

## 项目名称

基于图学习的动态推荐任务研究：面向时序图未来链接预测的时间感知候选节点重排序策略。

## 申请目的

本项目需要在第六届“计图”人工智能算法挑战赛图学习任务中复现官方 CRAFT baseline，并在其基础上完成时间窗口、时间衰减、近期邻居采样和候选节点重排序等消融实验。为保证中期汇报和后续正式提交能够按时完成，需要申请稳定 GPU 算力。

## 为什么需要 GPU

1. 任务数据是时序图交互数据，训练过程需要反复构造 source 的历史邻居序列并对候选 destination 集合打分。
2. 官方 CRAFT baseline 包含可学习节点表示、近期邻居采样、target-aware matching 等模块，多轮训练在 CPU 上耗时较长。
3. 后续实验需要同时比较 dataset1/dataset2、不同 epochs、early_stop、batch_size、num_neighbors、时间窗口长度和时间衰减系数。
4. 中期之后需要完成多组消融实验和多随机种子复现实验，单次运行结果不足以支撑可靠结论。
5. 需要保留 checkpoint、日志、预测文件和指标表格，便于答辩展示和结果复核。

## 计划使用方式

| 阶段 | 内容 | 预估需求 |
| --- | --- | --- |
| baseline 复现 | dataset1/dataset2 官方 CRAFT baseline 多轮复现 | 20-30 GPU 小时 |
| 参数搜索 | epochs、early_stop、batch_size、num_neighbors 调参 | 30-40 GPU 小时 |
| 方法实现 | 时间窗口、时间衰减、近期邻居模块调试 | 20-30 GPU 小时 |
| 消融实验 | 官方 baseline 与各增强模块对比 | 30-50 GPU 小时 |
| 结果复核 | 固定随机种子、多次运行、生成提交文件 | 10-20 GPU 小时 |

## 推荐配置

- GPU：NVIDIA RTX 4090 / A6000 / A100 或同等级显卡，显存建议 24GB 以上。
- CPU：8 核以上。
- 内存：64GB 以上。
- 存储：200GB 以上，用于数据、checkpoint、日志和结果文件。
- 软件：Python 3.10、Jittor、JittorGeometric、CUDA 环境。

## 预期产出

1. 可复现的官方 baseline 训练、推理和评测流程。
2. 时间感知候选节点重排序策略的实现代码。
3. 官方 baseline、时间窗口、时间衰减、近期邻居采样、综合策略的消融实验表格。
4. MRR、Recall@K、NDCG@K、Hit Rate 等指标结果。
5. 中期汇报 PPT 所需的实验记录、图表和案例。
6. 最终比赛提交文件和答辩演示材料。
