# 数据说明

## 任务输入

动态图推荐任务的基础输入为历史交互三元组：

```text
source, destination, time
```

- `source`：源节点，可表示用户、论文、网页或其他交互主体。
- `destination`：目标节点，可表示商品、论文、网页或候选交互对象。
- `time`：交互时间戳，用于保证训练、验证和测试严格按时间顺序切分。

脚本兼容以下字段别名：

| 标准字段 | 可兼容别名 |
| --- | --- |
| source | src, u, user, user_id, from, head |
| destination | dst, v, item, item_id, to, tail, target |
| time | timestamp, ts, t, datetime |

## 推荐目录

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

## 候选集格式

验证集和测试集需要包含候选 destination 集合。支持两种格式。

### 单列 candidates

```csv
source,time,destination,candidates
u1,100,v3,"v3 v9 v10 v12"
```

### 多列 candidate_*

```csv
source,time,destination,candidate_0,candidate_1,candidate_2
u1,100,v3,v9,v3,v12
```

测试集如果没有真实 `destination` 标签，也可以只提供：

```csv
source,time,candidates
u1,120,"v1 v2 v3 v4"
```

## 时间泄漏检查

1. 训练集必须早于验证集和测试集。
2. 验证/测试阶段只能使用预测时间之前的历史交互。
3. 构造 source 历史邻居时，应过滤 `event_time > predict_time` 的交互。
4. 如果使用全局热门度，需要确认统计范围不包含验证/测试未来交互。

## 结果文件

默认推理结果保存在：

```text
results/{dataset}/submission.csv
```

其中 `ranked_destinations` 是按得分从高到低排序的候选节点列表。
