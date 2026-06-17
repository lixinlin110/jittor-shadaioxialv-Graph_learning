import argparse
import json
import os
import os.path as osp
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgbm_candidate_ranker import apply_residual, pack_submission, read_scores, write_scores
from make_round65_dataset2_rich_ranker import (
    N_CANDIDATES,
    RICH_FEATURE_NAMES,
    RichTemporalStats,
    TestCandidateContext,
    make_rich_features,
    parse_float_list,
    score_test,
    signed_tag,
    train_ranker,
)


def sample_queries(df, max_queries, seed, recency_power):
    if not max_queries or len(df) <= max_queries:
        return df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    t = df["time"].values.astype(np.float64)
    denom = max(1.0, float(t.max() - t.min()))
    weights = 0.15 + np.exp(recency_power * ((t - t.min()) / denom))
    weights = weights / weights.sum()
    idx = rng.choice(len(df), size=max_queries, replace=False, p=weights)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def build_strict_query_frame(train_df, context, seed, max_rows_per_item, desc):
    rng = np.random.RandomState(seed)
    rows = []
    src_to_train = {int(src): group for src, group in train_df.groupby("src", sort=False)}

    for src, test_rows in tqdm(context.by_src.items(), ncols=120, desc=desc):
        src = int(src)
        group = src_to_train.get(src)
        if group is None or len(group) == 0:
            continue

        item_to_rows = defaultdict(list)
        for row_idx in test_rows:
            for dst_raw in context.candidates[int(row_idx)]:
                dst = int(dst_raw)
                bucket = item_to_rows[dst]
                if len(bucket) < max_rows_per_item:
                    bucket.append(int(row_idx))

        if not item_to_rows:
            continue

        for src_raw, dst_raw, time_raw in group[["src", "dst", "time"]].itertuples(index=False):
            dst = int(dst_raw)
            candidate_rows = item_to_rows.get(dst)
            if not candidate_rows:
                continue
            row_idx = candidate_rows[int(rng.randint(len(candidate_rows)))]
            rows.append((int(src_raw), dst, int(time_raw), row_idx))

    return pd.DataFrame(rows, columns=["src", "dst", "time", "candidate_row"]).sort_values("time").reset_index(drop=True)


def build_matrix(strict_df, stats, context, max_queries, seed, desc, recency_power):
    sampled = sample_queries(strict_df, max_queries, seed, recency_power)
    n_queries = len(sampled)
    x = np.empty((n_queries * N_CANDIDATES, len(RICH_FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(n_queries * N_CANDIDATES, dtype=np.int8)
    group = np.full(n_queries, N_CANDIDATES, dtype=np.int32)

    offset = 0
    for row in tqdm(sampled.itertuples(index=False), total=n_queries, ncols=120, desc=desc):
        src = int(row.src)
        dst = int(row.dst)
        cur_time = int(row.time)
        candidates = context.candidates[int(row.candidate_row)]
        labels = (candidates == dst).astype(np.int8)
        if labels.sum() == 0:
            raise RuntimeError("strict query lost its positive candidate")
        x[offset : offset + N_CANDIDATES] = make_rich_features(stats, context, src, cur_time, candidates)
        y[offset : offset + N_CANDIDATES] = labels
        offset += N_CANDIDATES

    return x, y, group


def evaluate_strict_mrr(model, strict_df, stats, context, max_queries, seed):
    sampled = sample_queries(strict_df, max_queries, seed, recency_power=0.25)
    rr = []
    hit_at_10 = []
    for row in tqdm(sampled.itertuples(index=False), total=len(sampled), ncols=120, desc="Validate strict ranker"):
        src = int(row.src)
        dst = int(row.dst)
        cur_time = int(row.time)
        candidates = context.candidates[int(row.candidate_row)]
        labels = (candidates == dst).astype(np.int8)
        pred = model.predict(make_rich_features(stats, context, src, cur_time, candidates))
        positive = np.where(labels > 0)[0]
        best_pos = pred[positive].max()
        rank = 1 + int(np.sum(pred > best_pos))
        rr.append(1.0 / rank)
        hit_at_10.append(1.0 if rank <= 10 else 0.0)
    return {
        "strict_mrr": float(np.mean(rr)) if rr else 0.0,
        "strict_hitrate@10": float(np.mean(hit_at_10)) if hit_at_10 else 0.0,
        "strict_queries": int(len(sampled)),
    }


def run_dataset2(args):
    data_root = osp.join(args.data_dir, "dataset2")
    train_df = pd.read_csv(osp.join(data_root, "train.csv")).sort_values("time").reset_index(drop=True)
    test_df = pd.read_csv(osp.join(data_root, "test.csv")).reset_index(drop=True)
    context = TestCandidateContext(test_df, keep_top_per_src=args.keep_top_per_src)

    split0 = train_df[train_df["split"] == 0].sort_values("time").reset_index(drop=True)
    split1 = train_df[train_df["split"] == 1].sort_values("time").reset_index(drop=True)
    cut = int(len(split0) * args.split0_cut)
    hist0 = split0.iloc[:cut].reset_index(drop=True)
    train0_tail = split0.iloc[cut:].reset_index(drop=True)

    strict_tail = build_strict_query_frame(
        train0_tail,
        context,
        args.seed,
        args.max_rows_per_item,
        "Build strict split0-tail queries",
    )
    strict_split1 = build_strict_query_frame(
        split1,
        context,
        args.seed + 11,
        args.max_rows_per_item,
        "Build strict split1 queries",
    )
    if len(strict_tail) == 0 or len(strict_split1) == 0:
        raise RuntimeError("strict query builder produced no training data")

    stats_hist0 = RichTemporalStats(hist0, args.recent_limit, args.transition_recent)
    x_train, y_train, group_train = build_matrix(
        strict_tail,
        stats_hist0,
        context,
        args.train_queries_tail,
        args.seed,
        "Build strict split0-tail matrix",
        args.recency_power,
    )
    val_model = train_ranker(x_train, y_train, group_train, args, args.seed)

    stats_split0 = RichTemporalStats(split0, args.recent_limit, args.transition_recent)
    val_metrics = evaluate_strict_mrr(
        val_model,
        strict_split1,
        stats_split0,
        context,
        args.val_queries,
        args.seed + 77,
    )

    x_extra, y_extra, group_extra = build_matrix(
        strict_split1,
        stats_split0,
        context,
        args.train_queries_split1,
        args.seed + 17,
        "Build strict split1 matrix",
        args.recency_power,
    )
    x_final = np.vstack([x_train, x_extra])
    y_final = np.concatenate([y_train, y_extra])
    group_final = np.concatenate([group_train, group_extra])
    del x_train, y_train, group_train, x_extra, y_extra, group_extra

    model = train_ranker(x_final, y_final, group_final, args, args.seed + 99)
    full_stats = RichTemporalStats(train_df, args.recent_limit, args.transition_recent)
    raw = score_test(model, test_df, full_stats, context, args.batch_rows)

    run_root = osp.join(args.runs_dir, args.run_name)
    out_dir = osp.join(run_root, "dataset2")
    os.makedirs(out_dir, exist_ok=True)
    raw_file = osp.join(out_dir, "dataset2_strict_raw.npy")
    sigmoid_file = osp.join(out_dir, "dataset2_strict_sigmoid.csv")
    np.save(raw_file, raw)
    write_scores(1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0))), sigmoid_file)

    importance = [
        {"feature": name, "gain": float(gain)}
        for name, gain in sorted(
            zip(RICH_FEATURE_NAMES, model.booster_.feature_importance(importance_type="gain")),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return {
        "rows_train": int(len(train_df)),
        "rows_split0": int(len(split0)),
        "rows_split1": int(len(split1)),
        "rows_test": int(len(test_df)),
        "strict_tail_queries": int(len(strict_tail)),
        "strict_split1_queries": int(len(strict_split1)),
        "train_queries_tail": int(min(args.train_queries_tail, len(strict_tail))),
        "train_queries_split1": int(min(args.train_queries_split1, len(strict_split1))),
        "val_metrics": val_metrics,
        "raw_file": raw_file,
        "sigmoid_file": sigmoid_file,
        "feature_importance": importance,
    }


def copy_dataset1(source_dir, run_root):
    src = osp.join(source_dir, "dataset1", "dataset1_result.csv")
    dst = osp.join(run_root, "dataset1", "dataset1_result.csv")
    if not osp.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(osp.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def write_variant(name, fixed_d1_dir, base_d2, raw, alpha, margin):
    run_root = osp.join("./experiments", name)
    copy_dataset1(fixed_d1_dir, run_root)
    d2_scores, modified = apply_residual(base_d2, raw, alpha, margin)
    write_scores(d2_scores, osp.join(run_root, "dataset2", "dataset2_result.csv"))
    zip_path = pack_submission(run_root)
    record = {
        "name": name,
        "zip": zip_path,
        "dataset2_alpha": float(alpha),
        "dataset2_margin": float(margin),
        "dataset2_modified_rows": int(modified),
        "dataset2_modified_ratio": float(modified / len(base_d2)),
    }
    print("Variant:", record)
    return record


def alias(source_name, alias_name):
    src = osp.join("./experiments", source_name, "result.zip")
    dst_dir = osp.join("./experiments", alias_name)
    dst = osp.join(dst_dir, "result.zip")
    if not osp.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--runs_dir", default="./experiments")
    parser.add_argument("--run_name", default="round68_d2_strict_candidate")
    parser.add_argument("--fixed_d1_dir", default="./experiments/current_best_12971683198949764")
    parser.add_argument("--base_d2_dir", default="./experiments/current_best_12971683198949764")
    parser.add_argument("--recent_limit", type=int, default=960)
    parser.add_argument("--transition_recent", type=int, default=12)
    parser.add_argument("--keep_top_per_src", type=int, default=512)
    parser.add_argument("--max_rows_per_item", type=int, default=4)
    parser.add_argument("--split0_cut", type=float, default=0.78)
    parser.add_argument("--train_queries_tail", type=int, default=50000)
    parser.add_argument("--train_queries_split1", type=int, default=38000)
    parser.add_argument("--val_queries", type=int, default=30000)
    parser.add_argument("--recency_power", type=float, default=1.0)
    parser.add_argument("--batch_rows", type=int, default=5000)
    parser.add_argument("--n_estimators", type=int, default=520)
    parser.add_argument("--learning_rate", type=float, default=0.028)
    parser.add_argument("--num_leaves", type=int, default=95)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_child_samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample_bytree", type=float, default=0.92)
    parser.add_argument("--reg_alpha", type=float, default=0.04)
    parser.add_argument("--reg_lambda", type=float, default=0.50)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--alphas", default="0.006,0.010,0.014,0.018,-0.006")
    parser.add_argument("--margins", default="0.14,0.18,0.22")
    args = parser.parse_args()

    record = run_dataset2(args)
    raw = np.load(record["raw_file"])
    base_d2 = read_scores(osp.join(args.base_d2_dir, "dataset2", "dataset2_result.csv"))

    variants = []
    for alpha in parse_float_list(args.alphas):
        for margin in parse_float_list(args.margins):
            name = f"{args.run_name}_{signed_tag(alpha)}_m{int(round(margin * 1000)):03d}"
            variants.append(write_variant(name, args.fixed_d1_dir, base_d2, raw, alpha, margin))

    preferred = [
        f"{args.run_name}_p010_m180",
        f"{args.run_name}_p014_m180",
        f"{args.run_name}_p006_m180",
        f"{args.run_name}_p018_m180",
        f"{args.run_name}_p010_m140",
        f"{args.run_name}_n006_m180",
    ]
    recommended = []
    for idx, name in enumerate(preferred, start=1):
        src_zip = osp.join(args.runs_dir, name, "result.zip")
        if not osp.exists(src_zip):
            continue
        alias_name = f"round68_recommended_{idx}_{name.replace(args.run_name + '_', '')}"
        recommended.append(alias(name, alias_name))

    summary = {
        "method": "dataset2_strict_candidate_lgbm_ranker",
        "description": "Dataset2-only rich LambdaRank model trained only on historical positives that already appear in same-src public test candidate rows. Dataset1 is fixed at current online best.",
        "args": vars(args),
        "record": record,
        "variants": variants,
        "recommended_order": recommended,
    }
    summary_path = osp.join(args.runs_dir, args.run_name, "round68_dataset2_strict_candidate_summary.json")
    os.makedirs(osp.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Summary:", summary_path)
    print("Recommended:")
    for path in recommended:
        print(path)


if __name__ == "__main__":
    main()
