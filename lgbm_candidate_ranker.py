import argparse
import json
import os
import os.path as osp
import warnings
import zipfile
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


DATASETS = ("dataset1", "dataset2")
N_CANDIDATES = 100
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def row_zscore(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.mean()) / std


def read_scores(path):
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def write_scores(scores, output_file):
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for row in scores:
            f.write(",".join(f"{float(p):.8f}" for p in row) + "\n")


def pack_submission(run_root):
    result_zip = osp.join(run_root, "result.zip")
    files = {
        "dataset1.csv": osp.join(run_root, "dataset1", "dataset1_result.csv"),
        "dataset2.csv": osp.join(run_root, "dataset2", "dataset2_result.csv"),
    }
    missing = [path for path in files.values() if not osp.exists(path)]
    if missing:
        raise FileNotFoundError("Missing result files: " + ", ".join(missing))
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in files.items():
            zf.write(path, arcname)
    return result_zip


class TemporalStats:
    def __init__(self, df, recent_limit):
        self.recent_limit = recent_limit
        self.pair_count = defaultdict(int)
        self.pair_last = {}
        self.dst_count = defaultdict(int)
        self.dst_last = {}
        self.src_count = defaultdict(int)
        self.src_recent_seq = defaultdict(list)
        self.src_recent = {}
        self.num_edges = max(1, len(df))
        self.time_min = int(df["time"].min())
        self.time_max = int(df["time"].max())
        self.time_span = max(1, self.time_max - self.time_min)
        self._build(df)

    def _build(self, df):
        src_values = df["src"].values
        dst_values = df["dst"].values
        time_values = df["time"].values
        for s_raw, d_raw, t_raw in tqdm(
            zip(src_values, dst_values, time_values),
            total=len(df),
            ncols=120,
            desc="Build temporal stats",
        ):
            s = int(s_raw)
            d = int(d_raw)
            t = int(t_raw)
            key = (s, d)
            self.pair_count[key] += 1
            self.pair_last[key] = t
            self.dst_count[d] += 1
            self.dst_last[d] = t
            self.src_count[s] += 1
            self.src_recent_seq[s].append(d)

        for s, seq in self.src_recent_seq.items():
            weights = {}
            rank = 1
            for d in reversed(seq[-self.recent_limit :]):
                if d not in weights:
                    weights[d] = 1.0 / rank
                    rank += 1
            self.src_recent[s] = weights


class TestCandidateSampler:
    def __init__(self, test_df, seed):
        self.rng = np.random.RandomState(seed)
        self.src = test_df["src"].values.astype(np.int64)
        self.candidates = test_df.iloc[:, 2:].values.astype(np.int64)
        self.by_src = defaultdict(list)
        for i, s in enumerate(self.src):
            self.by_src[int(s)].append(i)
        self.by_src = {s: np.asarray(v, dtype=np.int64) for s, v in self.by_src.items()}

        flat = self.candidates.ravel()
        values, counts = np.unique(flat, return_counts=True)
        self.pool_count = {int(v): int(c) for v, c in zip(values, counts)}
        self.pool_size = max(1, len(flat))

    def sample_row(self, src):
        rows = self.by_src.get(int(src))
        if rows is not None and len(rows) > 0 and self.rng.rand() < 0.80:
            return int(rows[self.rng.randint(len(rows))])
        return int(self.rng.randint(len(self.candidates)))

    def sample_candidates_with_positive(self, src, dst):
        row_idx = self.sample_row(src)
        candidates = self.candidates[row_idx].copy()
        dst = int(dst)
        labels = (candidates == dst).astype(np.int8)
        if labels.sum() == 0:
            replace_idx = int(self.rng.randint(len(candidates)))
            candidates[replace_idx] = dst
            labels[replace_idx] = 1
        return candidates, labels


FEATURE_NAMES = [
    "pair_count_log",
    "pair_seen",
    "pair_rate",
    "pair_rec_short",
    "pair_rec_mid",
    "pair_rec_long",
    "src_recent_weight",
    "dst_count_log",
    "dst_seen",
    "dst_share",
    "dst_rec_short",
    "dst_rec_mid",
    "dst_rec_long",
    "candidate_pool_count_log",
    "candidate_pool_seen",
    "duplicate_in_row",
    "pair_count_row_z",
    "pair_rec_row_z",
    "src_recent_row_z",
    "dst_count_row_z",
    "dst_rec_row_z",
    "candidate_pool_row_z",
]


def make_features(stats, sampler, src, cur_time, candidates):
    src = int(src)
    cur_time = int(cur_time)
    cand = np.asarray(candidates, dtype=np.int64)
    n = len(cand)

    pair_count_log = np.zeros(n, dtype=np.float64)
    pair_seen = np.zeros(n, dtype=np.float64)
    pair_rate = np.zeros(n, dtype=np.float64)
    pair_rec_short = np.zeros(n, dtype=np.float64)
    pair_rec_mid = np.zeros(n, dtype=np.float64)
    pair_rec_long = np.zeros(n, dtype=np.float64)
    src_recent_weight = np.zeros(n, dtype=np.float64)
    dst_count_log = np.zeros(n, dtype=np.float64)
    dst_seen = np.zeros(n, dtype=np.float64)
    dst_share = np.zeros(n, dtype=np.float64)
    dst_rec_short = np.zeros(n, dtype=np.float64)
    dst_rec_mid = np.zeros(n, dtype=np.float64)
    dst_rec_long = np.zeros(n, dtype=np.float64)
    candidate_pool_count_log = np.zeros(n, dtype=np.float64)
    candidate_pool_seen = np.zeros(n, dtype=np.float64)

    tau_short = max(1.0, 0.015 * stats.time_span)
    tau_mid = max(1.0, 0.060 * stats.time_span)
    tau_long = max(1.0, 0.200 * stats.time_span)
    src_total = max(1, stats.src_count.get(src, 0))
    recent_map = stats.src_recent.get(src, {})

    row_counts = defaultdict(int)
    for d_raw in cand:
        row_counts[int(d_raw)] += 1
    duplicate_in_row = np.asarray([row_counts[int(d)] - 1 for d in cand], dtype=np.float64)

    for j, d_raw in enumerate(cand):
        d = int(d_raw)
        key = (src, d)
        pair_count = stats.pair_count.get(key, 0)
        if pair_count:
            pair_count_log[j] = np.log1p(pair_count)
            pair_seen[j] = 1.0
            pair_rate[j] = pair_count / src_total
            delta = max(0, cur_time - stats.pair_last[key])
            pair_rec_short[j] = np.exp(-delta / tau_short)
            pair_rec_mid[j] = np.exp(-delta / tau_mid)
            pair_rec_long[j] = np.exp(-delta / tau_long)

        src_recent_weight[j] = recent_map.get(d, 0.0)

        dst_count = stats.dst_count.get(d, 0)
        if dst_count:
            dst_count_log[j] = np.log1p(dst_count)
            dst_seen[j] = 1.0
            dst_share[j] = dst_count / stats.num_edges
            delta = max(0, cur_time - stats.dst_last[d])
            dst_rec_short[j] = np.exp(-delta / tau_short)
            dst_rec_mid[j] = np.exp(-delta / tau_mid)
            dst_rec_long[j] = np.exp(-delta / tau_long)

        pool_count = sampler.pool_count.get(d, 0)
        if pool_count:
            candidate_pool_count_log[j] = np.log1p(pool_count)
            candidate_pool_seen[j] = 1.0

    return np.column_stack(
        [
            pair_count_log,
            pair_seen,
            pair_rate,
            pair_rec_short,
            pair_rec_mid,
            pair_rec_long,
            src_recent_weight,
            dst_count_log,
            dst_seen,
            dst_share,
            dst_rec_short,
            dst_rec_mid,
            dst_rec_long,
            candidate_pool_count_log,
            candidate_pool_seen,
            duplicate_in_row,
            row_zscore(pair_count_log),
            row_zscore(pair_rec_mid),
            row_zscore(src_recent_weight),
            row_zscore(dst_count_log),
            row_zscore(dst_rec_mid),
            row_zscore(candidate_pool_count_log),
        ]
    ).astype(np.float32)


def sample_query_rows(df, max_queries, seed):
    if max_queries and len(df) > max_queries:
        return df.sample(n=max_queries, random_state=seed).sort_index()
    return df


def build_matrix(query_df, stats, sampler, max_queries, seed, desc):
    sampled = sample_query_rows(query_df, max_queries, seed)
    n_queries = len(sampled)
    x = np.empty((n_queries * N_CANDIDATES, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(n_queries * N_CANDIDATES, dtype=np.int8)
    group = np.full(n_queries, N_CANDIDATES, dtype=np.int32)

    offset = 0
    for row in tqdm(sampled.itertuples(index=False), total=n_queries, ncols=120, desc=desc):
        src = int(getattr(row, "src"))
        dst = int(getattr(row, "dst"))
        cur_time = int(getattr(row, "time"))
        candidates, labels = sampler.sample_candidates_with_positive(src, dst)
        x[offset : offset + N_CANDIDATES] = make_features(stats, sampler, src, cur_time, candidates)
        y[offset : offset + N_CANDIDATES] = labels
        offset += N_CANDIDATES

    return x, y, group


def train_lgbm_ranker(x_train, y_train, group_train, seed, args):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=seed,
        n_jobs=args.n_jobs,
        importance_type="gain",
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        group=group_train,
        feature_name=FEATURE_NAMES,
    )
    return model


def evaluate_mrr(model, val_df, stats, sampler, max_queries, seed):
    sampled = sample_query_rows(val_df, max_queries, seed)
    rr = []
    for row in tqdm(sampled.itertuples(index=False), total=len(sampled), ncols=120, desc="Validate LGBM ranker"):
        src = int(getattr(row, "src"))
        dst = int(getattr(row, "dst"))
        cur_time = int(getattr(row, "time"))
        candidates, labels = sampler.sample_candidates_with_positive(src, dst)
        feat = make_features(stats, sampler, src, cur_time, candidates)
        pred = model.predict(feat)
        positive = np.where(labels > 0)[0]
        if len(positive) == 0:
            continue
        best_pos = pred[positive].max()
        rank = 1 + int(np.sum(pred > best_pos))
        rr.append(1.0 / rank)
    return float(np.mean(rr)) if rr else 0.0


def score_test(model, test_df, stats, sampler, batch_rows):
    candidates = test_df.iloc[:, 2:].values.astype(np.int64)
    srcs = test_df["src"].values.astype(np.int64)
    times = test_df["time"].values.astype(np.int64)
    out = np.empty(candidates.shape, dtype=np.float64)

    for start in tqdm(range(0, len(test_df), batch_rows), ncols=120, desc="Score test with LGBM"):
        end = min(start + batch_rows, len(test_df))
        feats = []
        for i in range(start, end):
            feats.append(make_features(stats, sampler, srcs[i], times[i], candidates[i]))
        x = np.vstack(feats)
        out[start:end] = model.predict(x).reshape(end - start, N_CANDIDATES)
    return out


def apply_residual(base_scores, ranker_raw, alpha, margin_threshold):
    base_logits = logit(base_scores)
    final_logits = base_logits.copy()
    modified = 0
    for i in range(base_logits.shape[0]):
        order = np.argsort(-base_logits[i])
        margin = base_logits[i, order[0]] - base_logits[i, order[1]]
        if margin > margin_threshold:
            continue
        final_logits[i] = base_logits[i] + alpha * row_zscore(ranker_raw[i])
        modified += 1
    return sigmoid(final_logits), modified


def parse_variant(text):
    parts = text.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("Variant must be name:d1_alpha:d2_alpha:d1_margin:d2_margin")
    return parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def run_dataset(dataset, args):
    data_root = osp.join(args.data_dir, dataset)
    train_df = pd.read_csv(osp.join(data_root, "train.csv")).sort_values("time").reset_index(drop=True)
    test_df = pd.read_csv(osp.join(data_root, "test.csv"))
    sampler = TestCandidateSampler(test_df, args.seed + (1 if dataset == "dataset1" else 2))

    n = len(train_df)
    cut70 = int(n * 0.70)
    cut85 = int(n * 0.85)
    hist70 = train_df.iloc[:cut70]
    train70_85 = train_df.iloc[cut70:cut85]
    hist85 = train_df.iloc[:cut85]
    train85_100 = train_df.iloc[cut85:]

    train_queries = args.train_queries_d1 if dataset == "dataset1" else args.train_queries_d2
    val_queries = args.val_queries_d1 if dataset == "dataset1" else args.val_queries_d2

    stats70 = TemporalStats(hist70, args.recent_limit)
    x_train, y_train, group_train = build_matrix(
        train70_85, stats70, sampler, train_queries, args.seed, f"Build {dataset} train fold"
    )
    val_model = train_lgbm_ranker(x_train, y_train, group_train, args.seed, args)

    stats85 = TemporalStats(hist85, args.recent_limit)
    val_mrr = evaluate_mrr(val_model, train85_100, stats85, sampler, val_queries, args.seed + 77)

    x_extra, y_extra, group_extra = build_matrix(
        train85_100, stats85, sampler, train_queries // 2, args.seed + 17, f"Build {dataset} recent fold"
    )
    x_final = np.vstack([x_train, x_extra])
    y_final = np.concatenate([y_train, y_extra])
    group_final = np.concatenate([group_train, group_extra])
    del x_train, y_train, group_train, x_extra, y_extra, group_extra

    model = train_lgbm_ranker(x_final, y_final, group_final, args.seed + 99, args)
    full_stats = TemporalStats(train_df, args.recent_limit)
    ranker_raw = score_test(model, test_df, full_stats, sampler, args.batch_rows)

    ranker_dir = osp.join(args.runs_dir, args.run_name, dataset)
    os.makedirs(ranker_dir, exist_ok=True)
    np.save(osp.join(ranker_dir, f"{dataset}_lgbm_raw.npy"), ranker_raw)
    write_scores(sigmoid(ranker_raw), osp.join(ranker_dir, f"{dataset}_lgbm_ranker_sigmoid.csv"))

    importance = [
        {"feature": name, "gain": float(gain)}
        for name, gain in sorted(
            zip(FEATURE_NAMES, model.booster_.feature_importance(importance_type="gain")),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return {
        "dataset": dataset,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_queries_first_fold": int(min(train_queries, len(train70_85))),
        "train_queries_recent_fold": int(min(train_queries // 2, len(train85_100))),
        "val_queries": int(min(val_queries, len(train85_100))),
        "val_mrr_testlike_candidates": val_mrr,
        "ranker_raw_file": osp.join(ranker_dir, f"{dataset}_lgbm_raw.npy"),
        "ranker_sigmoid_file": osp.join(ranker_dir, f"{dataset}_lgbm_ranker_sigmoid.csv"),
        "feature_importance": importance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--runs_dir", default="./experiments")
    parser.add_argument("--run_name", default="round46_lgbm_candidate_ranker")
    parser.add_argument("--base_pred_dir", default="./experiments/round44_bucketed_temporal_open")
    parser.add_argument("--recent_limit", type=int, default=320)
    parser.add_argument("--train_queries_d1", type=int, default=50000)
    parser.add_argument("--train_queries_d2", type=int, default=90000)
    parser.add_argument("--val_queries_d1", type=int, default=18000)
    parser.add_argument("--val_queries_d2", type=int, default=24000)
    parser.add_argument("--batch_rows", type=int, default=5000)
    parser.add_argument("--n_estimators", type=int, default=360)
    parser.add_argument("--learning_rate", type=float, default=0.035)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_child_samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.90)
    parser.add_argument("--reg_alpha", type=float, default=0.02)
    parser.add_argument("--reg_lambda", type=float, default=0.20)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        dest="variants",
        default=[],
        help="name:d1_alpha:d2_alpha:d1_margin:d2_margin",
    )
    args = parser.parse_args()

    if not args.variants:
        args.variants = [
            ("round46_lgbm_soft", 0.050, 0.050, 8.0, 8.0),
            ("round46_lgbm_medium", 0.090, 0.080, 12.0, 12.0),
            ("round46_lgbm_strong", 0.140, 0.115, 99.0, 99.0),
            ("round46_lgbm_very_strong", 0.220, 0.170, 99.0, 99.0),
        ]

    os.makedirs(osp.join(args.runs_dir, args.run_name), exist_ok=True)
    records = []
    ranker_raw_by_dataset = {}
    for dataset in DATASETS:
        record = run_dataset(dataset, args)
        records.append(record)
        ranker_raw_by_dataset[dataset] = np.load(record["ranker_raw_file"])

    variant_records = []
    for name, d1_alpha, d2_alpha, d1_margin, d2_margin in args.variants:
        for dataset in DATASETS:
            alpha = d1_alpha if dataset == "dataset1" else d2_alpha
            margin = d1_margin if dataset == "dataset1" else d2_margin
            base_path = osp.join(args.base_pred_dir, dataset, f"{dataset}_result.csv")
            base_scores = read_scores(base_path)
            final_scores, modified = apply_residual(base_scores, ranker_raw_by_dataset[dataset], alpha, margin)
            out_dir = osp.join(args.runs_dir, name, dataset)
            out_file = osp.join(out_dir, f"{dataset}_result.csv")
            write_scores(final_scores, out_file)
            variant_records.append(
                {
                    "name": name,
                    "dataset": dataset,
                    "alpha": alpha,
                    "margin_threshold": margin,
                    "modified_rows": int(modified),
                    "modified_ratio": float(modified / len(final_scores)),
                    "output_file": out_file,
                }
            )
        print("Submission zip:", pack_submission(osp.join(args.runs_dir, name)))

    summary = {
        "method": "lightgbm_lambdarank_candidate_reranker",
        "description": "Train LambdaRank on test-like 100-candidate groups sampled from public test candidates, then residual-blend it with the strongest available base predictions.",
        "base_pred_dir": args.base_pred_dir,
        "feature_names": FEATURE_NAMES,
        "records": records,
        "variants": variant_records,
    }
    summary_path = osp.join(args.runs_dir, args.run_name, "lgbm_candidate_ranker_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
