import argparse
import json
import os
import os.path as osp
import shutil
from collections import Counter, defaultdict
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from lgbm_candidate_ranker import apply_residual, logit, pack_submission, read_scores, row_zscore, sigmoid, write_scores


N_CANDIDATES = 100
DAY = 86400


RICH_FEATURE_NAMES = [
    "pair_count_log",
    "pair_seen",
    "pair_rate",
    "pair_rec_7d",
    "pair_rec_30d",
    "pair_rec_180d",
    "src_recent_weight",
    "src_recent_rank_inv",
    "src_recent_time_decay",
    "src_seen_recent_5",
    "src_seen_recent_20",
    "dst_count_log",
    "dst_seen",
    "dst_share",
    "dst_rec_7d",
    "dst_rec_30d",
    "dst_rec_180d",
    "dst_win_30_log",
    "dst_win_90_log",
    "dst_win_180_log",
    "cand_pool_count_log",
    "cand_pool_share",
    "cand_pool_recency",
    "cand_pool_first_norm",
    "cand_pool_mean_time_norm",
    "src_cand_count_log",
    "src_cand_share",
    "src_cand_last_decay",
    "trans_count_log",
    "trans_weighted_log",
    "trans_max_rec",
    "trans_seen",
    "row_pair_count_z",
    "row_pair_rec_z",
    "row_src_recent_z",
    "row_dst_count_z",
    "row_dst_rec_z",
    "row_dst_win90_z",
    "row_cand_pool_z",
    "row_src_cand_z",
    "row_trans_weight_z",
]


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def signed_tag(value, scale=1000):
    sign = "n" if value < 0 else "p"
    return f"{sign}{int(round(abs(value) * scale)):03d}"


class RichTemporalStats:
    def __init__(self, df, recent_limit=720, transition_recent=12):
        df = df.sort_values("time").reset_index(drop=True)
        self.recent_limit = int(recent_limit)
        self.transition_recent = int(transition_recent)
        self.pair_count = defaultdict(int)
        self.pair_last = {}
        self.dst_count = defaultdict(int)
        self.dst_last = {}
        self.dst_times = defaultdict(list)
        self.src_count = defaultdict(int)
        self.src_seq = defaultdict(list)
        self.src_time_seq = defaultdict(list)
        self.src_recent = {}
        self.src_recent_time = {}
        self.src_recent_items = {}
        self.trans_count = defaultdict(int)
        self.trans_last = {}
        self.num_edges = max(1, len(df))
        self.time_min = int(df["time"].min())
        self.time_max = int(df["time"].max())
        self.time_span = max(1, self.time_max - self.time_min)
        self._build(df)

    def _build(self, df):
        for s_raw, d_raw, t_raw in tqdm(
            zip(df["src"].values, df["dst"].values, df["time"].values),
            total=len(df),
            ncols=120,
            desc="Build rich temporal stats",
        ):
            s = int(s_raw)
            d = int(d_raw)
            t = int(t_raw)
            key = (s, d)
            self.pair_count[key] += 1
            self.pair_last[key] = t
            self.dst_count[d] += 1
            self.dst_last[d] = t
            self.dst_times[d].append(t)
            self.src_count[s] += 1
            self.src_seq[s].append(d)
            self.src_time_seq[s].append(t)

        for d, times in list(self.dst_times.items()):
            self.dst_times[d] = np.asarray(times, dtype=np.int64)

        for s, seq in self.src_seq.items():
            times = self.src_time_seq[s]
            recent_weight = {}
            recent_time = {}
            rank = 1
            for d, t in zip(reversed(seq[-self.recent_limit :]), reversed(times[-self.recent_limit :])):
                if d not in recent_weight:
                    recent_weight[d] = 1.0 / rank
                    recent_time[d] = int(t)
                    rank += 1
            self.src_recent[s] = recent_weight
            self.src_recent_time[s] = recent_time
            self.src_recent_items[s] = tuple(int(x) for x in reversed(seq[-self.transition_recent :]))

            prev = None
            prev_time = None
            for d, t in zip(seq, times):
                if prev is not None:
                    key = (int(prev), int(d))
                    self.trans_count[key] += 1
                    self.trans_last[key] = int(t)
                prev = d
                prev_time = t

    def dst_window_count(self, dst, cur_time, window):
        times = self.dst_times.get(int(dst))
        if times is None or len(times) == 0:
            return 0
        left = np.searchsorted(times, int(cur_time) - int(window), side="left")
        right = np.searchsorted(times, int(cur_time), side="right")
        return int(max(0, right - left))


class TestCandidateContext:
    def __init__(self, test_df, keep_top_per_src=512):
        self.src = test_df["src"].values.astype(np.int64)
        self.time = test_df["time"].values.astype(np.int64)
        self.candidates = test_df.iloc[:, 2:].values.astype(np.int64)
        self.time_min = int(test_df["time"].min())
        self.time_max = int(test_df["time"].max())
        self.time_span = max(1, self.time_max - self.time_min)
        self.pool_count = defaultdict(int)
        self.pool_first = {}
        self.pool_last = {}
        self.pool_time_sum = defaultdict(float)
        self.by_src = defaultdict(list)
        self.src_pool_count = {}
        self.src_pool_last = {}
        self.src_pool_total = {}
        self._build_global()
        self._build_src_limited(test_df, keep_top_per_src)

    def _build_global(self):
        for row_idx, (s, t, row) in enumerate(tqdm(
            zip(self.src, self.time, self.candidates),
            total=len(self.src),
            ncols=120,
            desc="Build test candidate context",
        )):
            self.by_src[int(s)].append(row_idx)
            for d_raw in row:
                d = int(d_raw)
                self.pool_count[d] += 1
                if d not in self.pool_first:
                    self.pool_first[d] = int(t)
                self.pool_last[d] = int(t)
                self.pool_time_sum[d] += float(t)
        self.pool_size = max(1, int(self.candidates.size))

    def _build_src_limited(self, test_df, keep_top_per_src):
        cand_cols = test_df.columns[2:]
        for src, group in tqdm(test_df.groupby("src", sort=False), ncols=120, desc="Build src candidate context"):
            vals = group[cand_cols].values.ravel()
            if len(vals) == 0:
                continue
            counts = Counter(int(x) for x in vals)
            kept = {d for d, c in counts.items() if c >= 2}
            if len(kept) > keep_top_per_src:
                kept = {d for d, _ in counts.most_common(keep_top_per_src)}
            if not kept:
                continue
            src = int(src)
            self.src_pool_total[src] = int(len(vals))
            self.src_pool_count[src] = {d: int(counts[d]) for d in kept}
            last = {}
            for row in group.itertuples(index=False):
                t = int(getattr(row, "time"))
                for d_raw in row[2:]:
                    d = int(d_raw)
                    if d in kept:
                        last[d] = t
            self.src_pool_last[src] = last

    def sample_row_candidates(self, src, rng):
        rows = self.by_src.get(int(src), ())
        if len(rows) > 0 and rng.rand() < 0.80:
            return self.candidates[int(rows[rng.randint(len(rows))])].copy()
        return self.candidates[int(rng.randint(len(self.candidates)))].copy()


def make_rich_features(stats, context, src, cur_time, candidates):
    src = int(src)
    cur_time = int(cur_time)
    cand = np.asarray(candidates, dtype=np.int64)
    n = len(cand)
    out = {name: np.zeros(n, dtype=np.float64) for name in RICH_FEATURE_NAMES[:32]}

    tau_7 = 7 * DAY
    tau_30 = 30 * DAY
    tau_180 = 180 * DAY
    src_total = max(1, stats.src_count.get(src, 0))
    recent_map = stats.src_recent.get(src, {})
    recent_time = stats.src_recent_time.get(src, {})
    recent_items = stats.src_recent_items.get(src, ())
    src_cand_count_map = context.src_pool_count.get(src, {})
    src_cand_last_map = context.src_pool_last.get(src, {})

    for j, d_raw in enumerate(cand):
        d = int(d_raw)
        key = (src, d)
        pair_count = stats.pair_count.get(key, 0)
        if pair_count:
            out["pair_count_log"][j] = np.log1p(pair_count)
            out["pair_seen"][j] = 1.0
            out["pair_rate"][j] = pair_count / src_total
            delta = max(0, cur_time - stats.pair_last[key])
            out["pair_rec_7d"][j] = np.exp(-delta / tau_7)
            out["pair_rec_30d"][j] = np.exp(-delta / tau_30)
            out["pair_rec_180d"][j] = np.exp(-delta / tau_180)

        recent_weight = recent_map.get(d, 0.0)
        out["src_recent_weight"][j] = recent_weight
        if recent_weight > 0:
            out["src_recent_rank_inv"][j] = recent_weight
            delta = max(0, cur_time - recent_time.get(d, cur_time))
            out["src_recent_time_decay"][j] = np.exp(-delta / tau_30)
            rank = int(round(1.0 / recent_weight))
            out["src_seen_recent_5"][j] = 1.0 if rank <= 5 else 0.0
            out["src_seen_recent_20"][j] = 1.0 if rank <= 20 else 0.0

        dst_count = stats.dst_count.get(d, 0)
        if dst_count:
            out["dst_count_log"][j] = np.log1p(dst_count)
            out["dst_seen"][j] = 1.0
            out["dst_share"][j] = dst_count / stats.num_edges
            delta = max(0, cur_time - stats.dst_last[d])
            out["dst_rec_7d"][j] = np.exp(-delta / tau_7)
            out["dst_rec_30d"][j] = np.exp(-delta / tau_30)
            out["dst_rec_180d"][j] = np.exp(-delta / tau_180)
            out["dst_win_30_log"][j] = np.log1p(stats.dst_window_count(d, cur_time, 30 * DAY))
            out["dst_win_90_log"][j] = np.log1p(stats.dst_window_count(d, cur_time, 90 * DAY))
            out["dst_win_180_log"][j] = np.log1p(stats.dst_window_count(d, cur_time, 180 * DAY))

        pool_count = context.pool_count.get(d, 0)
        if pool_count:
            out["cand_pool_count_log"][j] = np.log1p(pool_count)
            out["cand_pool_share"][j] = pool_count / context.pool_size
            last_delta = abs(cur_time - context.pool_last.get(d, cur_time))
            out["cand_pool_recency"][j] = np.exp(-last_delta / max(1.0, 30 * DAY))
            first_t = context.pool_first.get(d, context.time_min)
            out["cand_pool_first_norm"][j] = (first_t - context.time_min) / context.time_span
            out["cand_pool_mean_time_norm"][j] = (
                context.pool_time_sum.get(d, float(context.time_min)) / max(1, pool_count) - context.time_min
            ) / context.time_span

        src_pool_count = src_cand_count_map.get(d, 0)
        if src_pool_count:
            out["src_cand_count_log"][j] = np.log1p(src_pool_count)
            out["src_cand_share"][j] = src_pool_count / max(1, context.src_pool_total.get(src, 0))
            delta = abs(cur_time - src_cand_last_map.get(d, cur_time))
            out["src_cand_last_decay"][j] = np.exp(-delta / max(1.0, 30 * DAY))

        trans_count_total = 0
        trans_weighted = 0.0
        trans_max_rec = 0.0
        for rank, prev_d in enumerate(recent_items, start=1):
            trans_key = (int(prev_d), d)
            tc = stats.trans_count.get(trans_key, 0)
            if not tc:
                continue
            trans_count_total += tc
            trans_weighted += tc / rank
            delta = max(0, cur_time - stats.trans_last.get(trans_key, cur_time))
            trans_max_rec = max(trans_max_rec, float(np.exp(-delta / tau_180)))
        if trans_count_total:
            out["trans_count_log"][j] = np.log1p(trans_count_total)
            out["trans_weighted_log"][j] = np.log1p(trans_weighted)
            out["trans_max_rec"][j] = trans_max_rec
            out["trans_seen"][j] = 1.0

    matrix = np.column_stack(
        [
            out["pair_count_log"],
            out["pair_seen"],
            out["pair_rate"],
            out["pair_rec_7d"],
            out["pair_rec_30d"],
            out["pair_rec_180d"],
            out["src_recent_weight"],
            out["src_recent_rank_inv"],
            out["src_recent_time_decay"],
            out["src_seen_recent_5"],
            out["src_seen_recent_20"],
            out["dst_count_log"],
            out["dst_seen"],
            out["dst_share"],
            out["dst_rec_7d"],
            out["dst_rec_30d"],
            out["dst_rec_180d"],
            out["dst_win_30_log"],
            out["dst_win_90_log"],
            out["dst_win_180_log"],
            out["cand_pool_count_log"],
            out["cand_pool_share"],
            out["cand_pool_recency"],
            out["cand_pool_first_norm"],
            out["cand_pool_mean_time_norm"],
            out["src_cand_count_log"],
            out["src_cand_share"],
            out["src_cand_last_decay"],
            out["trans_count_log"],
            out["trans_weighted_log"],
            out["trans_max_rec"],
            out["trans_seen"],
            row_zscore(out["pair_count_log"]),
            row_zscore(out["pair_rec_30d"]),
            row_zscore(out["src_recent_weight"]),
            row_zscore(out["dst_count_log"]),
            row_zscore(out["dst_rec_30d"]),
            row_zscore(out["dst_win_90_log"]),
            row_zscore(out["cand_pool_count_log"]),
            row_zscore(out["src_cand_count_log"]),
            row_zscore(out["trans_weighted_log"]),
        ]
    )
    return matrix.astype(np.float32)


def sample_queries(df, max_queries, seed, recency_power=1.5):
    if not max_queries or len(df) <= max_queries:
        return df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    t = df["time"].values.astype(np.float64)
    denom = max(1.0, float(t.max() - t.min()))
    w = 0.25 + np.exp(recency_power * ((t - t.min()) / denom))
    w = w / w.sum()
    idx = rng.choice(len(df), size=max_queries, replace=False, p=w)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def build_matrix(query_df, stats, context, max_queries, seed, desc, recency_power):
    rng = np.random.RandomState(seed)
    sampled = sample_queries(query_df, max_queries, seed, recency_power)
    n_queries = len(sampled)
    x = np.empty((n_queries * N_CANDIDATES, len(RICH_FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(n_queries * N_CANDIDATES, dtype=np.int8)
    group = np.full(n_queries, N_CANDIDATES, dtype=np.int32)
    offset = 0
    for row in tqdm(sampled.itertuples(index=False), total=n_queries, ncols=120, desc=desc):
        src = int(getattr(row, "src"))
        dst = int(getattr(row, "dst"))
        cur_time = int(getattr(row, "time"))
        candidates = context.sample_row_candidates(src, rng)
        labels = (candidates == dst).astype(np.int8)
        if labels.sum() == 0:
            replace_idx = int(rng.randint(N_CANDIDATES))
            candidates[replace_idx] = dst
            labels[replace_idx] = 1
        x[offset : offset + N_CANDIDATES] = make_rich_features(stats, context, src, cur_time, candidates)
        y[offset : offset + N_CANDIDATES] = labels
        offset += N_CANDIDATES
    return x, y, group


def train_ranker(x_train, y_train, group_train, args, seed):
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
    model.fit(x_train, y_train, group=group_train, feature_name=RICH_FEATURE_NAMES)
    return model


def evaluate_mrr(model, val_df, stats, context, max_queries, seed):
    rng = np.random.RandomState(seed)
    sampled = sample_queries(val_df, max_queries, seed, recency_power=0.5)
    rr = []
    for row in tqdm(sampled.itertuples(index=False), total=len(sampled), ncols=120, desc="Validate rich ranker"):
        src = int(getattr(row, "src"))
        dst = int(getattr(row, "dst"))
        cur_time = int(getattr(row, "time"))
        candidates = context.sample_row_candidates(src, rng)
        labels = (candidates == dst).astype(np.int8)
        if labels.sum() == 0:
            replace_idx = int(rng.randint(N_CANDIDATES))
            candidates[replace_idx] = dst
            labels[replace_idx] = 1
        pred = model.predict(make_rich_features(stats, context, src, cur_time, candidates))
        positive = np.where(labels > 0)[0]
        best_pos = pred[positive].max()
        rr.append(1.0 / (1 + int(np.sum(pred > best_pos))))
    return float(np.mean(rr)) if rr else 0.0


def score_test(model, test_df, stats, context, batch_rows):
    candidates = test_df.iloc[:, 2:].values.astype(np.int64)
    srcs = test_df["src"].values.astype(np.int64)
    times = test_df["time"].values.astype(np.int64)
    out = np.empty(candidates.shape, dtype=np.float64)
    for start in tqdm(range(0, len(test_df), batch_rows), ncols=120, desc="Score rich dataset2 model"):
        end = min(start + batch_rows, len(test_df))
        feats = []
        for i in range(start, end):
            feats.append(make_rich_features(stats, context, srcs[i], times[i], candidates[i]))
        x = np.vstack(feats)
        out[start:end] = model.predict(x).reshape(end - start, N_CANDIDATES)
    return out


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

    stats_hist0 = RichTemporalStats(hist0, args.recent_limit, args.transition_recent)
    x_train, y_train, group_train = build_matrix(
        train0_tail,
        stats_hist0,
        context,
        args.train_queries_tail,
        args.seed,
        "Build rich split0 tail fold",
        args.recency_power,
    )
    val_model = train_ranker(x_train, y_train, group_train, args, args.seed)

    stats_split0 = RichTemporalStats(split0, args.recent_limit, args.transition_recent)
    val_mrr = evaluate_mrr(val_model, split1, stats_split0, context, args.val_queries, args.seed + 77)

    x_extra, y_extra, group_extra = build_matrix(
        split1,
        stats_split0,
        context,
        args.train_queries_split1,
        args.seed + 17,
        "Build rich split1 fold",
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
    raw_file = osp.join(out_dir, "dataset2_rich_raw.npy")
    sigmoid_file = osp.join(out_dir, "dataset2_rich_sigmoid.csv")
    np.save(raw_file, raw)
    write_scores(sigmoid(raw), sigmoid_file)
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
        "train_queries_tail": int(min(args.train_queries_tail, len(train0_tail))),
        "train_queries_split1": int(min(args.train_queries_split1, len(split1))),
        "val_queries": int(min(args.val_queries, len(split1))),
        "val_mrr_testlike_candidates": float(val_mrr),
        "raw_file": raw_file,
        "sigmoid_file": sigmoid_file,
        "feature_importance": importance,
    }


def write_variant(name, fixed_d1_dir, base_d2, raw, alpha, margin):
    run_root = osp.join("./experiments", name)
    d1_src = osp.join(fixed_d1_dir, "dataset1", "dataset1_result.csv")
    if not osp.exists(d1_src):
        raise FileNotFoundError(d1_src)
    d1_dst = osp.join(run_root, "dataset1", "dataset1_result.csv")
    os.makedirs(osp.dirname(d1_dst), exist_ok=True)
    shutil.copyfile(d1_src, d1_dst)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--runs_dir", default="./experiments")
    parser.add_argument("--run_name", default="round65_d2_rich_ranker")
    parser.add_argument("--fixed_d1_dir", default="./experiments/current_best_12971305063123648")
    parser.add_argument("--base_d2_dir", default="./experiments/current_best_12971305063123648")
    parser.add_argument("--recent_limit", type=int, default=960)
    parser.add_argument("--transition_recent", type=int, default=12)
    parser.add_argument("--keep_top_per_src", type=int, default=512)
    parser.add_argument("--split0_cut", type=float, default=0.84)
    parser.add_argument("--train_queries_tail", type=int, default=95000)
    parser.add_argument("--train_queries_split1", type=int, default=65000)
    parser.add_argument("--val_queries", type=int, default=30000)
    parser.add_argument("--recency_power", type=float, default=1.2)
    parser.add_argument("--batch_rows", type=int, default=5000)
    parser.add_argument("--n_estimators", type=int, default=460)
    parser.add_argument("--learning_rate", type=float, default=0.030)
    parser.add_argument("--num_leaves", type=int, default=95)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_child_samples", type=int, default=86)
    parser.add_argument("--subsample", type=float, default=0.88)
    parser.add_argument("--colsample_bytree", type=float, default=0.90)
    parser.add_argument("--reg_alpha", type=float, default=0.04)
    parser.add_argument("--reg_lambda", type=float, default=0.45)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--alphas", default="-0.04,-0.02,0.02,0.04")
    parser.add_argument("--margins", default="0.12,0.18")
    args = parser.parse_args()

    record = run_dataset2(args)
    raw = np.load(record["raw_file"])
    base_d2 = read_scores(osp.join(args.base_d2_dir, "dataset2", "dataset2_result.csv"))

    variants = []
    for alpha in parse_float_list(args.alphas):
        for margin in parse_float_list(args.margins):
            name = f"{args.run_name}_{signed_tag(alpha)}_m{int(round(margin * 100)):03d}"
            variants.append(write_variant(name, args.fixed_d1_dir, base_d2, raw, alpha, margin))

    preferred = [
        f"{args.run_name}_p020_m018",
        f"{args.run_name}_n020_m018",
        f"{args.run_name}_p040_m018",
        f"{args.run_name}_n040_m018",
    ]
    recommended = []
    for idx, name in enumerate(preferred, start=1):
        src_zip = osp.join(args.runs_dir, name, "result.zip")
        if not osp.exists(src_zip):
            continue
        alias_dir = osp.join(args.runs_dir, f"round65_recommended_{idx}_{name.replace(args.run_name + '_', '')}")
        os.makedirs(alias_dir, exist_ok=True)
        dst_zip = osp.join(alias_dir, "result.zip")
        shutil.copyfile(src_zip, dst_zip)
        recommended.append(dst_zip)

    summary = {
        "method": "dataset2_rich_lgbm_ranker",
        "description": "Feature-rich dataset2 reranker: split-aware validation, source recent sequence features, item transition features, and public test candidate context.",
        "args": vars(args),
        "record": record,
        "variants": variants,
        "recommended_order": recommended,
    }
    summary_path = osp.join(args.runs_dir, args.run_name, "round65_dataset2_rich_summary.json")
    os.makedirs(osp.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Summary:", summary_path)
    print("Recommended:")
    for path in recommended:
        print(path)


if __name__ == "__main__":
    main()
