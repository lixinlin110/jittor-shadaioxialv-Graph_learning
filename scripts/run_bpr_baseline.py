from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight BPR-MF baseline and score official candidates."
    )
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. dataset1")
    parser.add_argument("--data_dir", default="data", help="directory containing dataset folders")
    parser.add_argument("--output_dir", default="results", help="where outputs are written")
    parser.add_argument("--factors", type=int, default=64, help="embedding dimension")
    parser.add_argument("--epochs", type=int, default=5, help="number of BPR epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--reg", type=float, default=5e-4, help="L2 regularization")
    parser.add_argument("--pop_alpha", type=float, default=0.01, help="popularity prior weight")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_neg_tries", type=int, default=50)
    parser.add_argument("--submission_copy", action="store_true", help="also write {dataset}_result.csv")
    return parser


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def read_dataset(data_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = data_dir / dataset
    train_file = dataset_dir / "train.csv"
    test_file = dataset_dir / "test.csv"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"missing train.csv or test.csv under {dataset_dir}")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    required_train = {"src", "dst", "time"}
    required_test = {"src", "time"}
    if not required_train.issubset(train_df.columns):
        raise ValueError(f"train.csv must contain columns {sorted(required_train)}")
    if not required_test.issubset(test_df.columns):
        raise ValueError(f"test.csv must contain columns {sorted(required_test)}")
    return train_df.sort_values("time").reset_index(drop=True), test_df


def build_maps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    src_values = pd.concat([train_df["src"], test_df["src"]], ignore_index=True).astype(int).unique()
    candidate_values = pd.concat([train_df["dst"], test_df.iloc[:, 2:].stack()], ignore_index=True).astype(int).unique()
    user_map = {int(value): idx for idx, value in enumerate(src_values)}
    item_map = {int(value): idx for idx, value in enumerate(candidate_values)}
    return user_map, item_map


def make_training_arrays(
    train_df: pd.DataFrame, user_map: dict[int, int], item_map: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, dict[int, set[int]], np.ndarray]:
    users = train_df["src"].astype(int).map(user_map).to_numpy(dtype=np.int64)
    items = train_df["dst"].astype(int).map(item_map).to_numpy(dtype=np.int64)
    positives: dict[int, set[int]] = defaultdict(set)
    for user, item in zip(users, items):
        positives[int(user)].add(int(item))
    item_pop = np.bincount(items, minlength=len(item_map)).astype(np.float32)
    return users, items, positives, item_pop


def sample_negative(
    rng: np.random.Generator,
    num_items: int,
    positives: set[int],
    max_tries: int,
) -> int:
    for _ in range(max_tries):
        neg = int(rng.integers(0, num_items))
        if neg not in positives:
            return neg
    return int(rng.integers(0, num_items))


def train_bpr(
    users: np.ndarray,
    items: np.ndarray,
    positives: dict[int, set[int]],
    num_users: int,
    num_items: int,
    factors: int,
    epochs: int,
    lr: float,
    reg: float,
    seed: int,
    max_neg_tries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    user_emb = 0.01 * rng.standard_normal((num_users, factors), dtype=np.float32)
    item_emb = 0.01 * rng.standard_normal((num_items, factors), dtype=np.float32)
    item_bias = np.zeros(num_items, dtype=np.float32)
    order = np.arange(len(users))

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        total_loss = 0.0
        iterator = tqdm(order, desc=f"BPR epoch {epoch}", ncols=100)
        for idx in iterator:
            user = int(users[idx])
            pos = int(items[idx])
            neg = sample_negative(rng, num_items, positives[user], max_neg_tries)

            user_vec = user_emb[user].copy()
            pos_vec = item_emb[pos].copy()
            neg_vec = item_emb[neg].copy()
            x_uij = float(item_bias[pos] - item_bias[neg] + np.dot(user_vec, pos_vec - neg_vec))
            clipped = max(min(x_uij, 30.0), -30.0)
            grad = 1.0 / (1.0 + math.exp(clipped))

            user_emb[user] += lr * (grad * (pos_vec - neg_vec) - reg * user_vec)
            item_emb[pos] += lr * (grad * user_vec - reg * pos_vec)
            item_emb[neg] += lr * (-grad * user_vec - reg * neg_vec)
            item_bias[pos] += lr * (grad - reg * item_bias[pos])
            item_bias[neg] += lr * (-grad - reg * item_bias[neg])

            total_loss += math.log1p(math.exp(-clipped))
            iterator.set_postfix(loss=f"{total_loss / max(1, iterator.n):.4f}")
    return user_emb, item_emb, item_bias


def score_candidates(
    test_df: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    item_bias: np.ndarray,
    item_pop: np.ndarray,
    pop_alpha: float,
) -> np.ndarray:
    rows = []
    max_pop = float(np.log1p(item_pop).max() + 1e-6)
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Scoring", ncols=100):
        src = int(row["src"])
        candidates = row.iloc[2:].astype(int).to_numpy()
        scores = np.zeros(len(candidates), dtype=np.float32)
        user_idx = user_map.get(src)
        for pos, candidate in enumerate(candidates):
            item_idx = item_map.get(int(candidate))
            if item_idx is None:
                continue
            value = float(item_bias[item_idx])
            if user_idx is not None:
                value += float(np.dot(user_emb[user_idx], item_emb[item_idx]))
            if pop_alpha:
                value += pop_alpha * float(np.log1p(item_pop[item_idx]) / max_pop)
            scores[pos] = value
        std = float(scores.std())
        normalized = scores - float(scores.mean())
        if std > 1e-6:
            normalized /= std
        rows.append(sigmoid(normalized))
    return np.vstack(rows)


def save_scores(scores: np.ndarray, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(",".join(f"{float(value):.8f}" for value in row) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    started_at = time.time()
    train_df, test_df = read_dataset(Path(args.data_dir), args.dataset)
    user_map, item_map = build_maps(train_df, test_df)
    users, items, positives, item_pop = make_training_arrays(train_df, user_map, item_map)

    user_emb, item_emb, item_bias = train_bpr(
        users=users,
        items=items,
        positives=positives,
        num_users=len(user_map),
        num_items=len(item_map),
        factors=args.factors,
        epochs=args.epochs,
        lr=args.lr,
        reg=args.reg,
        seed=args.seed,
        max_neg_tries=args.max_neg_tries,
    )

    scores = score_candidates(
        test_df=test_df,
        user_map=user_map,
        item_map=item_map,
        user_emb=user_emb,
        item_emb=item_emb,
        item_bias=item_bias,
        item_pop=item_pop,
        pop_alpha=args.pop_alpha,
    )

    dataset_dir = Path(args.output_dir) / args.dataset
    output_file = dataset_dir / f"{args.dataset}_bpr_result.csv"
    save_scores(scores, output_file)
    if args.submission_copy:
        save_scores(scores, dataset_dir / f"{args.dataset}_result.csv")

    np.savez_compressed(
        dataset_dir / "bpr_mf_model.npz",
        user_emb=user_emb,
        item_emb=item_emb,
        item_bias=item_bias,
        item_pop=item_pop,
    )
    metadata = {
        "dataset": args.dataset,
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
        "num_users": int(len(user_map)),
        "num_items": int(len(item_map)),
        "factors": args.factors,
        "epochs": args.epochs,
        "lr": args.lr,
        "reg": args.reg,
        "pop_alpha": args.pop_alpha,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "output_file": str(output_file),
    }
    with (dataset_dir / "bpr_mf_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
