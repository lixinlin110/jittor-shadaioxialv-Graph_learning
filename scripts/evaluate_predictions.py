from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import evaluate_ranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ranked destination predictions with MRR, Recall@K, HitRate@K and NDCG@K."
    )
    parser.add_argument("--input", default=None, help="CSV file with targets and ranked predictions")
    parser.add_argument("--target_col", default="destination")
    parser.add_argument("--ranked_col", default="ranked_destinations")
    parser.add_argument("--ks", default="1,5,10", help="comma separated K values")
    parser.add_argument("--demo", action="store_true", help="run a built-in demo")
    return parser


def split_ranked_items(value: object) -> list[str]:
    text = str(value).replace(",", " ").replace("|", " ")
    return [item for item in text.split() if item]


def load_ranked_lists(
    path: str | Path,
    target_col: str,
    ranked_col: str,
) -> tuple[list[list[str]], list[str]]:
    frame = pd.read_csv(path)
    if target_col not in frame.columns:
        raise ValueError(f"target column not found: {target_col}")

    if ranked_col in frame.columns:
        ranked_lists = [split_ranked_items(value) for value in frame[ranked_col]]
    else:
        candidate_cols = [
            col
            for col in frame.columns
            if str(col).lower().startswith(("candidate", "cand_", "rank_"))
        ]
        if not candidate_cols:
            raise ValueError(
                f"ranked column '{ranked_col}' not found and no candidate columns found"
            )
        ranked_lists = [
            [str(row[col]) for col in candidate_cols if pd.notna(row[col])]
            for _, row in frame.iterrows()
        ]

    targets = [str(value) for value in frame[target_col]]
    return ranked_lists, targets


def demo_data() -> tuple[list[list[str]], list[str]]:
    ranked_lists = [
        ["v2", "v1", "v3", "v4"],
        ["v8", "v3", "v1", "v2"],
        ["v5", "v6", "v7", "v8"],
    ]
    targets = ["v1", "v8", "v9"]
    return ranked_lists, targets


def main() -> None:
    args = build_parser().parse_args()
    ks = tuple(int(item) for item in args.ks.split(",") if item.strip())

    if args.demo:
        ranked_lists, targets = demo_data()
    elif args.input:
        ranked_lists, targets = load_ranked_lists(args.input, args.target_col, args.ranked_col)
    else:
        raise SystemExit("please provide --input or use --demo")

    metrics = evaluate_ranking(ranked_lists, targets, ks=ks)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
