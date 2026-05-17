from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import parse_candidates, read_candidate_rows
from src.models import TimeAwareReranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank candidate destinations.")
    parser.add_argument("--dataset", default="dataset1")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--candidate_file", required=True)
    parser.add_argument("--output_file", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = TimeAwareReranker.load(args.model_path)
    candidates = read_candidate_rows(args.candidate_file)

    rows = []
    for index, row in candidates.iterrows():
        candidate_list = parse_candidates(row)
        if not candidate_list:
            continue
        ranked = model.rank(row["source"], row["time"], candidate_list)
        rows.append(
            {
                "source": row["source"],
                "time": row["time"],
                "ranked_destinations": " ".join(candidate for candidate, _ in ranked),
                "ranked_scores": " ".join(f"{score:.6f}" for _, score in ranked),
            }
        )

    output_file = args.output_file or f"results/{args.dataset}/submission.csv"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"saved submission: {output_path}")


if __name__ == "__main__":
    main()
