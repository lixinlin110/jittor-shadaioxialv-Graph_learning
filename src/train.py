from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import load_dataset, make_demo_dataset, parse_candidates
from src.metrics import evaluate_ranking
from src.models import RerankConfig, TimeAwareReranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate a time-aware reranker for temporal graph recommendation."
    )
    parser.add_argument("--dataset", default="dataset1")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--demo", action="store_true", help="run with synthetic demo data")

    # Keep these names compatible with the official CRAFT baseline commands.
    parser.add_argument("--model", default="CRAFT")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_neighbors", type=int, default=30)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--time_decay_tau", type=float, default=30.0)
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--recent_weight", type=float, default=1.5)
    parser.add_argument("--global_weight", type=float, default=0.2)
    parser.add_argument("--repeat_weight", type=float, default=0.4)
    return parser


def evaluate_if_available(model: TimeAwareReranker, valid_frame) -> dict[str, float]:
    if valid_frame is None or valid_frame.empty:
        return {}

    ranked_lists: list[list[str]] = []
    targets: list[str] = []
    for _, row in valid_frame.iterrows():
        candidates = parse_candidates(row)
        if not candidates:
            continue
        ranked = model.rank(row["source"], row["time"], candidates)
        ranked_lists.append([candidate for candidate, _ in ranked])
        targets.append(str(row["destination"]))

    return evaluate_ranking(ranked_lists, targets, ks=(1, 5, 10)) if targets else {}


def main() -> None:
    args = build_parser().parse_args()

    bundle = make_demo_dataset() if args.demo else load_dataset(args.data_dir, args.dataset)
    config = RerankConfig(
        num_neighbors=args.num_neighbors,
        time_decay_tau=args.time_decay_tau,
        source_weight=args.source_weight,
        recent_weight=args.recent_weight,
        global_weight=args.global_weight,
        repeat_weight=args.repeat_weight,
    )
    model = TimeAwareReranker(config).fit(bundle.train)

    dataset_dir = Path(args.output_dir) / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_path = dataset_dir / "time_aware_reranker.json"
    metrics_path = dataset_dir / "metrics.json"
    metadata_path = dataset_dir / "run_metadata.json"

    model.save(model_path)
    metrics = evaluate_if_available(model, bundle.valid)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "dataset": args.dataset,
        "model_interface": args.model,
        "epochs": args.epochs,
        "early_stop": args.early_stop,
        "batch_size": args.batch_size,
        "num_neighbors": args.num_neighbors,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "learning_rate": args.learning_rate,
        "train_edges": int(len(bundle.train)),
        "valid_rows": 0 if bundle.valid is None else int(len(bundle.valid)),
        "artifact": str(model_path),
        "note": "This is a lightweight time-aware reranker around the official CRAFT baseline interface.",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved model: {model_path}")
    if metrics:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print("no validation candidate file found; training artifact was still saved")


if __name__ == "__main__":
    main()
