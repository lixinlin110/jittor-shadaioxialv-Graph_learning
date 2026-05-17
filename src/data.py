from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SOURCE_ALIASES = ("source", "src", "u", "user", "user_id", "from", "head")
DESTINATION_ALIASES = ("destination", "dst", "v", "item", "item_id", "to", "tail", "target")
TIME_ALIASES = ("time", "timestamp", "ts", "t", "datetime")


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame | None
    test: pd.DataFrame | None


def resolve_dataset_dir(data_dir: str | Path, dataset: str) -> Path:
    root = Path(data_dir)
    nested = root / dataset
    return nested if nested.exists() else root


def read_edges(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        frame = pd.read_csv(path)

    return normalize_edge_columns(frame)


def normalize_edge_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    lower_to_original = {str(col).lower(): col for col in frame.columns}

    for standard_name, aliases in (
        ("source", SOURCE_ALIASES),
        ("destination", DESTINATION_ALIASES),
        ("time", TIME_ALIASES),
    ):
        for alias in aliases:
            if alias in lower_to_original:
                rename[lower_to_original[alias]] = standard_name
                break

    frame = frame.rename(columns=rename).copy()
    missing = {"source", "destination", "time"} - set(frame.columns)
    if missing:
        raise ValueError(
            "missing required edge columns "
            f"{sorted(missing)}; available columns: {list(frame.columns)}"
        )

    frame["source"] = frame["source"].astype(str)
    frame["destination"] = frame["destination"].astype(str)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["source", "destination", "time"])
    frame = frame.sort_values("time").reset_index(drop=True)
    return frame


def read_candidate_rows(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        frame = pd.read_csv(path)

    rename: dict[str, str] = {}
    lower_to_original = {str(col).lower(): col for col in frame.columns}
    for standard_name, aliases in (
        ("source", SOURCE_ALIASES),
        ("destination", DESTINATION_ALIASES),
        ("time", TIME_ALIASES),
    ):
        for alias in aliases:
            if alias in lower_to_original:
                rename[lower_to_original[alias]] = standard_name
                break

    frame = frame.rename(columns=rename).copy()
    missing = {"source", "time"} - set(frame.columns)
    if missing:
        raise ValueError(
            "missing required candidate columns "
            f"{sorted(missing)}; available columns: {list(frame.columns)}"
        )
    frame["source"] = frame["source"].astype(str)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["source", "time"]).reset_index(drop=True)
    if "destination" in frame.columns:
        frame["destination"] = frame["destination"].astype(str)
    return frame


def find_split_file(dataset_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def load_dataset(data_dir: str | Path, dataset: str) -> DatasetBundle:
    dataset_dir = resolve_dataset_dir(data_dir, dataset)
    train_path = find_split_file(
        dataset_dir,
        ("train.csv", "train_edges.csv", "edges_train.csv", "train.parquet"),
    )
    if train_path is None:
        raise FileNotFoundError(
            f"cannot find train file under {dataset_dir}; expected train.csv"
        )

    valid_path = find_split_file(
        dataset_dir,
        ("valid.csv", "val.csv", "validation.csv", "valid_candidates.csv"),
    )
    test_path = find_split_file(
        dataset_dir,
        ("test.csv", "test_candidates.csv", "candidate.csv", "candidates.csv"),
    )

    return DatasetBundle(
        train=read_edges(train_path),
        valid=read_edges(valid_path) if valid_path else None,
        test=read_candidate_rows(test_path) if test_path else None,
    )


def make_demo_dataset(seed: int = 2026) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    sources = [f"u{i}" for i in range(16)]
    destinations = [f"v{i}" for i in range(40)]

    rows = []
    for step in range(360):
        source = sources[step % len(sources)]
        trend = (step // 24 + sources.index(source)) % len(destinations)
        if rng.random() < 0.72:
            destination = destinations[(trend + int(rng.integers(0, 5))) % len(destinations)]
        else:
            destination = destinations[int(rng.integers(0, len(destinations)))]
        rows.append({"source": source, "destination": destination, "time": step})

    train = pd.DataFrame(rows[:280])
    valid_rows = []
    for row in rows[280:330]:
        true_dst = row["destination"]
        distractors = rng.choice(
            [d for d in destinations if d != true_dst],
            size=19,
            replace=False,
        )
        candidates = [true_dst] + list(distractors)
        rng.shuffle(candidates)
        valid_rows.append({**row, "candidates": " ".join(candidates)})

    test_rows = []
    for row in rows[330:350]:
        candidates = rng.choice(destinations, size=20, replace=False)
        test_rows.append(
            {
                "source": row["source"],
                "destination": row["destination"],
                "time": row["time"],
                "candidates": " ".join(candidates),
            }
        )

    return DatasetBundle(
        train=normalize_edge_columns(train),
        valid=normalize_edge_columns(pd.DataFrame(valid_rows)),
        test=normalize_edge_columns(pd.DataFrame(test_rows)),
    )


def parse_candidates(row: pd.Series) -> list[str]:
    if "candidates" in row and pd.notna(row["candidates"]):
        raw = str(row["candidates"]).replace(",", " ").replace("|", " ")
        return [item for item in raw.split() if item]

    candidate_cols = [
        col
        for col in row.index
        if str(col).lower() != "candidates"
        and str(col).lower().startswith(("candidate", "cand_"))
    ]
    if candidate_cols:
        values = [row[col] for col in candidate_cols if pd.notna(row[col])]
        return [str(value) for value in values]

    if "destination" in row and pd.notna(row["destination"]):
        return [str(row["destination"])]

    return []
