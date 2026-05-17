from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RerankConfig:
    num_neighbors: int = 30
    time_decay_tau: float = 30.0
    source_weight: float = 1.0
    recent_weight: float = 1.5
    global_weight: float = 0.2
    repeat_weight: float = 0.4


class TimeAwareReranker:
    """Lightweight candidate reranker for temporal link prediction.

    This module is intentionally small and dependency-light. It mirrors the
    signals we want to add around the official CRAFT baseline: recent source
    history, time-decayed interactions and candidate-level reranking.
    """

    def __init__(self, config: RerankConfig | None = None):
        self.config = config or RerankConfig()
        self.source_history: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self.source_destination_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()
        self.max_time = 0.0

    def fit(self, edges: pd.DataFrame) -> "TimeAwareReranker":
        required = {"source", "destination", "time"}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"missing columns for fit: {sorted(missing)}")

        for row in edges.sort_values("time").itertuples(index=False):
            source = str(row.source)
            destination = str(row.destination)
            timestamp = float(row.time)
            self.source_history[source].append((timestamp, destination))
            self.source_destination_counts[source][destination] += 1
            self.global_counts[destination] += 1
            self.max_time = max(self.max_time, timestamp)
        return self

    def score(self, source: str, candidate: str, timestamp: float) -> float:
        source = str(source)
        candidate = str(candidate)
        timestamp = float(timestamp)
        history = [
            item for item in self.source_history.get(source, []) if item[0] <= timestamp
        ]
        recent_history = history[-self.config.num_neighbors :]

        source_count = self.source_destination_counts.get(source, Counter())[candidate]
        global_count = self.global_counts[candidate]

        decayed = 0.0
        for event_time, destination in recent_history:
            if destination != candidate:
                continue
            gap = max(timestamp - event_time, 0.0)
            decayed += float(np.exp(-gap / max(self.config.time_decay_tau, 1e-6)))

        repeated = 1.0 if source_count > 0 else 0.0
        score = 0.0
        score += self.config.source_weight * np.log1p(source_count)
        score += self.config.recent_weight * decayed
        score += self.config.global_weight * np.log1p(global_count)
        score += self.config.repeat_weight * repeated
        return float(score)

    def rank(
        self, source: str, timestamp: float, candidates: list[str]
    ) -> list[tuple[str, float]]:
        scored = [
            (str(candidate), self.score(source, str(candidate), timestamp))
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "source_history": {
                source: [[event_time, destination] for event_time, destination in history]
                for source, history in self.source_history.items()
            },
            "source_destination_counts": {
                source: dict(counter)
                for source, counter in self.source_destination_counts.items()
            },
            "global_counts": dict(self.global_counts),
            "max_time": self.max_time,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TimeAwareReranker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(RerankConfig(**payload.get("config", {})))
        model.source_history = defaultdict(
            list,
            {
                source: [(float(event_time), str(destination)) for event_time, destination in history]
                for source, history in payload.get("source_history", {}).items()
            },
        )
        model.source_destination_counts = defaultdict(
            Counter,
            {
                source: Counter({str(k): int(v) for k, v in counter.items()})
                for source, counter in payload.get("source_destination_counts", {}).items()
            },
        )
        model.global_counts = Counter(
            {str(k): int(v) for k, v in payload.get("global_counts", {}).items()}
        )
        model.max_time = float(payload.get("max_time", 0.0))
        return model
