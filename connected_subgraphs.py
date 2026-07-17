from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd


class ConnectedSubgraphDiscovery:
    def __init__(self, graphrag_dir: str | Path = "./graphrag") -> None:
        graphrag_dir = Path(graphrag_dir)
        self.entities = pd.read_parquet(graphrag_dir / "create_final_entities.parquet")
        self.relationships = pd.read_parquet(graphrag_dir / "create_final_relationships.parquet")
        self.entity_rows = {
            str(row["name"]): row.to_dict() for _, row in self.entities.iterrows()
        }
        if len(self.entity_rows) != len(self.entities):
            raise ValueError("Entity names must be unique")
        self.names = tuple(sorted(self.entity_rows))
        self.outgoing = {name: set() for name in self.names}
        self.incoming = {name: set() for name in self.names}
        self.incident = {name: set() for name in self.names}
        self.edge_rows: dict[frozenset[str], list[dict[str, Any]]] = {}
        for _, relationship in self.relationships.iterrows():
            source = str(relationship["source"])
            target = str(relationship["target"])
            if source not in self.entity_rows or target not in self.entity_rows:
                continue
            self.outgoing[source].add(target)
            self.incoming[target].add(source)
            self.incident[source].add(target)
            self.incident[target].add(source)
            self.edge_rows.setdefault(frozenset((source, target)), []).append(
                relationship.to_dict()
            )

    def _neighbors(self, name: str, directed: bool) -> set[str]:
        return self.outgoing[name] if directed else self.incident[name]

    def is_connected(self, entities: Sequence[str], directed: bool = False) -> bool:
        selected = set(entities)
        if not selected:
            return False
        unknown = selected.difference(self.entity_rows)
        if unknown:
            raise KeyError(f"Unknown entities: {sorted(unknown)}")
        start = next(iter(selected))
        visited = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in self._neighbors(current, directed).intersection(selected):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited == selected

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            return value.tolist()
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    def build_subgraph(self, entities: Sequence[str]) -> dict[str, Any]:
        selected = set(entities)
        unknown = selected.difference(self.entity_rows)
        if unknown:
            raise KeyError(f"Unknown entities: {sorted(unknown)}")
        ordered_names = sorted(selected)
        nodes = []
        for name in ordered_names:
            row = self.entity_rows[name]
            nodes.append(
                {
                    key: self._json_value(value)
                    for key, value in row.items()
                    if key != "graph_embedding" and key != "description_embedding"
                }
            )
        edges = []
        for pair, rows in self.edge_rows.items():
            if pair.issubset(selected):
                for row in rows:
                    edges.append(
                        {key: self._json_value(value) for key, value in row.items()}
                    )
        edges.sort(key=lambda row: (str(row["source"]), str(row["target"]), str(row["id"])))
        degree = {name: 0 for name in ordered_names}
        in_degree = {name: 0 for name in ordered_names}
        out_degree = {name: 0 for name in ordered_names}
        for edge in edges:
            source = str(edge["source"])
            target = str(edge["target"])
            degree[source] += 1
            degree[target] += 1
            out_degree[source] += 1
            in_degree[target] += 1
        return {
            "entities": nodes,
            "relationships": edges,
            "entity_names": ordered_names,
            "entity_count": len(nodes),
            "relationship_count": len(edges),
            "degree": degree,
            "in_degree": in_degree,
            "out_degree": out_degree,
        }

    def enumerate(
        self,
        min_size: int = 2,
        max_size: int = 3,
        directed: bool = False,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        self._validate_sizes(min_size, max_size)
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        emitted = 0
        for size in range(min_size, max_size + 1):
            for entities in itertools.combinations(self.names, size):
                if self.is_connected(entities, directed):
                    yield self.build_subgraph(entities)
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return

    def sample(
        self,
        count: int,
        min_size: int = 2,
        max_size: int = 3,
        directed: bool = False,
        seed: int | None = None,
        max_attempts: int | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_sizes(min_size, max_size)
        if count < 1:
            raise ValueError("count must be >= 1")
        if max_attempts is None:
            max_attempts = max(100, count * 50)
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        rng = random.Random(seed)
        discovered: dict[tuple[str, ...], dict[str, Any]] = {}
        attempts = 0
        viable_starts = [name for name in self.names if self._neighbors(name, directed)]
        while len(discovered) < count and attempts < max_attempts and viable_starts:
            attempts += 1
            target_size = rng.randint(min_size, max_size)
            selected = {rng.choice(viable_starts)}
            while len(selected) < target_size:
                frontier = set()
                for entity in selected:
                    frontier.update(self._neighbors(entity, directed))
                frontier.difference_update(selected)
                if not frontier:
                    break
                selected.add(rng.choice(sorted(frontier)))
            if len(selected) >= min_size and self.is_connected(tuple(selected), directed):
                key = tuple(sorted(selected))
                discovered.setdefault(key, self.build_subgraph(key))
        return list(discovered.values())

    def _validate_sizes(self, min_size: int, max_size: int) -> None:
        if min_size < 1:
            raise ValueError("min_size must be >= 1")
        if max_size < min_size:
            raise ValueError("max_size must be >= min_size")
        if max_size > len(self.names):
            raise ValueError("max_size exceeds the number of entities")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphrag-dir", default="./graphrag")
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    discovery = ConnectedSubgraphDiscovery(args.graphrag_dir)
    if args.sample is not None:
        subgraphs = discovery.sample(
            args.sample,
            args.min_size,
            args.max_size,
            args.directed,
            args.seed,
            args.max_attempts,
        )
    else:
        subgraphs = list(
            discovery.enumerate(
                args.min_size, args.max_size, args.directed, args.limit
            )
        )
    print(json.dumps(subgraphs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
