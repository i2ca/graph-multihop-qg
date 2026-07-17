from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from connected_subgraphs import ConnectedSubgraphDiscovery


class StructureFrequencySampler:
    def __init__(
        self,
        graphrag_dir: str | Path = "./graphrag",
        min_size: int = 2,
        max_size: int = 3,
        directed: bool = True,
        subgraphs: Iterable[dict[str, Any]] | None = None,
        enumeration_limit: int | None = None,
        sampling_threshold: int = 100000,
        sample_count: int = 10000,
        seed: int | None = None,
    ) -> None:
        self.discovery = ConnectedSubgraphDiscovery(graphrag_dir)
        self.directed = directed
        if subgraphs is None:
            combinations = sum(math.comb(len(self.discovery.names), size) for size in range(min_size, max_size + 1))
            if combinations > sampling_threshold:
                subgraphs = self.discovery.sample(sample_count, min_size, max_size, False, seed)
            else:
                subgraphs = self.discovery.enumerate(min_size=min_size, max_size=max_size, directed=False, limit=enumeration_limit)
        self.instances_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.schemas: dict[str, dict[str, Any]] = {}
        for subgraph in subgraphs:
            structure_id, schema = self.canonicalize(subgraph, directed)
            self.instances_by_structure[structure_id].append(subgraph)
            self.schemas[structure_id] = schema
        self.instances_by_structure = dict(self.instances_by_structure)
        if not self.instances_by_structure:
            raise ValueError("No connected subgraphs were provided or discovered")

    @staticmethod
    def _node_type(node: dict[str, Any]) -> str:
        value = node.get("type", node.get("entity_type", ""))
        return "" if value is None else str(value)

    @classmethod
    def canonicalize(
        cls, subgraph: dict[str, Any], directed: bool = True
    ) -> tuple[str, dict[str, Any]]:
        nodes = subgraph.get("entities", [])
        relationships = subgraph.get("relationships", [])
        if not nodes:
            raise ValueError("A structure must contain at least one entity")
        names = [str(node["name"]) for node in nodes]
        if len(names) != len(set(names)):
            raise ValueError("Entity names within a subgraph must be unique")
        node_types = {str(node["name"]): cls._node_type(node) for node in nodes}
        edge_counts: dict[tuple[str, str], int] = defaultdict(int)
        for relationship in relationships:
            source = str(relationship["source"])
            target = str(relationship["target"])
            if source not in node_types or target not in node_types:
                continue
            if directed:
                edge_counts[(source, target)] += 1
            else:
                edge_counts[tuple(sorted((source, target)))] += 1
        best_key: str | None = None
        best_schema: dict[str, Any] | None = None
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in names:
            grouped[node_types[name]].append(name)
        type_order = sorted(grouped)
        permutation_groups = [
            list(itertools.permutations(sorted(grouped[node_type])))
            for node_type in type_order
        ]
        for group_permutations in itertools.product(*permutation_groups):
            order = tuple(
                name for permutation in group_permutations for name in permutation
            )
            position = {name: index for index, name in enumerate(order)}
            types = [node_types[name] for name in order]
            canonical_edges = []
            for edge, count in edge_counts.items():
                source, target = edge
                left = position[source]
                right = position[target]
                if not directed and left > right:
                    left, right = right, left
                canonical_edges.append([left, right, count])
            canonical_edges.sort()
            schema = {
                "directed": directed,
                "entity_types": types,
                "edges": canonical_edges,
            }
            key = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            if best_key is None or key < best_key:
                best_key = key
                best_schema = schema
        if best_key is None or best_schema is None:
            raise RuntimeError("Unable to canonicalize structure")
        return best_key, best_schema

    def frequencies(self) -> list[dict[str, Any]]:
        total = sum(len(instances) for instances in self.instances_by_structure.values())
        output = []
        for structure_id, instances in self.instances_by_structure.items():
            frequency = len(instances)
            output.append(
                {
                    "structure_id": structure_id,
                    "schema": self.schemas[structure_id],
                    "frequency": frequency,
                    "proportion": frequency / total,
                }
            )
        output.sort(key=lambda item: (-item["frequency"], item["structure_id"]))
        return output

    def _weights(
        self,
        structure_ids: Sequence[str],
        policy: str,
        strength: float,
    ) -> list[float]:
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("strength must be a positive finite number")
        frequencies = [len(self.instances_by_structure[key]) for key in structure_ids]
        if policy == "proportional":
            return [float(frequency) for frequency in frequencies]
        if policy == "common":
            return [float(frequency) ** strength for frequency in frequencies]
        if policy == "rare":
            return [1.0 / (float(frequency) ** strength) for frequency in frequencies]
        if policy == "uniform":
            return [1.0] * len(structure_ids)
        raise ValueError("policy must be proportional, common, rare, or uniform")

    def sample(
        self,
        count: int,
        policy: str = "proportional",
        strength: float = 2.0,
        seed: int | None = None,
        replace: bool = True,
    ) -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("count must be >= 1")
        rng = random.Random(seed)
        available = {
            key: list(instances)
            for key, instances in self.instances_by_structure.items()
        }
        if not replace and count > sum(len(items) for items in available.values()):
            raise ValueError("count exceeds the number of available instances")
        output = []
        for _ in range(count):
            structure_ids = sorted(key for key, items in available.items() if items)
            if not structure_ids:
                break
            weights = self._weights(structure_ids, policy, strength)
            structure_id = rng.choices(structure_ids, weights=weights, k=1)[0]
            instances = available[structure_id]
            index = rng.randrange(len(instances))
            instance = instances[index]
            output.append(
                {
                    "structure_id": structure_id,
                    "schema": self.schemas[structure_id],
                    "frequency": len(self.instances_by_structure[structure_id]),
                    "sampling_policy": policy,
                    "subgraph": instance,
                }
            )
            if not replace:
                instances.pop(index)
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphrag-dir", default="./graphrag")
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--enumeration-limit", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--policy",
        choices=("proportional", "common", "rare", "uniform"),
        default="proportional",
    )
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--without-replacement", action="store_true")
    parser.add_argument("--frequencies", action="store_true")
    args = parser.parse_args()
    sampler = StructureFrequencySampler(
        graphrag_dir=args.graphrag_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        directed=not args.undirected,
        enumeration_limit=args.enumeration_limit,
    )
    if args.frequencies or args.count is None:
        output = sampler.frequencies()
    else:
        output = sampler.sample(
            count=args.count,
            policy=args.policy,
            strength=args.strength,
            seed=args.seed,
            replace=not args.without_replacement,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
