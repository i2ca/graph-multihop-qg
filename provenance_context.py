from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


class ProvenanceContextRetriever:
    def __init__(self, graphrag_dir: str | Path = "./graphrag") -> None:
        graphrag_dir = Path(graphrag_dir)
        self.entities = pd.read_parquet(
            graphrag_dir / "create_final_entities.parquet"
        )
        self.relationships = pd.read_parquet(
            graphrag_dir / "create_final_relationships.parquet"
        )
        self.text_units = pd.read_parquet(
            graphrag_dir / "create_final_text_units.parquet"
        )
        self.documents = pd.read_parquet(
            graphrag_dir / "create_final_documents.parquet"
        )

        self._entity_rows = self._index_unique(self.entities, "name", "entity")
        self._text_unit_rows = self._index_unique(
            self.text_units, "id", "text unit"
        )
        self._document_rows = self._index_unique(self.documents, "id", "document")

    @staticmethod
    def _index_unique(
        dataframe: pd.DataFrame, column: str, label: str
    ) -> dict[str, dict[str, Any]]:
        duplicated = dataframe[column].duplicated(keep=False)
        if duplicated.any():
            values = dataframe.loc[duplicated, column].astype(str).unique().tolist()
            raise ValueError(f"Duplicate {label} {column}s: {values}")
        return {
            str(row[column]): row.to_dict()
            for _, row in dataframe.iterrows()
        }

    @staticmethod
    def _ids(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, Iterable):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    def _relationship_rows(
        self, source: str, target: str, directed: bool
    ) -> list[dict[str, Any]]:
        forward = (self.relationships["source"] == source) & (
            self.relationships["target"] == target
        )
        selected = forward
        if not directed:
            reverse = (self.relationships["source"] == target) & (
                self.relationships["target"] == source
            )
            selected = forward | reverse
        return [row.to_dict() for _, row in self.relationships[selected].iterrows()]

    def retrieve_for_relation(
        self,
        source: str,
        target: str,
        *,
        directed: bool = False,
        include_entity_context: bool = False,
        max_text_units: int | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.retrieve_for_path(
            [source, target],
            directed=directed,
            include_entity_context=include_entity_context,
            max_text_units=max_text_units,
            max_tokens=max_tokens,
        )

    def retrieve_for_path(
        self,
        entity_path: Sequence[str],
        *,
        directed: bool = False,
        include_entity_context: bool = True,
        max_text_units: int | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        if len(entity_path) < 2:
            raise ValueError("entity_path must contain at least two entities")
        if max_text_units is not None and max_text_units < 1:
            raise ValueError("max_text_units must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        for entity in entity_path:
            if entity not in self._entity_rows:
                raise KeyError(f"Unknown entity: {entity}")

        relation_rows: list[dict[str, Any]] = []
        ranked_text_unit_ids: list[str] = []
        for source, target in zip(entity_path, entity_path[1:]):
            rows = self._relationship_rows(source, target, directed)
            if not rows:
                direction = "directed " if directed else ""
                raise KeyError(f"No {direction}relationship found: {source} -> {target}")
            relation_rows.extend(rows)
            for row in rows:
                ranked_text_unit_ids.extend(self._ids(row.get("text_unit_ids")))

        if include_entity_context:
            for entity in entity_path:
                ranked_text_unit_ids.extend(
                    self._ids(self._entity_rows[entity].get("text_unit_ids"))
                )

        ranked_text_unit_ids = list(dict.fromkeys(ranked_text_unit_ids))

        results: list[dict[str, Any]] = []
        used_tokens = 0
        selected_entity_ids = {
            str(self._entity_rows[name]["id"]): name for name in entity_path
        }
        selected_relation_ids = {
            str(row["id"]): {
                "source": str(row["source"]),
                "target": str(row["target"]),
            }
            for row in relation_rows
        }

        for text_unit_id in ranked_text_unit_ids:
            if max_text_units is not None and len(results) >= max_text_units:
                break
            row = self._text_unit_rows.get(text_unit_id)
            if row is None:
                raise KeyError(f"Unknown provenance text unit: {text_unit_id}")

            n_tokens = int(row.get("n_tokens") or 0)
            if max_tokens is not None and used_tokens + n_tokens > max_tokens:
                continue

            document_records = []
            for document_id in self._ids(row.get("document_ids")):
                document = self._document_rows.get(document_id)
                if document is None:
                    raise KeyError(f"Unknown provenance document: {document_id}")
                document_records.append(
                    {"id": document_id, "title": str(document.get("title", ""))}
                )

            results.append(
                {
                    "text_unit_id": text_unit_id,
                    "text": str(row["text"]),
                    "n_tokens": n_tokens,
                    "documents": document_records,
                    "path_entities": [
                        selected_entity_ids[entity_id]
                        for entity_id in self._ids(row.get("entity_ids"))
                        if entity_id in selected_entity_ids
                    ],
                    "path_relationships": [
                        selected_relation_ids[relationship_id]
                        for relationship_id in self._ids(row.get("relationship_ids"))
                        if relationship_id in selected_relation_ids
                    ],
                }
            )
            used_tokens += n_tokens

        return results

    @staticmethod
    def format_for_prompt(passages: Sequence[dict[str, Any]]) -> str:
        blocks = []
        for index, passage in enumerate(passages, start=1):
            titles = ", ".join(
                document["title"] for document in passage["documents"]
            ) or "unknown document"
            blocks.append(
                f"[Source {index} | {titles} | text_unit={passage['text_unit_id']}]\n"
                f"{passage['text']}"
            )
        return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve provenance-linked context for a GraphRAG entity path."
    )
    parser.add_argument("entities", nargs="+", help="Entity names in path order")
    parser.add_argument("--graphrag-dir", default="./graphrag")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--relationship-only", action="store_true")
    parser.add_argument("--max-text-units", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--json", action="store_true", help="Print structured JSON instead of prompt text"
    )
    args = parser.parse_args()

    retriever = ProvenanceContextRetriever(args.graphrag_dir)
    passages = retriever.retrieve_for_path(
        args.entities,
        directed=args.directed,
        include_entity_context=not args.relationship_only,
        max_text_units=args.max_text_units,
        max_tokens=args.max_tokens,
    )
    if args.json:
        print(json.dumps(passages, ensure_ascii=False, indent=2))
    else:
        print(retriever.format_for_prompt(passages))


if __name__ == "__main__":
    main()
