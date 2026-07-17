from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from connected_subgraphs import ConnectedSubgraphDiscovery
from graph_ambiguity_analysis import GraphAmbiguityAnalyzer
from llm_api import LlmApi
from mhqg_common import json_text, query_json
from multi_hop_validity_review import MultiHopValidityReviewer
from provenance_context import ProvenanceContextRetriever
from single_hop_grounding_review import SingleHopGroundingReviewer
from structure_frequency_sampling import StructureFrequencySampler


class QuestionStructureCatalogue:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            value = {"catalogue_version": "1.0", "structures": []}
        self.value = value
        self.structures = {
            item["structure_id"]: item for item in value.get("structures", [])
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.value["structures"] = list(self.structures.values())
        self.path.write_text(
            json.dumps(self.value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, structure_id: str) -> dict[str, Any] | None:
        return self.structures.get(structure_id)

    def put(self, structure: dict[str, Any]) -> None:
        structure.setdefault("review_status", "pending")
        self.structures[structure["structure_id"]] = structure

    def add_example(self, structure_id: str, example: dict[str, Any]) -> None:
        example.setdefault("review_status", "pending")
        self.structures[structure_id].setdefault("examples", []).append(example)

    def approve_structure(self, structure_id: str, replacement: dict[str, Any] | None = None) -> dict[str, Any]:
        structure = self.structures[structure_id]
        if replacement is not None:
            examples = structure.get("examples", [])
            structure = {**replacement, "structure_id": structure_id}
            structure.setdefault("examples", examples)
            self.structures[structure_id] = structure
        structure["review_status"] = "approved"
        self.save()
        return structure

    def approve_example(self, structure_id: str, example_index: int, replacement: dict[str, Any] | None = None) -> dict[str, Any]:
        examples = self.structures[structure_id].setdefault("examples", [])
        example = examples[example_index]
        if replacement is not None:
            example = replacement
            examples[example_index] = example
        example["review_status"] = "approved"
        self.save()
        return example

    def approved_structure(self, structure_id: str) -> dict[str, Any] | None:
        structure = self.get(structure_id)
        if structure is None or structure.get("review_status") != "approved":
            return None
        return structure

    def approved_examples(self, structure_id: str) -> list[dict[str, Any]]:
        structure = self.structures[structure_id]
        return [example for example in structure.get("examples", []) if example.get("review_status") == "approved"]


class StructureInducer:
    def __init__(self, llm: LlmApi, graphrag_dir: str | Path = "./graphrag") -> None:
        self.llm = llm
        self.discovery = ConnectedSubgraphDiscovery(graphrag_dir)

    def induce(
        self, subgraph: dict[str, Any], structure_id: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        prompt = f"""<instructions>
Convert the abstract connected graph into a reusable question structure. Assign stable node roles n0, n1, and so on. Define a coherent hop order, dependencies, open entities, hidden intermediate entities, and one answer entity. Preserve every graph edge. Reverse traversal is allowed and must be marked. Return only JSON with schema {{"name":string,"topology":{{"directed":boolean,"nodes":[{{"role":string,"entity_type":string}}],"edges":[{{"source":string,"target":string,"relation_role":string}}]}},"hop_order":[object],"open_entities":[string],"hidden_entities":[string],"answer_entity":string}}.
</instructions>
<schema>{json_text(schema)}</schema>
<instance>{json_text(subgraph)}</instance>"""
        result = query_json(self.llm, prompt)
        return {"structure_id": structure_id, **result, "examples": []}


class StructureExampleGenerator:
    def __init__(self, llm: LlmApi, provenance: ProvenanceContextRetriever) -> None:
        self.llm = llm
        self.provenance = provenance

    def generate(
        self,
        structure: dict[str, Any],
        role_entities: dict[str, str],
        seed_examples: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        passages = []
        seen = set()
        for hop in structure.get("hop_order", []):
            if "source" not in hop or "target" not in hop:
                continue
            for passage in self.provenance.retrieve_for_relation(
                role_entities[hop["source"]],
                role_entities[hop["target"]],
                directed=False,
                include_entity_context=True,
            ):
                if passage["text_unit_id"] not in seen:
                    seen.add(passage["text_unit_id"])
                    passages.append(passage)
        prompt = f"""<instructions>
Create a high-quality worked example for the supplied question structure and entity assignment. Generate one factual single-hop question and answer for every relational hop, then compose a fluent open-ended multi-hop question. Mention every open entity, do not mention hidden entities, and make the answer the assigned answer entity. Use only the context. Return only JSON with schema {{"single_hop_questions":[{{"hop":integer,"question":string,"answer":string}}],"multi_hop_question":string,"answer":string}}.
</instructions>
<structure>{json_text(structure)}</structure>
<entities>{json_text(role_entities)}</entities>
<seed_examples>{json_text(list(seed_examples))}</seed_examples>
<context>{self.provenance.format_for_prompt(passages)}</context>"""
        result = query_json(self.llm, prompt)
        return {
            "entities": role_entities,
            **result,
            "contexts": passages,
            "relation_descriptions": [
                str(relationship.get("description", ""))
                for relationship in self.provenance.relationships.to_dict("records")
                if str(relationship.get("source", "")) in role_entities.values()
                and str(relationship.get("target", "")) in role_entities.values()
            ],
        }

    @staticmethod
    def _ordered_entities(
        structure: dict[str, Any], role_entities: dict[str, str]
    ) -> list[str]:
        hops = [
            hop
            for hop in structure.get("hop_order", [])
            if "source" in hop and "target" in hop
        ]
        if not hops:
            raise ValueError("Structure has no relational hops")
        path = [role_entities[hops[0]["source"]]]
        for hop in hops:
            source = role_entities[hop["source"]]
            target = role_entities[hop["target"]]
            if path[-1] == source:
                path.append(target)
            elif path[-1] == target:
                path.append(source)
        if len(path) < 2:
            raise ValueError("Unable to derive a path")
        return path


class GraphQuestionGenerator:
    def __init__(
        self,
        llm: LlmApi,
        evaluator_llm: LlmApi,
        graphrag_dir: str | Path = "./graphrag",
        catalogue_path: str | Path = "./question_structure_catalogue.json",
        general_examples: Sequence[dict[str, Any]] = (),
    ) -> None:
        self.llm = llm
        self.evaluator_llm = evaluator_llm
        self.graphrag_dir = Path(graphrag_dir)
        self.catalogue = QuestionStructureCatalogue(catalogue_path)
        self.provenance = ProvenanceContextRetriever(graphrag_dir)
        self.inducer = StructureInducer(llm, graphrag_dir)
        self.example_generator = StructureExampleGenerator(llm, self.provenance)
        self.single_reviewer = SingleHopGroundingReviewer(llm, self.provenance)
        self.multi_reviewer = MultiHopValidityReviewer(llm, self.provenance)
        self.ambiguity = GraphAmbiguityAnalyzer(llm, graphrag_dir, self.provenance)
        self.general_examples = list(general_examples)

    def build_structure_examples(
        self,
        count_per_structure: int = 1,
        seed_examples: Sequence[dict[str, Any]] = (),
        min_size: int = 3,
        max_size: int = 5,
    ) -> list[dict[str, Any]]:
        """Induce structures and persist corpus-specific worked examples."""
        if count_per_structure < 1:
            raise ValueError("count_per_structure must be >= 1")
        sampler = StructureFrequencySampler(
            self.graphrag_dir, min_size, max_size, directed=True
        )
        created = []
        for structure_id, instances in sampler.instances_by_structure.items():
            structure = self.catalogue.get(structure_id)
            if structure is None:
                structure = self.inducer.induce(
                    instances[0], structure_id, sampler.schemas[structure_id]
                )
                self.catalogue.put(structure)
            existing = len(structure.get("examples", []))
            selected = self._representative_instances(instances, max(0, count_per_structure - existing))
            for subgraph in selected:
                assignment = self._role_assignment(structure, subgraph)
                example = self.example_generator.generate(
                    structure, assignment, seed_examples or self.general_examples
                )
                self.catalogue.add_example(structure_id, example)
                created.append({"structure_id": structure_id, "example": example})
        self.catalogue.save()
        return created

    @staticmethod
    def _instance_quality(subgraph: dict[str, Any]) -> float:
        relationships = subgraph.get("relationships", [])
        weights = [float(row.get("weight", 0.0) or 0.0) for row in relationships]
        evidence = sum(len(row.get("text_unit_ids", []) or []) for row in relationships)
        descriptions = sum(bool(str(row.get("description", "")).strip()) for row in relationships)
        return sum(weights) + 0.25 * evidence + 0.5 * descriptions

    @staticmethod
    def _instance_signature(subgraph: dict[str, Any]) -> set[str]:
        values = {str(entity.get("name", "")).casefold() for entity in subgraph.get("entities", [])}
        values.update(str(relationship.get("description", "")).casefold() for relationship in subgraph.get("relationships", []))
        return {value for value in values if value}

    @classmethod
    def _representative_instances(cls, instances: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        remaining = list(instances)
        selected: list[dict[str, Any]] = []
        while remaining and len(selected) < count:
            selected_signatures = [cls._instance_signature(item) for item in selected]
            def score(item: dict[str, Any]) -> tuple[float, str]:
                signature = cls._instance_signature(item)
                similarity = max((len(signature & other) / len(signature | other) if signature | other else 0.0 for other in selected_signatures), default=0.0)
                names = "\0".join(sorted(str(value) for value in item.get("entity_names", [])))
                return cls._instance_quality(item) - similarity, names
            chosen = max(remaining, key=score)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    @staticmethod
    def _tokens(value: Any) -> set[str]:
        return set(re.findall(r"[\w]+", json.dumps(value, ensure_ascii=False).casefold()))

    @classmethod
    def _best_examples(cls, examples: Sequence[dict[str, Any]], subgraph: dict[str, Any], count: int) -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("count must be >= 1")
        if not examples:
            return []
        target = cls._tokens([relationship.get("description", "") for relationship in subgraph.get("relationships", [])])
        def score(example: dict[str, Any]) -> tuple[float, int]:
            source = cls._tokens({"questions": example.get("single_hop_questions", []), "contexts": example.get("contexts", []), "relations": example.get("relation_descriptions", [])})
            overlap = len(target & source) / len(target | source) if target | source else 0.0
            return overlap, len(example.get("contexts", []))
        return sorted(examples, key=score, reverse=True)[:count]

    @classmethod
    def _best_example(cls, examples: Sequence[dict[str, Any]], subgraph: dict[str, Any]) -> dict[str, Any] | None:
        selected = cls._best_examples(examples, subgraph, 1)
        return selected[0] if selected else None

    @staticmethod
    def _role_assignment(
        structure: dict[str, Any], subgraph: dict[str, Any]
    ) -> dict[str, str]:
        nodes = structure["topology"]["nodes"]
        available: dict[str, list[str]] = {}
        for entity in subgraph["entities"]:
            available.setdefault(str(entity.get("type", "")), []).append(
                str(entity["name"])
            )
        assignment = {}
        for node in nodes:
            entity_type = str(node.get("entity_type", ""))
            choices = available.get(entity_type, [])
            if not choices:
                raise ValueError(f"No entity for role {node['role']}")
            assignment[node["role"]] = choices.pop(0)
        return assignment

    @staticmethod
    def _relational_hops(structure: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            hop
            for hop in structure.get("hop_order", [])
            if "source" in hop and "target" in hop
        ]

    def _operation_hops(structure: dict[str, Any]) -> list[dict[str, Any]]:
        return [hop for hop in structure.get("hop_order", []) if "operation" in hop]

    def _resolved_hops(
        structure: dict[str, Any], assignment: dict[str, str]
    ) -> list[dict[str, Any]]:
        resolved = []
        for hop in structure.get("hop_order", []):
            item = dict(hop)
            for key in ("source", "target"):
                if key in item:
                    item[key] = assignment[item[key]]
            for key in ("inputs", "returns"):
                if key in item:
                    item[key] = [assignment.get(role, role) for role in item[key]]
            resolved.append(item)
        return resolved

    def _single_prompt(
        self,
        source: str,
        target: str,
        relationship: str,
        context: str,
        examples: Sequence[dict[str, Any]],
    ) -> str:
        return f"""{json_text(list(examples))}
<instructions>
Given two entities and a directed relationship, create one factual single-hop question about the source whose unique answer is the target. Do not mention the target. Use only the context. Return only JSON with schema {{"question":string,"answer":string}}.
</instructions>
<source>{source}</source><target>{target}</target><relationship>{relationship}</relationship><context>{context}</context>"""

    def _composition_prompt(
        self,
        structure: dict[str, Any],
        assignment: dict[str, str],
        questions: list[dict[str, Any]],
        context: str,
        examples: Sequence[dict[str, Any]],
    ) -> str:
        open_entities = [
            assignment[role] for role in structure.get("open_entities", [])
        ]
        hidden_entities = [
            assignment[role] for role in structure.get("hidden_entities", [])
        ]
        return f"""{json_text(list(examples))}
<instructions>
Compose one coherent open-ended multi-hop question from all supplied single-hop questions and answers. Explicitly include every open entity and never mention hidden entities or the answer. Preserve the dependency diagram, integrate all hops naturally, require every hop, and ensure one context-supported answer. Return only JSON with schema {{"question":string,"answer":string}}.
</instructions>
<open_entities>{json_text(open_entities)}</open_entities><hidden_entities>{json_text(hidden_entities)}</hidden_entities><entity_diagram>{json_text(structure)}</entity_diagram><questions>{json_text(questions)}</questions><context>{context}</context>"""

    def generate_one(
        self,
        sampled: dict[str, Any],
        ambiguity_policy: str = "discard",
        self_review: bool = True,
        max_restarts: int = 2,
        review_rigor: str = "strict",
        auto_examples: bool = True,
        example_count: int = 3,
    ) -> dict[str, Any]:
        structure_id = sampled["structure_id"]
        subgraph = sampled["subgraph"]
        structure = self.catalogue.approved_structure(structure_id)
        if structure is None:
            if self.catalogue.get(structure_id) is None:
                structure = self.inducer.induce(subgraph, structure_id, sampled["schema"])
                self.catalogue.put(structure)
                if auto_examples:
                    assignment = self._role_assignment(structure, subgraph)
                    generated_example = self.example_generator.generate(structure, assignment, self.general_examples)
                    self.catalogue.add_example(structure_id, generated_example)
                self.catalogue.save()
            return {"accepted": False, "stage": "structure_review_required", "structure_id": structure_id}
        assignment = self._role_assignment(structure, subgraph)
        examples = self.catalogue.approved_examples(structure_id)
        examples = examples or [example for example in self.general_examples if example.get("review_status") == "approved"]
        if not examples:
            return {"accepted": False, "stage": "example_review_required", "structure_id": structure_id}
        selected_examples = self._best_examples(examples, subgraph, example_count)
        hop_results = []
        entity_path = []
        for index, hop in enumerate(self._relational_hops(structure), start=1):
            source = assignment[hop["source"]]
            target = assignment[hop["target"]]
            if hop.get("traverses_edge_in_reverse"):
                relationship_rows = self.provenance._relationship_rows(
                    target, source, True
                )
            else:
                relationship_rows = self.provenance._relationship_rows(
                    source, target, True
                )
            if not relationship_rows:
                relationship_rows = self.provenance._relationship_rows(
                    source, target, False
                )
            relationship = "\n".join(
                str(row.get("description", "")) for row in relationship_rows
            )
            passages = self.provenance.retrieve_for_relation(
                source, target, directed=False, include_entity_context=False
            )
            context = self.provenance.format_for_prompt(passages)
            single_examples = [example["single_hop_questions"][index - 1] for example in selected_examples if index <= len(example.get("single_hop_questions", []))]
            generated = query_json(
                self.llm,
                self._single_prompt(
                    source, target, relationship, context, single_examples
                ),
            )
            question = str(generated["question"])
            review = (
                self.single_reviewer.review_with_regeneration(question, source, target)
                if self_review
                else {"status": "accepted", "question": question}
            )
            if review["status"] != "accepted":
                return {
                    "accepted": False,
                    "stage": "single_hop_review",
                    "review": review,
                }
            hop_results.append(
                {
                    "hop": index,
                    "source": source,
                    "target": target,
                    "question": review["question"],
                    "answer": target,
                }
            )
            if not entity_path:
                entity_path.extend([source, target])
            elif entity_path[-1] == source:
                entity_path.append(target)
        all_entities = list(dict.fromkeys(assignment.values()))
        context_passages = []
        seen = set()
        for hop in hop_results:
            for passage in self.provenance.retrieve_for_relation(
                hop["source"],
                hop["target"],
                directed=False,
                include_entity_context=True,
            ):
                if passage["text_unit_id"] not in seen:
                    seen.add(passage["text_unit_id"])
                    context_passages.append(passage)
        context = self.provenance.format_for_prompt(context_passages)
        composed = query_json(
            self.llm,
            self._composition_prompt(
                structure, assignment, hop_results, context, selected_examples
            ),
        )
        question = str(composed["question"])
        answer_role = structure["answer_entity"]
        expected_answer = assignment.get(answer_role)
        operation_review = None
        if self._operation_hops(structure) or expected_answer is None:
            operation_review = query_json(
                self.llm,
                f"""<instructions>Execute the graph operations using only the facts and context. Support comparison, argmin, argmax, counting, temporal ordering, intersection, and selection. Return only JSON with schema {{"valid":boolean,"answer":string,"operation_results":[object],"reason":string}}.</instructions><structure>{json_text(structure)}</structure><entities>{json_text(assignment)}</entities><hops>{json_text(hop_results)}</hops><context>{context}</context>""",
            )
            if operation_review.get("valid") is not True:
                return {
                    "accepted": False,
                    "stage": "operation_review",
                    "review": operation_review,
                }
            expected_answer = str(operation_review.get("answer", ""))
        if not expected_answer:
            expected_answer = str(composed.get("answer", ""))
        resolved_hops = self._resolved_hops(structure, assignment)
        multi_review = None
        if self_review:
            multi_review = self.multi_reviewer.review_structure(
                question, resolved_hops, expected_answer, rigor=review_rigor
            )
            if not multi_review["valid"]:
                return {
                    "accepted": False,
                    "stage": "multi_hop_review",
                    "review": multi_review,
                }
        ambiguity = None
        if (
            len(entity_path) >= 2
            and entity_path[-1] == expected_answer
            and not self._operation_hops(structure)
            and len(entity_path) == len(assignment)
        ):
            ambiguity = self.ambiguity.analyze(
                question, entity_path, policy=ambiguity_policy
            )
        else:
            ambiguity = self.ambiguity.analyze_structure(
                question,
                structure,
                assignment,
                expected_answer,
                policy=ambiguity_policy,
            )
        if ambiguity is not None:
            if not ambiguity["accepted"]:
                return {"accepted": False, "stage": "ambiguity", "review": ambiguity}
            question = ambiguity["question"]
        return {
            "accepted": True,
            "question": question,
            "answer": expected_answer,
            "structure_id": structure_id,
            "structure": structure,
            "entities": assignment,
            "entity_path": entity_path,
            "resolved_hops": resolved_hops,
            "single_hops": hop_results,
            "context": context,
            "context_passages": context_passages,
            "multi_hop_review": multi_review,
            "operation_review": operation_review,
            "ambiguity": ambiguity,
            "distractors": ambiguity.get("distractors", []) if ambiguity else [],
            "all_entities": all_entities,
        }

    def generate(
        self,
        count: int,
        policy: str = "proportional",
        min_size: int = 3,
        max_size: int = 5,
        ambiguity_policy: str = "discard",
        self_review: bool = True,
        seed: int | None = None,
        review_rigor: str = "strict",
        max_attempts: int | None = None,
        auto_examples: bool = True,
        example_count: int = 3,
        max_restarts: int = 2,
    ) -> list[dict[str, Any]]:
        if review_rigor not in {"strict", "balanced", "loose"}:
            raise ValueError("review_rigor must be strict, balanced, or loose")
        sampler = StructureFrequencySampler(
            self.graphrag_dir, min_size, max_size, directed=True
        )
        max_attempts = max_attempts or max(count * 20, count)
        output = []
        attempt = 0
        while len(output) < count and attempt < max_attempts:
            restarts = 0
            while restarts <= max_restarts and attempt < max_attempts:
                sampled = sampler.sample(1, policy=policy, seed=None if seed is None else seed + attempt, replace=True)[0]
                attempt += 1
                result = self.generate_one(sampled, ambiguity_policy, self_review, review_rigor=review_rigor, auto_examples=auto_examples, example_count=example_count)
                if result["accepted"]:
                    output.append(result)
                    break
                restarts += 1
        return output
