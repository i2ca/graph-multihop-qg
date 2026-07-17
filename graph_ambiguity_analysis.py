from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from llm_api import LlmApi
from provenance_context import ProvenanceContextRetriever


class GraphAmbiguityAnalyzer:
    def __init__(
        self,
        llm: LlmApi,
        graphrag_dir: str | Path = "./graphrag",
        provenance: ProvenanceContextRetriever | None = None,
        max_candidate_paths: int = 500,
        validation_batch_size: int = 20,
    ) -> None:
        if max_candidate_paths < 1:
            raise ValueError("max_candidate_paths must be >= 1")
        if validation_batch_size < 1:
            raise ValueError("validation_batch_size must be >= 1")
        self.llm = llm
        self.max_candidate_paths = max_candidate_paths
        self.validation_batch_size = validation_batch_size
        self.last_search_complete = True
        self.provenance = provenance or ProvenanceContextRetriever(graphrag_dir)
        self.entities = self.provenance.entities
        self.relationships = self.provenance.relationships
        self.entity_rows = {
            str(row["name"]): row.to_dict() for _, row in self.entities.iterrows()
        }
        self.outgoing: dict[str, set[str]] = defaultdict(set)
        self.incoming: dict[str, set[str]] = defaultdict(set)
        self.relationship_rows: dict[tuple[str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for _, row in self.relationships.iterrows():
            source = str(row["source"])
            target = str(row["target"])
            if source not in self.entity_rows or target not in self.entity_rows:
                continue
            self.outgoing[source].add(target)
            self.incoming[target].add(source)
            self.relationship_rows[(source, target)].append(row.to_dict())

    @staticmethod
    def _extract_json(response: str) -> dict[str, Any]:
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("The LLM did not return a JSON object")
            value = json.loads(response[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("The LLM response must be a JSON object")
        return value

    def _query_json(self, prompt: str) -> dict[str, Any]:
        return self._extract_json(self.llm.query(prompt))

    def _entity_type(self, name: str) -> str:
        if name not in self.entity_rows:
            raise KeyError(f"Unknown entity: {name}")
        value = self.entity_rows[name].get("type")
        return "" if value is None or pd.isna(value) else str(value)

    def _neighbors(self, entity: str, directed: bool) -> set[str]:
        if directed:
            return self.outgoing.get(entity, set())
        return self.outgoing.get(entity, set()) | self.incoming.get(entity, set())

    def _edges_between(
        self, source: str, target: str, directed: bool
    ) -> list[dict[str, Any]]:
        rows = list(self.relationship_rows.get((source, target), []))
        if not directed:
            rows.extend(self.relationship_rows.get((target, source), []))
        return rows

    def enumerate_candidate_paths(
        self,
        intended_path: Sequence[str],
        open_indices: Sequence[int] = (0,),
        directed: bool = False,
    ) -> list[list[str]]:
        if len(intended_path) < 2:
            raise ValueError("intended_path must contain at least two entities")
        if len(set(intended_path)) != len(intended_path):
            raise ValueError("intended_path must be simple")
        for entity in intended_path:
            self._entity_type(entity)
        open_indices = tuple(sorted(set(open_indices)))
        if not open_indices:
            raise ValueError("At least one open entity index is required")
        if any(index < 0 or index >= len(intended_path) for index in open_indices):
            raise ValueError("open_indices contains an invalid index")
        fixed = {index: intended_path[index] for index in open_indices}
        types = [self._entity_type(entity) for entity in intended_path]
        if 0 in fixed:
            starts = [fixed[0]]
        else:
            starts = sorted(
                name for name in self.entity_rows if self._entity_type(name) == types[0]
            )
        paths: list[list[str]] = []

        def visit(path: list[str]) -> None:
            if len(paths) > self.max_candidate_paths:
                return
            position = len(path)
            if position == len(types):
                paths.append(path.copy())
                return
            previous = path[-1]
            if position in fixed:
                candidates = [fixed[position]]
            else:
                candidates = sorted(self._neighbors(previous, directed))
            for candidate in candidates:
                if candidate in path:
                    continue
                if candidate not in self._neighbors(previous, directed):
                    continue
                if self._entity_type(candidate) != types[position]:
                    continue
                path.append(candidate)
                visit(path)
                path.pop()
                if len(paths) > self.max_candidate_paths:
                    return

        for start in starts:
            if fixed.get(0, start) != start:
                continue
            visit([start])
            if len(paths) > self.max_candidate_paths:
                break
        intended = list(intended_path)
        self.last_search_complete = len(paths) <= self.max_candidate_paths
        if intended not in paths:
            paths.insert(0, intended)
        return paths[: self.max_candidate_paths]

    def _path_record(
        self,
        path_id: int,
        path: Sequence[str],
        intended_path: Sequence[str],
        directed: bool,
        answer_index: int,
    ) -> dict[str, Any]:
        hops = []
        weight = 0.0
        for index, (source, target) in enumerate(zip(path, path[1:]), start=1):
            rows = self._edges_between(source, target, directed)
            descriptions = []
            hop_weight = 0.0
            for row in rows:
                descriptions.append(str(row.get("description", "")))
                value = row.get("weight", 0.0)
                if value is not None and not pd.isna(value):
                    hop_weight = max(hop_weight, float(value))
            weight += hop_weight
            hops.append(
                {
                    "hop": index,
                    "source": source,
                    "target": target,
                    "descriptions": descriptions,
                    "weight": hop_weight,
                }
            )
        matching_positions = sum(
            candidate == intended for candidate, intended in zip(path, intended_path)
        )
        return {
            "path_id": path_id,
            "entities": list(path),
            "answer": path[answer_index],
            "hops": hops,
            "total_weight": weight,
            "matching_intended_positions": matching_positions,
            "is_intended_path": list(path) == list(intended_path),
        }

    def _validation_prompt(
        self,
        question: str,
        intended_constraints: Sequence[dict[str, Any]],
        candidates: Sequence[dict[str, Any]],
    ) -> str:
        return f"""<instructions>
Determine which candidate graph paths fully satisfy the question and the semantic relation constraints of every intended hop. A path is admissible only if all of its hops express the same required relations as the corresponding intended hops and its final candidate answers the question. Do not accept paths based only on matching entity types. Return one result for every candidate. Return only JSON with this schema: {{"results": [{{"path_id": integer, "admissible": boolean, "answer": string, "hop_matches": [boolean], "reason": string}}]}}.
</instructions>
<question>{question}</question>
<intended_hop_constraints>
{json.dumps(list(intended_constraints), ensure_ascii=False, indent=2)}
</intended_hop_constraints>
<candidate_paths>
{json.dumps(list(candidates), ensure_ascii=False, indent=2)}
</candidate_paths>"""

    def _disambiguation_prompt(
        self,
        question: str,
        intended_answer: str,
        intended_path: Sequence[str],
        alternative_paths: Sequence[dict[str, Any]],
    ) -> str:
        return f"""<instructions>
Rewrite the question so the intended answer is uniquely selected while preserving the intended multi-hop chain. Add the smallest natural constraint supported by the intended path that excludes every alternative answer. Do not reveal hidden intermediate entities or the answer. Do not invent facts. Return only JSON with this schema: {{"question": string, "added_constraint": string, "reason": string}}.
</instructions>
<question>{question}</question>
<intended_answer>{intended_answer}</intended_answer>
<intended_path>
{json.dumps(list(intended_path), ensure_ascii=False)}
</intended_path>
<alternative_paths>
{json.dumps(list(alternative_paths), ensure_ascii=False, indent=2)}
</alternative_paths>"""

    def _intended_constraints(
        self, intended_path: Sequence[str], directed: bool
    ) -> list[dict[str, Any]]:
        constraints = []
        for index, (source, target) in enumerate(
            zip(intended_path, intended_path[1:]), start=1
        ):
            rows = self._edges_between(source, target, directed)
            if not rows:
                raise KeyError(f"No relationship found for intended hop {index}")
            constraints.append(
                {
                    "hop": index,
                    "source_type": self._entity_type(source),
                    "target_type": self._entity_type(target),
                    "descriptions": [str(row.get("description", "")) for row in rows],
                }
            )
        return constraints

    def _validate_candidates(
        self,
        question: str,
        intended_constraints: Sequence[dict[str, Any]],
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results_by_id: dict[int, dict[str, Any]] = {}
        for start in range(0, len(candidates), self.validation_batch_size):
            batch = candidates[start : start + self.validation_batch_size]
            response = self._query_json(
                self._validation_prompt(question, intended_constraints, batch)
            )
            results = response.get("results")
            if not isinstance(results, list):
                raise ValueError("Candidate validation did not return a results list")
            expected_ids = {candidate["path_id"] for candidate in batch}
            returned_ids = {
                result.get("path_id") for result in results if isinstance(result, dict)
            }
            if returned_ids != expected_ids:
                raise ValueError("Candidate validation omitted or added path IDs")
            for result in results:
                results_by_id[int(result["path_id"])] = result
        output = []
        for candidate in candidates:
            record = dict(candidate)
            record["validation"] = results_by_id[candidate["path_id"]]
            record["admissible"] = record["validation"].get("admissible") is True
            output.append(record)
        return output

    @staticmethod
    def _answer_groups(
        paths: Sequence[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in paths:
            if path["admissible"]:
                groups[str(path["answer"])].append(path)
        return dict(groups)

    @staticmethod
    def _distractors(
        paths: Sequence[dict[str, Any]], intended_answer: str, limit: int
    ) -> list[dict[str, Any]]:
        best_by_answer: dict[str, dict[str, Any]] = {}
        for path in paths:
            answer = str(path["answer"])
            if answer == intended_answer:
                continue
            hop_matches = path["validation"].get("hop_matches", [])
            semantic_matches = sum(value is True for value in hop_matches)
            score = (
                semantic_matches * 1000.0
                + path["matching_intended_positions"] * 100.0
                + math.log1p(max(0.0, path["total_weight"]))
                + (10000.0 if path["admissible"] else 0.0)
            )
            record = {
                "answer": answer,
                "score": score,
                "admissible_alternative": path["admissible"],
                "path": path["entities"],
                "hop_matches": hop_matches,
                "reason": path["validation"].get("reason", ""),
            }
            existing = best_by_answer.get(answer)
            if existing is None or record["score"] > existing["score"]:
                best_by_answer[answer] = record
        ranked = sorted(
            best_by_answer.values(),
            key=lambda item: (-item["score"], item["answer"]),
        )
        return ranked[:limit]

    def analyze_structure(
        self,
        question: str,
        structure: dict[str, Any],
        assignment: dict[str, str],
        intended_answer: str,
        policy: str = "discard",
    ) -> dict[str, Any]:
        if policy not in {"retain", "disambiguate", "discard"}:
            raise ValueError("invalid ambiguity policy")
        nodes = structure.get("topology", {}).get("nodes", [])
        roles = [node["role"] for node in nodes]
        types = {node["role"]: str(node.get("entity_type", "")) for node in nodes}
        open_roles = set(structure.get("open_entities", []))
        fixed = {role: assignment[role] for role in open_roles}
        edges = [
            (edge["source"], edge["target"])
            for edge in structure.get("topology", {}).get("edges", [])
        ]
        candidates = []

        def visit(values: dict[str, str]) -> None:
            if len(candidates) > self.max_candidate_paths:
                return
            if len(values) == len(roles):
                candidates.append(dict(values))
                return
            role = roles[len(values)]
            names = (
                [fixed[role]]
                if role in fixed
                else [
                    name
                    for name in self.entity_rows
                    if self._entity_type(name) == types[role]
                ]
            )
            for name in names:
                if name in values.values():
                    continue
                ok = True
                for source, target in edges:
                    if (
                        source == role
                        and target in values
                        and not self._edges_between(name, values[target], False)
                    ):
                        ok = False
                    if (
                        target == role
                        and source in values
                        and not self._edges_between(values[source], name, False)
                    ):
                        ok = False
                if ok:
                    values[role] = name
                    visit(values)
                    values.pop(role)

        visit({})
        search_complete = len(candidates) <= self.max_candidate_paths
        candidates = candidates[:self.max_candidate_paths]
        answer_role = structure.get("answer_entity")
        records = [
            {"id": index, "roles": item, "answer": item.get(answer_role)}
            for index, item in enumerate(candidates)
        ]
        result = self._query_json(
            f"""<instructions>Test every candidate topology assignment against the complete question, relation semantics, operations, and constraints. Return only JSON with schema {{"results":[{{"id":integer,"admissible":boolean,"answer":string,"reason":string}}]}}.</instructions><question>{question}</question><structure>{json.dumps(structure, ensure_ascii=False)}</structure><candidates>{json.dumps(records, ensure_ascii=False)}</candidates>"""
        )
        admissible = [
            item for item in result.get("results", []) if item.get("admissible") is True
        ]
        answers = sorted({str(item.get("answer")) for item in admissible})
        unique = search_complete and answers == [intended_answer]
        revised_question = None
        disambiguation = None
        if policy == "disambiguate" and not unique and intended_answer in answers:
            disambiguation = self._query_json(
                f"""<instructions>Rewrite the question with the smallest supported constraint that preserves all graph hops and uniquely selects the intended answer. Do not reveal hidden entities or the answer. Return only JSON with schema {{"question":string,"reason":string}}.</instructions><question>{question}</question><intended_answer>{intended_answer}</intended_answer><alternatives>{json.dumps(answers, ensure_ascii=False)}</alternatives><structure>{json.dumps(structure, ensure_ascii=False)}</structure>"""
            )
            revised_question = str(disambiguation.get("question", "")).strip() or None
            if revised_question:
                result = self._query_json(
                    f"""<instructions>Retest every candidate against the rewritten question. Return only JSON with schema {{"results":[{{"id":integer,"admissible":boolean,"answer":string,"reason":string}}]}}.</instructions><question>{revised_question}</question><structure>{json.dumps(structure, ensure_ascii=False)}</structure><candidates>{json.dumps(records, ensure_ascii=False)}</candidates>"""
                )
                admissible = [
                    item
                    for item in result.get("results", [])
                    if item.get("admissible") is True
                ]
                answers = sorted({str(item.get("answer")) for item in admissible})
                unique = search_complete and answers == [intended_answer]
        accepted = bool(admissible) if policy == "retain" else unique
        decision = (
            "disambiguated"
            if accepted and revised_question
            else ("accepted" if accepted else "discarded")
        )
        return {
            "accepted": accepted,
            "decision": decision,
            "policy": policy,
            "question": revised_question or question,
            "original_question": question,
            "intended_answer": intended_answer,
            "globally_unique_after_policy": unique,
            "candidate_search_complete": search_complete,
            "acceptable_answers": (
                answers
                if policy == "retain"
                else ([intended_answer] if accepted else [])
            ),
            "candidate_assignments": records,
            "validation": result,
            "disambiguation": disambiguation,
            "distractors": [
                item
                for item in result.get("results", [])
                if item.get("admissible") is not True
            ][:4],
        }

    def analyze(
        self,
        question: str,
        intended_path: Sequence[str],
        open_indices: Sequence[int] = (0,),
        answer_index: int = -1,
        directed: bool = False,
        policy: str = "discard",
        distractor_count: int = 4,
    ) -> dict[str, Any]:
        if policy not in {"retain", "disambiguate", "discard"}:
            raise ValueError("policy must be retain, disambiguate, or discard")
        if distractor_count < 0:
            raise ValueError("distractor_count must be >= 0")
        answer_index %= len(intended_path)
        intended_answer = intended_path[answer_index]
        raw_paths = self.enumerate_candidate_paths(
            intended_path, open_indices, directed
        )
        candidates = [
            self._path_record(index, path, intended_path, directed, answer_index)
            for index, path in enumerate(raw_paths)
        ]
        constraints = self._intended_constraints(intended_path, directed)
        validated = self._validate_candidates(question, constraints, candidates)
        answer_groups = self._answer_groups(validated)
        admissible_answers = sorted(answer_groups)
        intended_admissible = intended_answer in answer_groups
        search_complete = self.last_search_complete
        globally_unique = search_complete and intended_admissible and admissible_answers == [
            intended_answer
        ]
        distractors = self._distractors(validated, intended_answer, distractor_count)
        revised_question = None
        disambiguation = None
        final_paths = validated
        final_answer_groups = answer_groups
        final_answers = admissible_answers
        final_unique = globally_unique
        if not globally_unique and policy == "disambiguate" and intended_admissible:
            alternatives = [
                path
                for path in validated
                if path["admissible"] and path["answer"] != intended_answer
            ]
            disambiguation = self._query_json(
                self._disambiguation_prompt(
                    question, intended_answer, intended_path, alternatives
                )
            )
            revised_question = disambiguation.get("question")
            if not isinstance(revised_question, str) or not revised_question.strip():
                raise ValueError("Disambiguation did not return a revised question")
            revised_question = revised_question.strip()
            final_paths = self._validate_candidates(
                revised_question, constraints, candidates
            )
            final_answer_groups = self._answer_groups(final_paths)
            final_answers = sorted(final_answer_groups)
            final_unique = search_complete and intended_answer in final_answer_groups and final_answers == [
                intended_answer
            ]
        if policy == "retain":
            accepted = intended_admissible
            decision = (
                "retained_with_alternatives"
                if accepted and not globally_unique
                else "accepted" if accepted else "discarded"
            )
        elif policy == "disambiguate":
            accepted = final_unique
            decision = (
                "disambiguated"
                if accepted and revised_question
                else "accepted" if accepted else "discarded"
            )
        else:
            accepted = globally_unique
            decision = "accepted" if accepted else "discarded"
        return {
            "accepted": accepted,
            "decision": decision,
            "policy": policy,
            "question": revised_question or question,
            "original_question": question,
            "intended_path": list(intended_path),
            "intended_answer": intended_answer,
            "intended_answer_admissible": intended_admissible,
            "globally_unique_before_policy": globally_unique,
            "globally_unique_after_policy": final_unique,
            "admissible_answers": final_answers,
            "acceptable_answers": (
                final_answers
                if policy == "retain"
                else [intended_answer] if accepted else []
            ),
            "answer_paths": {
                answer: [path["entities"] for path in paths]
                for answer, paths in final_answer_groups.items()
            },
            "candidate_path_count": len(candidates),
            "candidate_limit_reached": not search_complete,
            "candidate_search_complete": search_complete,
            "candidate_paths": final_paths,
            "disambiguation": disambiguation,
            "distractors": distractors,
        }
