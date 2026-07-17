from __future__ import annotations

from typing import Any, Sequence

from llm_api import LlmApi
from mhqg_common import query_json
from mhqg_evaluator import IndependentQuestionEvaluator
from mhqg_system import GraphQuestionGenerator


class AblationRunner:
    def __init__(self, generator_llm: LlmApi, evaluator_llm: LlmApi, generator: GraphQuestionGenerator) -> None:
        self.generator_llm = generator_llm
        self.evaluator = IndependentQuestionEvaluator(evaluator_llm)
        self.generator = generator

    def _direct(self, context: str) -> dict[str, Any]:
        return query_json(self.generator_llm, f"""<instructions>Create one unique, context-grounded open-ended multi-hop question. Return only JSON with schema {{"question":string,"answer":string}}.</instructions><context>{context}</context>""")

    def _no_graph(self, context: str) -> dict[str, Any]:
        singles = query_json(self.generator_llm, f"""<instructions>Create a connected sequence of at least two factual single-hop questions and answers from the context where each answer feeds the next question. Return only JSON with schema {{"hops":[{{"question":string,"answer":string}}]}}.</instructions><context>{context}</context>""")
        return query_json(self.generator_llm, f"""<instructions>Compose all hops into one fluent open-ended multi-hop question without revealing intermediate answers. Return only JSON with schema {{"question":string,"answer":string}}.</instructions><hops>{singles}</hops><context>{context}</context>""")

    def run(self, mode: str, count: int, contexts: Sequence[str] = (), **generation_options: Any) -> list[dict[str, Any]]:
        if mode not in {"baseline", "no_graph", "no_composition", "no_self_review", "full"}:
            raise ValueError("Unknown ablation mode")
        generated = []
        if mode in {"baseline", "no_graph"}:
            if not contexts:
                raise ValueError("Contexts are required")
            for context in contexts[:count]:
                item = self._direct(context) if mode == "baseline" else self._no_graph(context)
                generated.append({**item, "context": context})
        elif mode == "no_composition":
            items = self.generator.generate(count, self_review=True, **generation_options)
            for item in items:
                direct = self._direct(item["context"])
                generated.append({**item, **direct})
        else:
            generated = self.generator.generate(count, self_review=mode == "full", **generation_options)
        for item in generated:
            item["independent_evaluation"] = self.evaluator.evaluate(item["question"], item["context"])
        return generated
