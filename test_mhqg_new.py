from __future__ import annotations

import json

from mhqg_common import parse_json
from mhqg_evaluator import IndependentQuestionEvaluator


class FakeLlm:
    def __init__(self, values: list[dict]) -> None:
        self.values = iter(values)

    def query(self, prompt: str) -> str:
        return json.dumps(next(self.values))


def test_parse_json_fenced() -> None:
    assert parse_json("```json\n{\"x\": 1}\n```") == {"x": 1}


def test_evaluator_rejects_precheck() -> None:
    llm = FakeLlm([
        {"processable": False, "reason": "subjective", "evidence_units": [], "missing_evidence": []},
        {"valid_question": False, "multi_hop": False, "accepted": False, "supported_answer": None, "necessary_hops": [], "evidence_ids": [], "failure_reasons": ["unprocessable"], "confidence": 1.0},
    ])
    result = IndependentQuestionEvaluator(llm).evaluate("What is best?", "Nothing relevant")
    assert result["accepted"] is False
    assert result["stages"]["precheck_evidence_inventory"]["processable"] is False
