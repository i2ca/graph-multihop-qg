from __future__ import annotations

import json
import re
from typing import Any

from llm_api import LlmApi


def parse_json(value: str) -> dict[str, Any]:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The LLM did not return a JSON object")
        result = json.loads(value[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("The LLM response must be a JSON object")
    return result


def query_json(llm: LlmApi, prompt: str) -> dict[str, Any]:
    return parse_json(llm.query(prompt))


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
