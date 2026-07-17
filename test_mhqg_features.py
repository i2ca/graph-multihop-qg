from pathlib import Path

from mhqg_system import GraphQuestionGenerator, QuestionStructureCatalogue
from mhqg_indexing import CorpusGraphIndexer
from structure_frequency_sampling import StructureFrequencySampler


def test_catalogue_persists_generated_examples(tmp_path: Path) -> None:
    path = tmp_path / "catalogue.json"
    catalogue = QuestionStructureCatalogue(path)
    catalogue.put({"structure_id": "s", "examples": []})
    catalogue.add_example("s", {"multi_hop_question": "Q?"})
    catalogue.save()
    loaded = QuestionStructureCatalogue(path)
    assert loaded.get("s")["examples"][0]["multi_hop_question"] == "Q?"


def test_resolved_hops_preserve_branches_and_operations() -> None:
    structure = {
        "hop_order": [
            {"hop": 1, "source": "a", "target": "c"},
            {"hop": 2, "source": "b", "target": "c"},
            {
                "hop": 3,
                "operation": "argmax",
                "inputs": ["a", "b"],
                "returns": ["c"],
            },
        ]
    }
    assignment = {"a": "A", "b": "B", "c": "C"}
    resolved = GraphQuestionGenerator._resolved_hops(structure, assignment)
    assert resolved[0]["source"] == "A"
    assert resolved[1]["source"] == "B"
    assert resolved[2]["inputs"] == ["A", "B"]
    assert resolved[2]["returns"] == ["C"]


def test_operation_hop_detection() -> None:
    structure = {"hop_order": [{"source": "a", "target": "b"}, {"operation": "count"}]}


def test_catalogue_requires_explicit_approval(tmp_path: Path) -> None:
    path = tmp_path / "catalogue.json"
    catalogue = QuestionStructureCatalogue(path)
    catalogue.put({"structure_id": "s", "examples": []})
    catalogue.add_example("s", {"multi_hop_question": "Draft?"})
    assert catalogue.approved_structure("s") is None
    assert catalogue.approved_examples("s") == []
    catalogue.approve_structure("s")
    catalogue.approve_example("s", 0, {"multi_hop_question": "Reviewed?"})
    assert catalogue.approved_structure("s") is not None
    assert catalogue.approved_examples("s")[0]["multi_hop_question"] == "Reviewed?"


def test_representative_instances_prioritize_evidence_and_diversity() -> None:
    weak = {"entity_names": ["A", "B"], "entities": [{"name": "A"}, {"name": "B"}], "relationships": [{"description": "related", "weight": 0.1, "text_unit_ids": ["1"]}]}
    strong = {"entity_names": ["C", "D"], "entities": [{"name": "C"}, {"name": "D"}], "relationships": [{"description": "strong relation", "weight": 1.0, "text_unit_ids": ["2", "3"]}]}
    selected = GraphQuestionGenerator._representative_instances([weak, strong], 1)
    assert selected == [strong]


def test_best_example_uses_relation_similarity() -> None:
    examples = [
        {"multi_hop_question": "Q1", "relation_descriptions": ["was born in a city"]},
        {"multi_hop_question": "Q2", "relation_descriptions": ["directed a film"]},
    ]
    subgraph = {"relationships": [{"description": "film directed by person"}]}
    assert GraphQuestionGenerator._best_example(examples, subgraph)["multi_hop_question"] == "Q2"


def test_indexer_uses_paper_chunk_defaults() -> None:
    indexer = CorpusGraphIndexer(object())
    assert indexer.chunk_tokens == 1200
    assert indexer.overlap_tokens == 100


def test_indexer_chunks_with_configured_tokenizer() -> None:
    indexer = CorpusGraphIndexer(object(), chunk_tokens=3, overlap_tokens=1, token_encoder=list, token_decoder=lambda values: "".join(values))
    assert indexer._chunks("abcdef") == ["abc", "cde", "ef"]


def test_indexer_disables_claims_by_default() -> None:
    indexer = CorpusGraphIndexer(object())
    assert indexer.extract_claims is False
    assert "Return an empty claims array" in indexer._prompt("source", 1)


def test_best_examples_returns_requested_n_shots() -> None:
    examples = [
        {"multi_hop_question": "Q1", "relation_descriptions": ["born in city"]},
        {"multi_hop_question": "Q2", "relation_descriptions": ["city birthplace"]},
        {"multi_hop_question": "Q3", "relation_descriptions": ["directed film"]},
    ]
    subgraph = {"relationships": [{"description": "birth city"}]}
    selected = GraphQuestionGenerator._best_examples(examples, subgraph, 2)
    assert len(selected) == 2
    assert {item["multi_hop_question"] for item in selected} == {"Q1", "Q2"}
