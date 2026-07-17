from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_api import LlmApi

from llm_openai import LlmOpenaiApi
from llm_gemini import LlmGeminiApi
from mhqg_evaluator import IndependentQuestionEvaluator
from mhqg_experiments import AblationRunner
from mhqg_indexing import CorpusGraphIndexer
from mhqg_system import GraphQuestionGenerator, QuestionStructureCatalogue


def model(name: str, provider: str = "openai") -> LlmApi:
    if provider == "openai":
        return LlmOpenaiApi(model=name)
    if provider == "gemini":
        return LlmGeminiApi(model=name)
    raise ValueError(f"Unknown provider: {provider}")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_output(value: Any, path: str | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        print(text)


def generator(args: argparse.Namespace) -> GraphQuestionGenerator:
    return GraphQuestionGenerator(
        model(args.generator_model, args.generator_provider),
        model(args.evaluator_model, args.evaluator_provider),
        args.graphrag_dir,
        args.catalogue,
        read_json(args.general_examples) if args.general_examples else (),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator-provider", choices=("openai", "gemini"), default="openai"
    )
    parser.add_argument("--generator-model", default="gpt-5")
    parser.add_argument("--index-model", default="gpt-4.1")
    parser.add_argument(
        "--evaluator-provider", choices=("openai", "gemini"), default="gemini"
    )
    parser.add_argument("--evaluator-model", default="gemini-2.5-flash")
    parser.add_argument("--general-examples")
    parser.add_argument("--graphrag-dir", default="./graphrag")
    parser.add_argument("--catalogue", default="./question_structure_catalogue.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("documents", nargs="+")
    index_parser.add_argument("--cache-dir", default="./.mhqg_index_cache")
    index_parser.add_argument("--chunk-tokens", type=int, default=1200)
    index_parser.add_argument("--overlap-tokens", type=int, default=100)
    index_parser.add_argument("--extraction-passes", type=int, default=2)
    index_parser.add_argument("--extract-claims", action="store_true")
    index_parser.add_argument("--entity-types", nargs="*")
    index_parser.add_argument("--output")
    examples_parser = subparsers.add_parser("build-examples")
    examples_parser.add_argument("--count-per-structure", type=int, default=1)
    examples_parser.add_argument("--seed-examples")
    examples_parser.add_argument("--min-size", type=int, default=3)
    examples_parser.add_argument("--max-size", type=int, default=5)
    examples_parser.add_argument("--output")
    approve_structure_parser = subparsers.add_parser("approve-structure")
    approve_structure_parser.add_argument("structure_id")
    approve_structure_parser.add_argument("--replacement")
    approve_structure_parser.add_argument("--output")
    approve_example_parser = subparsers.add_parser("approve-example")
    approve_example_parser.add_argument("structure_id")
    approve_example_parser.add_argument("example_index", type=int)
    approve_example_parser.add_argument("--replacement")
    approve_example_parser.add_argument("--output")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--count", type=int, default=10)
    generate_parser.add_argument(
        "--policy",
        choices=("proportional", "common", "rare", "uniform"),
        default="proportional",
    )
    generate_parser.add_argument("--min-size", type=int, default=3)
    generate_parser.add_argument("--max-size", type=int, default=5)
    generate_parser.add_argument(
        "--ambiguity-policy",
        choices=("discard", "retain", "disambiguate"),
        default="discard",
    )
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument(
        "--review-rigor", choices=("strict", "balanced", "loose"), default="strict"
    )
    generate_parser.add_argument("--max-attempts", type=int)
    generate_parser.add_argument("--example-count", type=int, default=3)
    generate_parser.add_argument("--max-restarts", type=int, default=2)
    generate_parser.add_argument("--output")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("input")
    evaluate_parser.add_argument("--output")
    ablation_parser = subparsers.add_parser("ablate")
    ablation_parser.add_argument(
        "mode",
        choices=("baseline", "no_graph", "no_composition", "no_self_review", "full"),
    )
    ablation_parser.add_argument("--count", type=int, default=10)
    ablation_parser.add_argument("--contexts")
    ablation_parser.add_argument(
        "--policy",
        choices=("proportional", "common", "rare", "uniform"),
        default="proportional",
    )
    ablation_parser.add_argument("--min-size", type=int, default=3)
    ablation_parser.add_argument("--max-size", type=int, default=5)
    ablation_parser.add_argument(
        "--ambiguity-policy",
        choices=("discard", "retain", "disambiguate"),
        default="discard",
    )
    ablation_parser.add_argument("--seed", type=int)
    ablation_parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "index":
        kwargs = {
            "output_dir": args.graphrag_dir,
            "cache_dir": args.cache_dir,
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "extraction_passes": args.extraction_passes,
            "extract_claims": args.extract_claims,
        }
        if args.entity_types:
            kwargs["entity_types"] = args.entity_types
        result = CorpusGraphIndexer(
            model(args.index_model, args.generator_provider), **kwargs
        ).index(args.documents)
        write_output(result, args.output)
    elif args.command == "build-examples":
        seeds = read_json(args.seed_examples) if args.seed_examples else ()
        result = generator(args).build_structure_examples(
            args.count_per_structure, seeds, args.min_size, args.max_size
        )
        write_output(result, args.output)
    elif args.command == "approve-structure":
        catalogue = QuestionStructureCatalogue(args.catalogue)
        replacement = read_json(args.replacement) if args.replacement else None
        write_output(catalogue.approve_structure(args.structure_id, replacement), args.output)
    elif args.command == "approve-example":
        catalogue = QuestionStructureCatalogue(args.catalogue)
        replacement = read_json(args.replacement) if args.replacement else None
        write_output(catalogue.approve_example(args.structure_id, args.example_index, replacement), args.output)
    elif args.command == "generate":
        result = generator(args).generate(
            args.count,
            args.policy,
            args.min_size,
            args.max_size,
            args.ambiguity_policy,
            True,
            args.seed,
            args.review_rigor,
            args.max_attempts,
            True,
            args.example_count,
            args.max_restarts,
        )
        write_output(result, args.output)
    elif args.command == "evaluate":
        records = read_json(args.input)
        if isinstance(records, dict):
            records = [records]
        evaluator = IndependentQuestionEvaluator(
            model(args.evaluator_model, args.evaluator_provider)
        )
        result = [
            {
                **record,
                "evaluation": evaluator.evaluate(record["question"], record["context"]),
            }
            for record in records
        ]
        write_output(result, args.output)
    else:
        contexts = read_json(args.contexts) if args.contexts else []
        if contexts and isinstance(contexts[0], dict):
            contexts = [item["context"] for item in contexts]
        graph_generator = generator(args)
        runner = AblationRunner(
            graph_generator.llm, graph_generator.evaluator_llm, graph_generator
        )
        result = runner.run(
            args.mode,
            args.count,
            contexts,
            policy=args.policy,
            min_size=args.min_size,
            max_size=args.max_size,
            ambiguity_policy=args.ambiguity_policy,
            seed=args.seed,
        )
        write_output(result, args.output)


if __name__ == "__main__":
    main()
