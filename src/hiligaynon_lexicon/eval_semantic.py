from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hiligaynon_lexicon.embed import DEFAULT_INDEX_DIR, SemanticIndex
from hiligaynon_lexicon.metrics import first_relevant_rank, summarize_retrieval

DEFAULT_QUERIES = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "semantic_queries.json"
)


def evaluate(
    queries: list[dict[str, Any]],
    index: SemanticIndex,
    k: int = 10,
) -> dict[str, Any]:
    per_query = []
    for item in queries:
        hits = index.search(item["query"], k=k)
        predicted = [hit["lemma_normalized"] for hit in hits]
        relevant = {lemma.casefold() for lemma in item["relevant_lemmas"]}
        rank = first_relevant_rank(predicted, relevant)
        per_query.append(
            {
                "id": item["id"],
                "query": item["query"],
                "relevant_lemmas": sorted(relevant),
                "rank": rank,
                "top_lemmas": predicted[:5],
            }
        )
    return {
        "notes": (
            "Hand-labeled English paraphrase queries against Wiktionary glosses. "
            "This is not a published benchmark."
        ),
        "metrics": summarize_retrieval(per_query),
        "queries": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic search with a small labeled query set."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDEX_DIR / "semantic_eval.json",
    )
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    index = SemanticIndex.load(args.index_dir, load_encoder=True)
    report = evaluate(queries, index, k=args.k)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
