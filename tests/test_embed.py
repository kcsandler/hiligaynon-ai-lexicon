import json
from pathlib import Path

import numpy as np

from hiligaynon_lexicon.embed import SemanticIndex, embed_text
from hiligaynon_lexicon.metrics import (
    first_relevant_rank,
    mean_reciprocal_rank,
    recall_at_k,
    summarize_retrieval,
)


def test_embed_text_joins_lemma_and_gloss() -> None:
    assert embed_text("abayan", "(anatomy) waist") == "abayan: (anatomy) waist"
    assert embed_text("kag", "") == "kag"


def test_search_vector_ranks_cosine_neighbors() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"id": 1, "lemma": "a", "lemma_normalized": "a", "pos": "noun", "gloss": "one"},
        {"id": 2, "lemma": "b", "lemma_normalized": "b", "pos": "noun", "gloss": "two"},
        {"id": 3, "lemma": "c", "lemma_normalized": "c", "pos": "noun", "gloss": "three"},
    ]
    index = SemanticIndex(embeddings, metadata, model_name="test")
    hits = index.search_vector(np.array([1.0, 0.1, 0.0], dtype=np.float32), k=2)
    assert [hit["lemma"] for hit in hits] == ["a", "c"]
    assert hits[0]["score"] >= hits[1]["score"]


def test_index_roundtrip(tmp_path: Path) -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = [
        {"id": 1, "lemma": "x", "lemma_normalized": "x", "pos": "noun", "gloss": "ex"},
        {"id": 2, "lemma": "y", "lemma_normalized": "y", "pos": "noun", "gloss": "why"},
    ]
    SemanticIndex(embeddings, metadata, model_name="test").save(tmp_path)
    loaded = SemanticIndex.load(tmp_path)
    assert loaded.metadata[0]["lemma"] == "x"
    assert loaded.embeddings.shape == (2, 2)
    payload = json.loads((tmp_path / "embedding_meta.json").read_text(encoding="utf-8"))
    assert payload["model_name"] == "test"


def test_retrieval_metrics() -> None:
    assert recall_at_k(["abayan", "abang"], {"abayan"}, k=1) == 1.0
    assert recall_at_k(["abang", "abayan"], {"abayan"}, k=1) == 0.0
    assert recall_at_k(["abang", "abayan"], {"abayan"}, k=2) == 1.0
    assert first_relevant_rank(["abang", "abayan"], {"abayan"}) == 2
    assert mean_reciprocal_rank([1, 2, None]) == (1 + 0.5 + 0) / 3
    summary = summarize_retrieval(
        [
            {"rank": 1},
            {"rank": 2},
            {"rank": None},
        ]
    )
    assert summary["queries"] == 3
    assert summary["recall_at_1"] == 0.3333
    assert summary["recall_at_5"] == 0.6667
