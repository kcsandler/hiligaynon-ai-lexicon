from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hiligaynon_lexicon.db import DEFAULT_DB_PATH, connect, entry_from_row

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INDEX_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
)
Encoder = Callable[[list[str]], np.ndarray]


def embed_text(lemma: str, gloss: str) -> str:
    """English gloss carries most of the meaning for this multilingual MiniLM."""
    lemma = lemma.strip()
    gloss = gloss.strip()
    if lemma and gloss:
        return f"{lemma}: {gloss}"
    return lemma or gloss


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


_MODEL_CACHE: dict[str, Any] = {}


def encode_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = _MODEL_CACHE.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    return np.asarray(vectors, dtype=np.float32)


class SemanticIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (n_entries, dim)")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("metadata length must match embedding rows")
        self.embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self.metadata = metadata
        self.encoder = encoder
        self.model_name = model_name

    def search_vector(self, query_vector: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        k = max(1, min(k, len(self.metadata)))
        scores = self.embeddings @ query
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results = []
        for index in top:
            item = dict(self.metadata[int(index)])
            item["score"] = round(float(scores[int(index)]), 4)
            results.append(item)
        return results

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if self.encoder is None:
            vectors = encode_texts([query], model_name=self.model_name)
        else:
            vectors = l2_normalize(np.asarray(self.encoder([query]), dtype=np.float32))
        return self.search_vector(vectors[0], k=k)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "embeddings.npy", self.embeddings)
        (output_dir / "embedding_meta.json").write_text(
            json.dumps(
                {"model_name": self.model_name, "items": self.metadata},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_dir: Path,
        encoder: Encoder | None = None,
        load_encoder: bool = False,
    ) -> "SemanticIndex":
        embeddings = np.load(index_dir / "embeddings.npy")
        payload = json.loads((index_dir / "embedding_meta.json").read_text(encoding="utf-8"))
        model_name = payload.get("model_name", DEFAULT_MODEL_NAME)
        encoder_fn = encoder
        if encoder_fn is None and load_encoder:

            def encoder_fn(texts: list[str]) -> np.ndarray:
                return encode_texts(texts, model_name=model_name)

        return cls(
            embeddings=embeddings,
            metadata=payload["items"],
            encoder=encoder_fn,
            model_name=model_name,
        )


def build_index_from_db(
    db_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    connection = connect(db_path)
    try:
        rows = connection.execute(
            "SELECT id, lemma, lemma_normalized, pos, gloss FROM entries ORDER BY id"
        ).fetchall()
    finally:
        connection.close()

    texts = [embed_text(row["lemma"], row["gloss"]) for row in rows]
    embeddings = encode_texts(texts, model_name=model_name)
    metadata = [
        {
            "id": row["id"],
            "lemma": row["lemma"],
            "lemma_normalized": row["lemma_normalized"],
            "pos": row["pos"],
            "gloss": row["gloss"],
        }
        for row in rows
    ]
    index = SemanticIndex(embeddings, metadata, model_name=model_name)
    index.save(output_dir)
    return {
        "entries": len(metadata),
        "dimensions": int(embeddings.shape[1]),
        "model_name": model_name,
        "output_dir": str(output_dir),
    }


def hydrate_results(
    db_path: Path, hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not hits:
        return []
    ids = [int(hit["id"]) for hit in hits]
    placeholders = ",".join("?" for _ in ids)
    connection = connect(db_path)
    try:
        rows = connection.execute(
            f"SELECT * FROM entries WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        connection.close()
    by_id = {row["id"]: entry_from_row(row) for row in rows}
    hydrated = []
    for hit in hits:
        entry = dict(by_id.get(int(hit["id"]), {}))
        entry["score"] = hit.get("score")
        hydrated.append(entry)
    return hydrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MiniLM embeddings for semantic lexicon search."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    summary = build_index_from_db(args.db, args.output_dir, model_name=args.model)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
