from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from hiligaynon_lexicon.db import (
    DEFAULT_DB_PATH,
    connect,
    lookup as lookup_entries,
    search as search_entries,
    stats as lexicon_stats,
)
from hiligaynon_lexicon.embed import DEFAULT_INDEX_DIR, SemanticIndex, hydrate_results


def database_path() -> Path:
    override = os.environ.get("LEXICON_DB")
    return Path(override) if override else DEFAULT_DB_PATH


def index_dir() -> Path:
    override = os.environ.get("LEXICON_INDEX_DIR")
    return Path(override) if override else DEFAULT_INDEX_DIR


_semantic_index: SemanticIndex | None = None


def get_semantic_index() -> SemanticIndex:
    global _semantic_index
    if _semantic_index is None:
        path = index_dir()
        if not (path / "embeddings.npy").exists():
            raise HTTPException(
                status_code=503,
                detail=f"Semantic index is missing at {path}",
            )
        _semantic_index = SemanticIndex.load(path, load_encoder=True)
    return _semantic_index


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hiligaynon AI Lexicon",
        description=(
            "Lookup, full-text search, and semantic search over a cleaned "
            "Hiligaynon lexicon derived from Wiktionary (CC BY-SA 4.0)."
        ),
        version="0.1.0",
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> dict[str, str]:
        path = database_path()
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Lexicon database is missing at {path}",
            )
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        path = database_path()
        if not path.exists():
            raise HTTPException(status_code=503, detail="Lexicon database is missing.")
        with closing(connect(path)) as connection:
            return lexicon_stats(connection)

    @app.get("/lookup")
    def lookup(
        lemma: str = Query(..., min_length=1, description="Exact Hiligaynon lemma"),
    ) -> dict[str, Any]:
        path = database_path()
        if not path.exists():
            raise HTTPException(status_code=503, detail="Lexicon database is missing.")
        with closing(connect(path)) as connection:
            entries = lookup_entries(connection, lemma)
        if not entries:
            raise HTTPException(status_code=404, detail=f"No entries for {lemma!r}.")
        return {"query": lemma, "count": len(entries), "entries": entries}

    @app.get("/search")
    def search(
        q: str = Query(..., min_length=1, description="Lemma or English gloss"),
        pos: str | None = Query(default=None, description="Optional POS filter"),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        path = database_path()
        if not path.exists():
            raise HTTPException(status_code=503, detail="Lexicon database is missing.")
        with closing(connect(path)) as connection:
            entries = search_entries(connection, q, pos=pos, limit=limit)
        return {"query": q, "pos": pos, "count": len(entries), "entries": entries}

    @app.get("/semantic-search")
    def semantic_search(
        q: str = Query(..., min_length=1, description="Natural-language meaning query"),
        k: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        index = get_semantic_index()
        hits = index.search(q, k=k)
        path = database_path()
        entries = hydrate_results(path, hits) if path.exists() else hits
        return {
            "query": q,
            "count": len(entries),
            "model": index.model_name,
            "entries": entries,
        }

    return app


app = create_app()
