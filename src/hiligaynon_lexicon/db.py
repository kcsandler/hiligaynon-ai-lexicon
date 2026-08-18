from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from hiligaynon_lexicon.clean import normalize_lemma
from hiligaynon_lexicon.schema import LexiconEntry

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "lexicon.sqlite"
)
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "lexicon.csv"
)

SCHEMA_SQL = """
CREATE TABLE entries (
    id INTEGER PRIMARY KEY,
    lemma TEXT NOT NULL,
    lemma_normalized TEXT NOT NULL,
    pos TEXT NOT NULL,
    gloss TEXT NOT NULL,
    example TEXT NOT NULL DEFAULT '',
    example_english TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    source_name TEXT NOT NULL DEFAULT 'wiktionary',
    flags TEXT NOT NULL DEFAULT ''
);
CREATE INDEX idx_entries_lemma ON entries(lemma_normalized);
CREATE INDEX idx_entries_pos ON entries(pos);
CREATE VIRTUAL TABLE entries_fts USING fts5(
    lemma,
    lemma_normalized,
    gloss,
    content='entries',
    content_rowid='id',
    tokenize='unicode61'
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("DROP TABLE IF EXISTS entries_fts;")
    connection.executescript("DROP TABLE IF EXISTS entries;")
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def insert_entries(
    connection: sqlite3.Connection, entries: Iterable[LexiconEntry]
) -> int:
    rows = [
        (
            entry.lemma,
            entry.lemma_normalized,
            entry.pos,
            entry.gloss,
            entry.example,
            entry.example_english,
            entry.source_url,
            entry.source_name,
            "|".join(entry.flags),
        )
        for entry in entries
    ]
    connection.executemany(
        """
        INSERT INTO entries (
            lemma, lemma_normalized, pos, gloss, example, example_english,
            source_url, source_name, flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.execute(
        """
        INSERT INTO entries_fts(rowid, lemma, lemma_normalized, gloss)
        SELECT id, lemma, lemma_normalized, gloss FROM entries
        """
    )
    connection.commit()
    return len(rows)


def entry_from_row(row: sqlite3.Row) -> dict[str, Any]:
    flags = [flag for flag in (row["flags"] or "").split("|") if flag]
    return {
        "id": row["id"],
        "lemma": row["lemma"],
        "lemma_normalized": row["lemma_normalized"],
        "pos": row["pos"],
        "gloss": row["gloss"],
        "example": row["example"],
        "example_english": row["example_english"],
        "source_url": row["source_url"],
        "source_name": row["source_name"],
        "flags": flags,
    }


def lookup(connection: sqlite3.Connection, lemma: str) -> list[dict[str, Any]]:
    normalized = normalize_lemma(lemma)
    rows = connection.execute(
        """
        SELECT * FROM entries
        WHERE lemma_normalized = ?
        ORDER BY pos, lemma
        """,
        (normalized,),
    ).fetchall()
    return [entry_from_row(row) for row in rows]


def fts_query(raw: str) -> str:
    tokens = re.findall(r"[\w'-]+", raw, flags=re.UNICODE)
    if not tokens:
        return ""
    return " AND ".join(f"{token}*" for token in tokens)


def search(
    connection: sqlite3.Connection,
    query: str,
    pos: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    match_query = fts_query(query)
    if not match_query:
        return []

    limit = max(1, min(limit, 100))
    sql = """
        SELECT e.*
        FROM entries AS e
        JOIN entries_fts AS f ON e.id = f.rowid
        WHERE entries_fts MATCH ?
    """
    params: list[Any] = [match_query]
    if pos:
        sql += " AND e.pos = ?"
        params.append(pos.strip().casefold())
    sql += " ORDER BY bm25(entries_fts) LIMIT ?"
    params.append(limit)

    rows = connection.execute(sql, params).fetchall()
    return [entry_from_row(row) for row in rows]


def stats(connection: sqlite3.Connection) -> dict[str, Any]:
    total = connection.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
    unique = connection.execute(
        "SELECT COUNT(DISTINCT lemma_normalized) AS n FROM entries"
    ).fetchone()["n"]
    pos_rows = connection.execute(
        """
        SELECT pos, COUNT(*) AS n
        FROM entries
        GROUP BY pos
        ORDER BY n DESC, pos
        """
    ).fetchall()
    return {
        "entries": total,
        "unique_lemmas": unique,
        "pos_counts": {row["pos"]: row["n"] for row in pos_rows},
    }


def load_csv(csv_path: Path, db_path: Path) -> dict[str, Any]:
    import pandas as pd

    from hiligaynon_lexicon.schema import PROCESSED_COLUMNS

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    entries = [
        LexiconEntry(
            lemma=row["lemma"],
            lemma_normalized=row["lemma_normalized"],
            pos=row["pos"],
            gloss=row["gloss"],
            example=row["example"],
            example_english=row["example_english"],
            source_url=row["source_url"],
            source_name=row["source_name"] or "wiktionary",
            flags=tuple(flag for flag in row["flags"].split("|") if flag),
        )
        for row in frame.to_dict(orient="records")
        if all(column in row for column in PROCESSED_COLUMNS)
    ]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    connection = connect(db_path)
    try:
        init_schema(connection)
        inserted = insert_entries(connection, entries)
        summary = stats(connection)
        summary["inserted"] = inserted
        return summary
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a SQLite lexicon database from the processed CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    summary = load_csv(args.input, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
