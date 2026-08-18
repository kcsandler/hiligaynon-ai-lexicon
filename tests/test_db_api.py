from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hiligaynon_lexicon.api import create_app
from hiligaynon_lexicon.clean import clean_records
from hiligaynon_lexicon.db import connect, init_schema, insert_entries, lookup, search, stats

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.csv"


def fixture_entries():
    frame = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    entries, _stats = clean_records(frame.to_dict(orient="records"))
    return entries


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "lexicon.sqlite"
    connection = connect(path)
    init_schema(connection)
    insert_entries(connection, fixture_entries())
    connection.close()
    return path


def test_lookup_is_exact_and_case_insensitive(db_path: Path) -> None:
    connection = connect(db_path)
    try:
        rows = lookup(connection, "KAG")
        assert len(rows) == 1
        assert rows[0]["lemma"] == "kag"
        assert rows[0]["gloss"] == "and (coordinator)"
        assert lookup(connection, "missing-word") == []
    finally:
        connection.close()


def test_search_matches_gloss_and_can_filter_pos(db_path: Path) -> None:
    connection = connect(db_path)
    try:
        spouse = search(connection, "spouse")
        assert any(row["lemma"] == "aboy" for row in spouse)
        adjectives = search(connection, "broad", pos="adjective")
        assert any(row["lemma"] == "abaganhan" for row in adjectives)
        nouns = search(connection, "broad", pos="noun")
        assert nouns == []
    finally:
        connection.close()


def test_stats_counts_fixture_entries(db_path: Path) -> None:
    connection = connect(db_path)
    try:
        summary = stats(connection)
        assert summary["entries"] == 6
        assert summary["unique_lemmas"] == 6
    finally:
        connection.close()


@pytest.fixture
def client(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LEXICON_DB", str(db_path))
    return TestClient(create_app())


def test_api_lookup_and_missing_lemma(client: TestClient) -> None:
    found = client.get("/lookup", params={"lemma": "kag"})
    assert found.status_code == 200
    payload = found.json()
    assert payload["count"] == 1
    assert payload["entries"][0]["lemma"] == "kag"

    missing = client.get("/lookup", params={"lemma": "not-a-word"})
    assert missing.status_code == 404


def test_api_search_and_stats(client: TestClient) -> None:
    searched = client.get("/search", params={"q": "spouse"})
    assert searched.status_code == 200
    assert searched.json()["count"] >= 1

    summary = client.get("/stats")
    assert summary.status_code == 200
    assert summary.json()["entries"] == 6

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
