from pathlib import Path

from hiligaynon_lexicon.clean import clean_csv, clean_gloss, clean_records, normalize_pos

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.csv"


def load_fixture_rows() -> list[dict[str, str]]:
    import pandas as pd

    frame = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    return frame.to_dict(orient="records")


def test_normalize_pos_aliases() -> None:
    assert normalize_pos("phrasebook") == "phrase"
    assert normalize_pos("Adjective") == "adjective"
    assert normalize_pos("") == "unknown"


def test_clean_gloss_strips_synonyms_and_citations() -> None:
    gloss = clean_gloss("spouse\nSynonyms: asawa, abuyan")
    assert gloss == "spouse"

    cited = clean_gloss("A Spanish title\n1838, William Prescott, History of Ferdinand")
    assert cited == "A Spanish title"


def test_drops_category_iso_empty_and_duplicates() -> None:
    entries, stats = clean_records(load_fixture_rows())
    lemmas = {entry.lemma_normalized for entry in entries}

    assert "abaganhan" in lemmas
    assert "kag" in lemmas
    assert "phrase-item" in lemmas
    assert "abo" not in lemmas
    assert stats["dropped_category_index"] == 1
    assert stats["dropped_iso_homograph"] == 1
    assert stats["dropped_empty_gloss"] == 1
    assert stats["dropped_duplicate"] == 1
    assert stats["output_rows"] == 6


def test_flags_possible_inflections() -> None:
    entries, _stats = clean_records(load_fixture_rows())
    abierto = next(entry for entry in entries if entry.lemma == "abierto")
    assert "possible_non_hiligaynon_inflection" in abierto.flags


def test_clean_csv_writes_outputs(tmp_path: Path) -> None:
    stats = clean_csv(FIXTURE, tmp_path)
    assert (tmp_path / "lexicon.csv").exists()
    assert (tmp_path / "cleaning_stats.json").exists()
    assert stats["output_rows"] == 6
