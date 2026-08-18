from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from hiligaynon_lexicon.schema import PROCESSED_COLUMNS, LexiconEntry

ISO_GLOSS_RE = re.compile(r"ISO 639-3 language code", re.IGNORECASE)
CITATION_LINE_RE = re.compile(r"^\d{4},")
SYNONYM_LINE_RE = re.compile(r"^synonym", re.IGNORECASE)

POS_ALIASES = {
    "phrasebook": "phrase",
    "nouns": "noun",
    "verbs": "verb",
    "adjectives": "adjective",
    "determiners": "determiner",
    "pronouns": "pronoun",
}


def normalize_lemma(word: str) -> str:
    stripped = unicodedata.normalize("NFC", (word or "").strip())
    return stripped.casefold()


def normalize_pos(pos: str) -> str:
    label = unicodedata.normalize("NFC", (pos or "").strip().casefold())
    if not label:
        return "unknown"
    return POS_ALIASES.get(label, label)


def clean_gloss(raw: str) -> str:
    if not raw:
        return ""

    lines: list[str] = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        text = " ".join(line.split()).strip()
        if not text:
            continue
        if SYNONYM_LINE_RE.match(text) or CITATION_LINE_RE.match(text):
            break
        lines.append(text)

    return " ".join(lines).strip()


def drop_reason(word: str, gloss: str) -> str | None:
    lemma = (word or "").strip()
    if not lemma:
        return "empty_lemma"
    if lemma.startswith("Category:"):
        return "category_index"
    if ISO_GLOSS_RE.search(gloss):
        return "iso_homograph"
    if not gloss:
        return "empty_gloss"
    return None


def row_to_entry(row: dict[str, str]) -> tuple[LexiconEntry | None, str | None]:
    word = str(row.get("Word") or "")
    gloss = clean_gloss(str(row.get("meaning") or ""))
    reason = drop_reason(word, gloss)
    if reason:
        return None, reason

    lemma = word.strip()
    flags: list[str] = []
    if "inflection of" in gloss.lower() or gloss.lower().startswith("neuter of"):
        flags.append("possible_non_hiligaynon_inflection")

    entry = LexiconEntry(
        lemma=lemma,
        lemma_normalized=normalize_lemma(lemma),
        pos=normalize_pos(str(row.get("Part of speech") or "")),
        gloss=gloss,
        example=str(row.get("example") or "").strip(),
        example_english=str(row.get("English example") or "").strip(),
        source_url=str(row.get("source") or "").strip(),
        flags=tuple(flags),
    )
    return entry, None


def clean_records(rows: list[dict[str, str]]) -> tuple[list[LexiconEntry], dict[str, Any]]:
    dropped: Counter[str] = Counter()
    kept: list[LexiconEntry] = []
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        entry, reason = row_to_entry(row)
        if reason:
            dropped[reason] += 1
            continue

        key = (entry.lemma_normalized, entry.pos, entry.gloss.casefold())
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(entry)

    stats = {
        "input_rows": len(rows),
        "output_rows": len(kept),
        "unique_lemmas": len({entry.lemma_normalized for entry in kept}),
        **{f"dropped_{name}": count for name, count in sorted(dropped.items())},
    }
    stats["pos_counts"] = dict(Counter(entry.pos for entry in kept))
    return kept, stats


def entries_to_frame(entries: list[LexiconEntry]) -> pd.DataFrame:
    records = []
    for entry in entries:
        records.append(
            {
                "lemma": entry.lemma,
                "lemma_normalized": entry.lemma_normalized,
                "pos": entry.pos,
                "gloss": entry.gloss,
                "example": entry.example,
                "example_english": entry.example_english,
                "source_url": entry.source_url,
                "source_name": entry.source_name,
                "flags": "|".join(entry.flags),
            }
        )
    return pd.DataFrame(records, columns=list(PROCESSED_COLUMNS))


def clean_csv(input_path: Path, output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    rows = frame.to_dict(orient="records")
    entries, stats = clean_records(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    entries_to_frame(entries).to_csv(output_dir / "lexicon.csv", index=False)
    (output_dir / "cleaning_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a Wiktionary Hiligaynon lexicon CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stats = clean_csv(args.input, args.output_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
