from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LexiconEntry:
    lemma: str
    lemma_normalized: str
    pos: str
    gloss: str
    example: str
    example_english: str
    source_url: str
    source_name: str = "wiktionary"
    flags: tuple[str, ...] = field(default_factory=tuple)


RAW_COLUMNS = (
    "Word",
    "Part of speech",
    "affixation",
    "meaning",
    "example",
    "English example",
    "source",
)

PROCESSED_COLUMNS = (
    "lemma",
    "lemma_normalized",
    "pos",
    "gloss",
    "example",
    "example_english",
    "source_url",
    "source_name",
    "flags",
)
