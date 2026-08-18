# Hiligaynon AI Lexicon

Python toolkit for a **Hiligaynon (Ilonggo)** lexical database: cleaning noisy dictionary extracts, storing a searchable lexicon, and (next) semantic search over glosses.

This is not a generic chatbot. The first milestone is **data engineering for a low-resource language**.

## Status

Implemented:

- Wiktionary extract ingestion
- Cleaning, POS normalization, gloss cleanup, deduplication
- Documented source licenses and dataset limits

Latest cleaning run on the bundled extract:

| Metric | Count |
|---|---|
| Input rows | 1,987 |
| Kept rows | 1,933 |
| Unique lemmas | 1,673 |
| Dropped ISO homographs | 42 |
| Dropped duplicates | 8 |
| Dropped empty glosses | 4 |

POS after cleaning: noun 1,162 · verb 374 · adjective 316 · phrase 54 · determiner 15 · pronoun 12.

Not in this milestone yet:

- SQLite + FastAPI
- Embeddings / FAISS semantic search
- Wikipedia frequency counts
- Affix heuristics

## Overview

Hiligaynon has far less NLP tooling than English or Filipino. Dictionary pages exist, but scraped entries mix real lemmas with Wiktionary homographs (ISO language codes), category index rows, and noisy glosses.

This project turns a CC BY-SA Wiktionary extract into a reproducible, inspectable lexicon.

## Features

- Drop category index rows and ISO-code homograph pages
- Normalize part-of-speech labels
- Clean glosses (collapse whitespace, cut synonym/citation tails)
- Deduplicate `(lemma, POS, gloss)`
- Write processed CSV plus a JSON stats report

## Data sources

Public v1 uses **English Wiktionary Hiligaynon lemmas only**.

| Source | License | Used in public repo |
|---|---|---|
| [English Wiktionary](https://en.wiktionary.org/) Hiligaynon entries | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Yes |
| Motus / Kaufmann dictionary PDFs | Unclear / likely copyrighted | **No** |

See [ATTRIBUTION.md](ATTRIBUTION.md).

## Limitations

- Coverage is incomplete (about 1.7k unique lemmas after cleaning).
- POS tags come from Wiktionary category membership and are sometimes wrong (for example a noun gloss tagged as adjective).
- Homograph pages for ISO 639-3 codes are dropped; a later pass may recover real Hiligaynon senses of those spellings from other rows.
- No morphological analyzer yet. Affix fields in the scrape are mostly empty.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
python -m hiligaynon_lexicon.clean --input data/raw/wiktionary_hiligaynon.csv --output-dir data/processed
```

```bash
pytest
```

## Project structure

```text
src/hiligaynon_lexicon/   cleaning and (later) API code
data/raw/                 Wiktionary extract
data/processed/           cleaned lexicon + stats
tests/                    unit tests
```

## License

- Code: [MIT](LICENSE)
- Lexical data derived from Wiktionary: CC BY-SA 4.0 (see ATTRIBUTION.md)
