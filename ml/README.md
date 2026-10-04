# doodle-ml

Offline ML pipeline for Doodle Live: download Quick, Draw! data, preprocess strokes,
train the stroke-sequence classifier, evaluate it and export it to ONNX for the browser.

```bash
uv sync
uv run python -m doodle_ml.download       # ~6,000 recognised drawings x 50 categories -> data/
uv run python -m doodle_ml.dataset --summary-only
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Data

`doodle_ml.download` streams each category's *simplified* ndjson file from the public
Quick, Draw! bucket, keeps only `recognized` drawings, stops after a pool of 20,000 and
takes a seeded sample of 6,000. It is resumable per category: re-run it after a failure
and complete categories are skipped (files from a smaller
test run are downloaded again). Afterwards `doodle_ml.dataset` builds deterministic
5,000 / 500 / 500 train / val / test splits per category in `data/processed/` as
compressed `.npz` files (ragged strokes stored as flat arrays).

Quick sanity run: `uv run python -m doodle_ml.download --limit-categories 3 --per-class 1200`.

Data: [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) by Google, CC BY 4.0.
