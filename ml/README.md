# doodle-ml

Offline ML pipeline for Doodle Live: download Quick, Draw! data, preprocess strokes,
train the stroke-sequence classifier, evaluate it and export it to ONNX for the browser.

```bash
uv sync
uv run python -m doodle_ml.download       # ~6,000 recognised drawings x 50 categories -> data/
uv run python -m doodle_ml.dataset --summary-only
uv run python -m doodle_ml.fixtures       # regenerate shared/fixtures/preprocess_cases.json
uv run python -m doodle_ml.cache          # simplify all drawings once + compute offset_scale
uv run python -m doodle_ml.train --quick  # few-minute sanity run
uv run python -m doodle_ml.train          # full run (configs/train.yaml)
uv run python -m doodle_ml.evaluate       # writes ../reports/
uv run tensorboard --logdir runs
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

## Preprocessing contract

`preprocess.py` and `web/src/ml/preprocess.ts` implement the same pipeline (normalise,
resample, Ramer-Douglas-Peucker, stroke-3, scale, pad). `doodle_ml.fixtures` writes 20
golden cases to `shared/fixtures/preprocess_cases.json`; pytest and vitest both assert
their output matches within 1e-5, and pytest fails if the file is stale.

## Model and training

A small pre-norm transformer encoder over stroke-3 sequences (4 layers, d_model 128,
~0.56M parameters) with masked mean pooling. Attention is written out explicitly so the
padding mask behaves identically in training, evaluation and ONNX export; a test checks
that padding never changes the output.

- **Prefix augmentation:** half of the training drawings are cut to their first k strokes
  and re-normalised, exactly like the live canvas normalises an unfinished drawing.
- **Geometric augmentation:** independent x/y scaling (0.9-1.1) and rotation (±10°).
- **CPU-friendly batching:** batches group drawings of similar length and are padded only
  to the longest one (median drawing: 38 points vs. max_len 200).
- **Reproducible:** augmentation randomness is derived from (seed, epoch, index), so results
  don't depend on the number of DataLoader workers.

`evaluate.py` runs partial drawings through the exact live preprocessing pipeline, so the
accuracy-vs-strokes curve reflects what the game sees.
