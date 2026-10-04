# doodle-ml

Offline ML pipeline for Doodle Live: download Quick, Draw! data, preprocess strokes,
train the stroke-sequence classifier, evaluate it and export it to ONNX for the browser.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
