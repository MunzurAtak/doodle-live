import json
from pathlib import Path

import numpy as np
import pytest
import torch

from doodle_ml.cache import simplify_set
from doodle_ml.export import ParityError, check_parity, run_export
from doodle_ml.model import ModelConfig, StrokeTransformer
from doodle_ml.preprocess import PreprocessConfig, preprocess
from doodle_ml.train import ModelSection, RunConfig, save_checkpoint
from helpers import SHAPES, synthetic_set


def test_check_parity_passes_and_fails() -> None:
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert check_parity(logits, logits + 1e-6, logits)["int8_top1_agreement"] == 1.0
    with pytest.raises(ParityError, match="fp32"):
        check_parity(logits, logits + 1e-2, logits)
    with pytest.raises(ParityError, match="int8"):
        check_parity(logits, logits, logits[:, ::-1])


@pytest.fixture
def exported(tmp_path: Path) -> dict[str, Path]:
    torch.manual_seed(0)
    config = ModelConfig(n_classes=3, max_len=200, d_model=32, n_layers=1, n_heads=2, d_ff=64)
    model = StrokeTransformer(config).eval()
    run = RunConfig(model=ModelSection(d_model=32, n_layers=1, n_heads=2, d_ff=64))
    preprocess_config = PreprocessConfig(offset_scale=40.0)
    ckpt = tmp_path / "best.pt"
    save_checkpoint(ckpt, model, run, preprocess_config, SHAPES, 1, {"val_top1": 0.5})

    processed = tmp_path / "processed"
    raw = synthetic_set(4, seed=3)
    raw.save(processed / "test.npz")
    simplify_set(raw, workers=1).save(processed / "simplified_test.npz")

    dirs = {name: tmp_path / name for name in ("models", "fixtures", "reports")}
    run_export(ckpt, processed, dirs["models"], dirs["fixtures"], dirs["reports"],
               parity_samples=12, min_int8_agreement=0.0)  # fmt: skip
    return {**dirs, "ckpt": ckpt}


def test_export_writes_model_meta_and_reports(exported: dict[str, Path]) -> None:
    assert (exported["models"] / "classifier.onnx").stat().st_size > 1000
    meta = json.loads((exported["models"] / "model_meta.json").read_text())
    assert meta["categories"] == SHAPES
    assert meta["max_len"] == 200 and meta["offset_scale"] == 40.0
    assert meta["input_names"] == ["strokes", "mask"] and meta["output_name"] == "logits"
    latency = json.loads((exported["reports"] / "latency.json").read_text())
    assert latency["int8"]["size_kb"] < latency["fp32"]["size_kb"]
    assert latency["parity"]["fp32_max_abs_diff"] <= 1e-4


def test_export_fixtures_match_the_onnx_model(exported: dict[str, Path]) -> None:
    import onnxruntime as ort

    pre = json.loads((exported["fixtures"] / "preprocess_cases.json").read_text())
    assert {c["config"]["offset_scale"] for c in pre["cases"]} == {40.0, 2.5}

    cases = json.loads((exported["fixtures"] / "model_cases.json").read_text())["cases"]
    assert len(cases) == 3  # one per class in the 3-class synthetic set
    session = ort.InferenceSession(str(exported["models"] / "classifier.onnx"))
    for case in cases:
        strokes, mask = preprocess(case["input"], PreprocessConfig(offset_scale=40.0))
        logits = session.run(["logits"], {"strokes": strokes[None], "mask": mask[None]})[0][0]
        np.testing.assert_allclose(logits, case["expected_logits"], atol=1e-4)
        assert case["expected_top3"][0] == SHAPES[int(np.argmax(logits))]
