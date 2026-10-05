"""Export the trained classifier for the browser (SPEC §6).

1. ONNX export (opset 17) with a fixed input shape: ``strokes [1, 200, 3]``,
   ``mask [1, 200]`` -> ``logits [1, n_classes]``.
2. Dynamic int8 quantization of the weights (onnxruntime).
3. Parity checks, failing loudly: PyTorch vs fp32 ONNX (max |diff| <= 1e-4) and
   PyTorch vs int8 ONNX (top-1 agreement >= 98% on 1,000 test drawings).
4. Writes ``web/public/models/classifier.onnx`` (int8) and ``model_meta.json``.
5. Regenerates ``shared/fixtures/preprocess_cases.json`` with the real ``offset_scale``
   and writes ``shared/fixtures/model_cases.json`` (raw drawings + expected int8 output).
6. Writes ``reports/latency.json`` (single-threaded CPU inference, fp32 vs int8).

Usage (from ``ml/``)::

    uv run python -m doodle_ml.export
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from doodle_ml.cache import load_simplified
from doodle_ml.config import ML_ROOT, PROCESSED_DIR
from doodle_ml.dataset import DrawingSet
from doodle_ml.evaluate import REPORTS_DIR
from doodle_ml.fixtures import FIXTURES_DIR, write_preprocess_fixture
from doodle_ml.model import StrokeTransformer
from doodle_ml.preprocess import PreprocessConfig, encode, preprocess
from doodle_ml.train import CHECKPOINT_DIR, load_model

WEB_MODELS_DIR = ML_ROOT.parent / "web" / "public" / "models"
INPUT_NAMES = ["strokes", "mask"]
OUTPUT_NAME = "logits"
META_VERSION = 1

FP32_TOLERANCE = 1e-4
MIN_INT8_AGREEMENT = 0.98


class ParityError(RuntimeError):
    pass


def export_onnx(model: StrokeTransformer, path: Path) -> None:
    max_len = model.config.max_len
    strokes = torch.zeros(1, max_len, 3)
    mask = torch.zeros(1, max_len)
    mask[0, :8] = 1.0
    program = torch.onnx.export(
        model.eval(),
        (strokes, mask),
        input_names=INPUT_NAMES,
        output_names=[OUTPUT_NAME],
        opset_version=17,
        dynamo=True,
        verbose=False,
    )
    program.save(str(path))


def quantize(fp32_path: Path, int8_path: Path) -> None:
    # The torch.export-based exporter stores intermediate shape annotations that ONNX
    # shape inference (run by the quantizer) disagrees with; drop them and let it re-infer.
    model = onnx.load(str(fp32_path))
    del model.graph.value_info[:]
    clean_path = fp32_path.with_name(fp32_path.stem + "_clean.onnx")
    onnx.save(model, str(clean_path))
    quantize_dynamic(str(clean_path), str(int8_path), weight_type=QuantType.QInt8)


def make_session(path: Path) -> ort.InferenceSession:
    """Single-threaded CPU session, like onnxruntime-web on GitHub Pages."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def encode_set(drawings: DrawingSet, config: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-length (max_len) inputs for every drawing of an already simplified set."""
    strokes = np.zeros((len(drawings), config.max_len, 3), dtype=np.float32)
    masks = np.zeros((len(drawings), config.max_len), dtype=np.float32)
    for i in range(len(drawings)):
        drawing = [[(float(x), float(y)) for x, y in s] for s in drawings.strokes(i)]
        strokes[i], masks[i] = encode(drawing, config)
    return strokes, masks


def run_onnx(session: ort.InferenceSession, strokes: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = [
        session.run([OUTPUT_NAME], {"strokes": strokes[i : i + 1], "mask": masks[i : i + 1]})[0]
        for i in range(len(strokes))
    ]
    return np.concatenate(out)


@torch.no_grad()
def run_torch(model: StrokeTransformer, strokes: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return model(torch.from_numpy(strokes), torch.from_numpy(masks)).numpy()


def check_parity(
    torch_logits: np.ndarray,
    fp32_logits: np.ndarray,
    int8_logits: np.ndarray,
    min_int8_agreement: float = MIN_INT8_AGREEMENT,
) -> dict[str, float]:
    fp32_max_diff = float(np.abs(torch_logits - fp32_logits).max())
    agreement = float((torch_logits.argmax(1) == int8_logits.argmax(1)).mean())
    result = {
        "n": int(len(torch_logits)),
        "fp32_max_abs_diff": fp32_max_diff,
        "int8_top1_agreement": agreement,
    }
    if fp32_max_diff > FP32_TOLERANCE:
        raise ParityError(f"fp32 ONNX differs from PyTorch by {fp32_max_diff:.2e} > 1e-4")
    if agreement < min_int8_agreement:
        raise ParityError(f"int8 top-1 agreement {agreement:.3%} < {min_int8_agreement:.0%}")
    return result


def measure_latency(path: Path, strokes: np.ndarray, masks: np.ndarray, runs: int = 200) -> dict:
    session = make_session(path)
    for i in range(10):  # warm-up
        session.run([OUTPUT_NAME], {"strokes": strokes[i : i + 1], "mask": masks[i : i + 1]})
    times = []
    for i in range(runs):
        j = i % len(strokes)
        start = time.perf_counter()
        session.run([OUTPUT_NAME], {"strokes": strokes[j : j + 1], "mask": masks[j : j + 1]})
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {
        "median_ms": round(statistics.median(times), 3),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 3),
        "size_kb": round(path.stat().st_size / 1024, 1),
    }


def build_meta(checkpoint: dict[str, Any], config: PreprocessConfig) -> dict[str, Any]:
    return {
        "version": META_VERSION,
        "categories": checkpoint["categories"],
        "max_len": config.max_len,
        "offset_scale": config.offset_scale,
        "rdp_epsilon": config.rdp_epsilon,
        "resample_spacing": config.resample_spacing,
        "input_names": INPUT_NAMES,
        "output_name": OUTPUT_NAME,
        "quantization": "dynamic-int8",
        "trained_epochs": int(checkpoint["epoch"]),
        "val_top1": round(float(checkpoint["metrics"]["val_top1"]), 4),
    }


def pick_model_cases(raw: DrawingSet, n: int = 5) -> list[int]:
    """Spread the cases over different classes and stroke counts."""
    rng = np.random.default_rng(0)
    classes = rng.choice(len(raw.categories), size=min(n, len(raw.categories)), replace=False)
    picks = []
    for c in classes:
        candidates = np.flatnonzero(raw.labels == c)
        picks.append(int(candidates[np.argmax(raw.strokes_per_drawing[candidates] >= 2)]))
    return picks


def write_model_cases(
    path: Path, session: ort.InferenceSession, raw: DrawingSet, config: PreprocessConfig
) -> list[dict[str, Any]]:
    cases = []
    for index in pick_model_cases(raw):
        strokes_raw = [s.astype(float).tolist() for s in raw.strokes(index)]
        strokes, mask = preprocess(strokes_raw, config)
        logits = session.run([OUTPUT_NAME], {"strokes": strokes[None], "mask": mask[None]})[0][0]
        top3 = np.argsort(-logits)[:3]
        cases.append(
            {
                "label": raw.categories[int(raw.labels[index])],
                "input": strokes_raw,
                "expected_top3": [raw.categories[int(i)] for i in top3],
                "expected_logits": [round(float(v), 6) for v in logits],
            }
        )
    payload = {
        "description": (
            "Raw test drawings with the int8 classifier.onnx output. Generated by "
            "ml/src/doodle_ml/export.py; do not edit by hand."
        ),
        # Dynamic int8 quantizes activations at run time; onnxruntime-web (WASM) and native
        # onnxruntime round slightly differently, so logits can differ by ~0.02. The fp32
        # model agrees to ~1e-6 across runtimes. Top-3 labels must match exactly.
        "logit_tolerance": 0.05,
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return cases


def run_export(
    checkpoint_path: Path = CHECKPOINT_DIR / "best.pt",
    processed_dir: Path = PROCESSED_DIR,
    web_models_dir: Path = WEB_MODELS_DIR,
    fixtures_dir: Path = FIXTURES_DIR,
    reports_dir: Path = REPORTS_DIR,
    parity_samples: int = 1000,
    min_int8_agreement: float = MIN_INT8_AGREEMENT,
) -> dict[str, Any]:
    model, checkpoint = load_model(checkpoint_path)
    config = PreprocessConfig(**checkpoint["preprocess"])

    simplified = load_simplified("test", processed_dir)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(simplified), size=min(parity_samples, len(simplified)), replace=False)
    strokes, masks = encode_set(simplified.subset(np.sort(sample)), config)

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "classifier_fp32.onnx"
        int8_path = Path(tmp) / "classifier_int8.onnx"
        export_onnx(model, fp32_path)
        quantize(fp32_path, int8_path)
        fp32_session, int8_session = make_session(fp32_path), make_session(int8_path)

        parity = check_parity(
            run_torch(model, strokes, masks),
            run_onnx(fp32_session, strokes, masks),
            run_onnx(int8_session, strokes, masks),
            min_int8_agreement,
        )
        latency = {
            "note": "onnxruntime CPU, 1 thread, batch 1, fixed shape [1, 200, 3]",
            "fp32": measure_latency(fp32_path, strokes, masks),
            "int8": measure_latency(int8_path, strokes, masks),
            "parity": parity,
        }

        web_models_dir.mkdir(parents=True, exist_ok=True)
        (web_models_dir / "classifier.onnx").write_bytes(int8_path.read_bytes())
        meta = build_meta(checkpoint, config)
        (web_models_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        write_preprocess_fixture(fixtures_dir / "preprocess_cases.json", config)
        raw_test = DrawingSet.load(processed_dir / "test.npz")
        write_model_cases(fixtures_dir / "model_cases.json", int8_session, raw_test, config)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latency.json").write_text(json.dumps(latency, indent=2) + "\n")
    return {"meta": meta, "latency": latency, "preprocess": asdict(config)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the classifier to ONNX for the web.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)
    out = run_export(args.checkpoint, args.processed_dir)
    lat, parity = out["latency"], out["latency"]["parity"]
    print(
        f"parity: fp32 max diff {parity['fp32_max_abs_diff']:.2e}, int8 top-1 agreement "
        f"{parity['int8_top1_agreement']:.2%} (n={parity['n']})"
    )
    for kind in ("fp32", "int8"):
        r = lat[kind]
        print(
            f"{kind}: {r['size_kb']:.0f} KB, median {r['median_ms']:.2f} ms, p95 {r['p95_ms']} ms"
        )
    print(f"wrote {WEB_MODELS_DIR}, {FIXTURES_DIR}, {REPORTS_DIR / 'latency.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
