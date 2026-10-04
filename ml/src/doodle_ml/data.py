"""PyTorch data pipeline: augmentation, length-bucketed batches, dynamic padding.

Dataset keys are ``(index, epoch)`` pairs. Augmentation draws its randomness from
``(seed, epoch, index)``, so a run is reproducible regardless of how many DataLoader
workers are used, and every epoch sees fresh augmentations.

Batches contain drawings of similar length and are padded only up to the longest one in
the batch (most drawings have under 60 points, far below max_len = 200). Because the
model ignores padding, this gives the same result as padding to 200, just faster.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from doodle_ml.augment import AugmentConfig, augment, encode_rows
from doodle_ml.dataset import DrawingSet
from doodle_ml.preprocess import PreprocessConfig

Key = tuple[int, int]  # (index, epoch)
Batch = tuple[Tensor, Tensor, Tensor]  # strokes (B, L, 3), mask (B, L), labels (B,)


class StrokeDataset(Dataset[tuple[np.ndarray, int]]):
    def __init__(
        self,
        drawings: DrawingSet,
        preprocess: PreprocessConfig,
        augment_config: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.drawings = drawings
        self.preprocess = preprocess
        self.augment_config = augment_config
        self.seed = seed

    def __len__(self) -> int:
        return len(self.drawings)

    def lengths(self) -> np.ndarray:
        """Unaugmented row count per drawing (used for bucketing)."""
        return np.minimum(self.drawings.points_per_drawing(), self.preprocess.max_len)

    def __getitem__(self, key: Key | int) -> tuple[np.ndarray, int]:
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        strokes, label = self.drawings[index]
        drawing = [s.astype(np.float64) for s in strokes]
        if self.augment_config is not None:
            rng = np.random.default_rng((self.seed, epoch, index))
            drawing = augment(drawing, rng, self.augment_config)
        rows = encode_rows(drawing, self.preprocess.offset_scale, self.preprocess.max_len)
        return rows, label


class BucketBatchSampler(Sampler[list[Key]]):
    """Shuffle, cut into pools of ``pool_batches`` batches, sort each pool by length,
    split into batches, shuffle the batches. Without ``shuffle`` it simply sorts by
    length (deterministic, for evaluation)."""

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        pool_batches: int = 50,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pool_batches = pool_batches
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[Key]]:
        n = len(self.lengths)
        if not self.shuffle:
            order = np.argsort(self.lengths, kind="stable")
            for i in range(0, n, self.batch_size):
                yield [(int(j), self.epoch) for j in order[i : i + self.batch_size]]
            return
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(n)
        pool = self.batch_size * self.pool_batches
        batches: list[np.ndarray] = []
        for start in range(0, n, pool):
            chunk = order[start : start + pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches.extend(
                chunk[i : i + self.batch_size] for i in range(0, len(chunk), self.batch_size)
            )
        for b in rng.permutation(len(batches)):
            yield [(int(j), self.epoch) for j in batches[b]]


def collate(items: Sequence[tuple[np.ndarray, int]]) -> Batch:
    """Pad to the longest drawing in the batch (at least 1 row)."""
    longest = max(1, max(len(rows) for rows, _ in items))
    strokes = torch.zeros(len(items), longest, 3)
    mask = torch.zeros(len(items), longest)
    for i, (rows, _) in enumerate(items):
        strokes[i, : len(rows)] = torch.from_numpy(rows)
        mask[i, : len(rows)] = 1.0
    labels = torch.tensor([label for _, label in items], dtype=torch.long)
    return strokes, mask, labels
