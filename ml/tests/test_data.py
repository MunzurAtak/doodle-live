import numpy as np
import torch

from doodle_ml.augment import AugmentConfig
from doodle_ml.data import BucketBatchSampler, StrokeDataset, collate
from doodle_ml.preprocess import PreprocessConfig
from helpers import synthetic_set


def test_collate_pads_to_longest_in_batch() -> None:
    items = [(np.ones((3, 3), np.float32), 0), (np.ones((5, 3), np.float32), 2)]
    strokes, mask, labels = collate(items)
    assert strokes.shape == (2, 5, 3)
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert not strokes[0, 3:].any()
    assert labels.tolist() == [0, 2]


def test_bucket_sampler_covers_every_index_once() -> None:
    lengths = np.random.default_rng(0).integers(1, 100, 1000)
    sampler = BucketBatchSampler(lengths, batch_size=32, shuffle=True, seed=1, pool_batches=4)
    indices = [i for batch in sampler for i, _ in batch]
    assert sorted(indices) == list(range(1000))
    assert len(list(sampler)) == len(sampler)


def test_bucket_sampler_groups_similar_lengths() -> None:
    lengths = np.random.default_rng(0).integers(1, 200, 4096)
    sampler = BucketBatchSampler(lengths, batch_size=64, shuffle=True, seed=1)
    spreads = [np.ptp(lengths[[i for i, _ in b]]) for b in sampler]
    assert np.mean(spreads) < 20  # vs ~190 for random batches


def test_bucket_sampler_epochs_differ_and_are_reproducible() -> None:
    lengths = np.arange(500)
    a = BucketBatchSampler(lengths, 16, shuffle=True, seed=1)
    b = BucketBatchSampler(lengths, 16, shuffle=True, seed=1)
    a.set_epoch(1)
    b.set_epoch(1)
    assert list(a) == list(b)
    b.set_epoch(2)
    assert list(a) != list(b)
    assert all(epoch == 2 for batch in b for _, epoch in batch)


def test_eval_sampler_sorted_by_length() -> None:
    lengths = np.array([5, 1, 3, 2])
    batches = list(BucketBatchSampler(lengths, 2, shuffle=False))
    assert [[i for i, _ in b] for b in batches] == [[1, 3], [2, 0]]


def test_dataset_augmentation_depends_on_epoch_only() -> None:
    drawings = synthetic_set(4, dtype=np.float32)
    ds = StrokeDataset(drawings, PreprocessConfig(offset_scale=10.0), AugmentConfig(), seed=7)
    np.testing.assert_array_equal(ds[(2, 1)][0], ds[(2, 1)][0])
    assert not np.array_equal(ds[(2, 1)][0], ds[(2, 2)][0])


def test_dataset_without_augmentation_matches_plain_encoding() -> None:
    drawings = synthetic_set(2, dtype=np.float32)
    ds = StrokeDataset(drawings, PreprocessConfig(offset_scale=2.0))
    rows, label = ds[0]
    first = drawings.strokes(0)[0].astype(np.float64)
    np.testing.assert_allclose(rows[0, :2], first[0] / 2.0, rtol=1e-6)
    assert label == 0
    assert torch.is_tensor(collate([ds[0], ds[1]])[0])
