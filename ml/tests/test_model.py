import torch

from doodle_ml.model import ModelConfig, StrokeTransformer, count_parameters

SMALL = ModelConfig(n_classes=5, max_len=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def make_model(config: ModelConfig = SMALL) -> StrokeTransformer:
    torch.manual_seed(0)
    return StrokeTransformer(config).eval()


def test_output_shape() -> None:
    model = make_model()
    logits = model(torch.randn(3, 20, 3), torch.ones(3, 20))
    assert logits.shape == (3, 5)


def test_default_size_matches_spec() -> None:
    params = count_parameters(StrokeTransformer(ModelConfig()))
    assert 500_000 < params < 800_000


def test_padding_does_not_change_output() -> None:
    """Dynamic padding in training and fixed-length (200) export must agree."""
    model = make_model()
    x = torch.randn(1, 10, 3)
    short = model(x, torch.ones(1, 10))
    padded_x = torch.cat([x, torch.randn(1, 22, 3) * 100], dim=1)  # garbage in padding
    padded_mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 22)], dim=1)
    padded = model(padded_x, padded_mask)
    torch.testing.assert_close(short, padded, atol=1e-5, rtol=1e-5)


def test_batch_items_are_independent_of_each_other() -> None:
    model = make_model()
    a, b = torch.randn(1, 12, 3), torch.randn(1, 12, 3)
    mask_a = torch.ones(1, 12)
    mask_b = torch.cat([torch.ones(1, 5), torch.zeros(1, 7)], dim=1)
    together = model(torch.cat([a, b]), torch.cat([mask_a, mask_b]))
    torch.testing.assert_close(together[0:1], model(a, mask_a), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        together[1:2], model(b[:, :5], torch.ones(1, 5)), atol=1e-5, rtol=1e-5
    )


def test_all_padding_input_is_finite() -> None:
    logits = make_model()(torch.zeros(2, 8, 3), torch.zeros(2, 8))
    assert torch.isfinite(logits).all()


def test_gradients_flow() -> None:
    model = make_model().train()
    loss = model(torch.randn(4, 9, 3), torch.ones(4, 9)).logsumexp(-1).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
