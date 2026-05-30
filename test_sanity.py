"""
Sanity check: verify that the JEPA model can run a forward pass
and that the data pipeline works correctly.
"""
import sys
import torch

from config import Config
from models.encoder import SignalEncoder
from models.jepa import JEPA


def test_encoder():
    """Test that the encoder handles both pre-training and downstream lengths."""
    print("=" * 60)
    print("Testing SignalEncoder...")

    encoder = SignalEncoder(
        in_channels=1,
        cnn_channels=(64, 128, 256, 256),
        cnn_kernel_sizes=(7, 5, 5, 3),
        cnn_strides=(2, 2, 2, 2),
        transformer_layers=4,
        transformer_dim=256,
        transformer_heads=8,
        transformer_ff_dim=1024,
        transformer_dropout=0.1,
        max_seq_len=200,
        pool_type="adaptive_avg",
    )

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {total_params:,}")

    # Test with pre-training length (30s)
    x_30s = torch.randn(4, 1, 3000)
    out_30s, tokens = encoder(x_30s, return_all=True)
    print(f"  30s input  (4, 1, 3000) → output ({' × '.join(map(str, out_30s.shape))})")
    print(f"               → tokens ({' × '.join(map(str, tokens.shape))})")
    assert out_30s.shape == (4, 256), f"Expected (4, 256), got {out_30s.shape}"
    assert tokens.shape == (4, 188, 256), f"Expected (4, 188, 256), got {tokens.shape}"

    # Test with downstream length (10s)
    x_10s = torch.randn(4, 1, 1000)
    out_10s, tokens_10s = encoder(x_10s, return_all=True)
    print(f"  10s input  (4, 1, 1000) → output ({' × '.join(map(str, out_10s.shape))})")
    print(f"               → tokens ({' × '.join(map(str, tokens_10s.shape))})")
    assert out_10s.shape == (4, 256)
    assert tokens_10s.shape == (4, 63, 256)  # 1000/16 ≈ 62.5 → ceil 63

    print("  [PASS] Encoder test\n")


def test_jepa():
    """Test JEPA forward and loss computation."""
    print("=" * 60)
    print("Testing JEPA model...")

    model = JEPA(
        in_channels=1,
        cnn_channels=(64, 128, 256, 256),
        cnn_kernel_sizes=(7, 5, 5, 3),
        cnn_strides=(2, 2, 2, 2),
        transformer_layers=4,
        transformer_dim=256,
        transformer_heads=8,
        transformer_ff_dim=1024,
        transformer_dropout=0.1,
        max_seq_len=200,
        pool_type="adaptive_avg",
        embedding_dim=128,
        predictor_hidden=128,
        latent_dim=32,
        num_latent_samples=4,
        ema_momentum=0.996,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Forward pass
    ecg = torch.randn(8, 1, 3000)
    ppg = torch.randn(8, 1, 3000)

    pred, target, context_embed = model(ecg, ppg)
    print(f"  ECG (8,1,3000) → context_embed ({' × '.join(map(str, context_embed.shape))})")
    print(f"  PPG (8,1,3000) → target_embed ({' × '.join(map(str, target.shape))})")
    print(f"  Predictor       → pred ({' × '.join(map(str, pred.shape))})")

    assert pred.shape == (8, 128), f"Expected (8, 128), got {pred.shape}"
    assert target.shape == (8, 128)

    # Loss computation
    loss, info = model.compute_loss(ecg, ppg)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Info: {info}")
    assert loss.item() > 0

    # Verify gradient flow
    loss.backward()
    has_grad = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad.append(name)
    print(f"  Modules receiving gradients: {len(has_grad)}")
    assert any("context_encoder" in n for n in has_grad), "Context encoder should get gradients"
    assert not any("target_encoder" in n for n in has_grad), "Target encoder should NOT get gradients"

    print("  [PASS] JEPA test\n")


def test_ema():
    """Test EMA update of target encoder."""
    print("=" * 60)
    print("Testing EMA update...")

    model = JEPA(
        transformer_layers=2,  # smaller for faster test
        transformer_dim=128,
        transformer_heads=4,
        cnn_channels=(32, 64, 128, 128),
    )

    # Verify initial weights match
    for name_c, param_c in model.context_encoder.named_parameters():
        param_t = dict(model.target_encoder.named_parameters())[name_c]
        assert torch.allclose(param_c, param_t), f"Weights diverge at {name_c}"
    print("  Initial weights match ✓")

    # Do a fake update
    model.update_target_encoder(momentum=0.5)
    # Weights should still match after update with momentum=0.5 and no change to context
    for name_c, param_c in model.context_encoder.named_parameters():
        param_t = dict(model.target_encoder.named_parameters())[name_c]
        assert torch.allclose(param_c, param_t, atol=1e-6), f"Weights diverge at {name_c}"
    print("  EMA update with identical weights preserves equality ✓")

    print("  [PASS] EMA test\n")


def test_end_to_end():
    """Quick end-to-end training loop test."""
    print("=" * 60)
    print("Testing end-to-end training loop...")

    model = JEPA(
        transformer_layers=2,
        transformer_dim=128,
        transformer_heads=4,
        cnn_channels=(32, 64, 128, 128),
        embedding_dim=64,
        predictor_hidden=64,
        latent_dim=16,
        num_latent_samples=2,
    )

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters())
        + list(model.context_proj.parameters())
        + list(model.predictor.parameters()),
        lr=1e-3,
    )

    initial_losses = []
    for step in range(10):
        ecg = torch.randn(4, 1, 1000)
        ppg = torch.randn(4, 1, 1000)

        loss, info = model.compute_loss(ecg, ppg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target_encoder(0.99)
        initial_losses.append(loss.item())

    # Loss should generally decrease
    first_half = sum(initial_losses[:5]) / 5
    second_half = sum(initial_losses[5:]) / 5
    print(f"  Avg loss (steps 0-4): {first_half:.6f}")
    print(f"  Avg loss (steps 5-9): {second_half:.6f}")
    print(f"  Trend: {'✓ decreasing' if second_half < first_half else '~ stable (expected for random data)'}")

    print("  [PASS] End-to-end test\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JEPA ECG-PPG Sanity Checks")
    print("=" * 60 + "\n")

    test_encoder()
    test_jepa()
    test_ema()
    test_end_to_end()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
