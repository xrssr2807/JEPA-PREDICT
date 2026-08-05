import torch

from config import Config
from models.crossmodal_mae import (
    CrossModalMaskedAutoencoder,
    make_patch_mask,
    masked_patch_mse,
)
from train_downstream import build_encoder, load_pretrained_encoder


def tiny_model(objective):
    config = Config()
    model_config = config.model
    model_config.cnn_channels = [8, 16]
    model_config.cnn_kernel_sizes = [7, 5]
    model_config.cnn_strides = [2, 2]
    model_config.cnn_use_se = False
    model_config.cnn_use_inception = False
    model_config.transformer_layers = 1
    model_config.transformer_dim = 32
    model_config.transformer_heads = 4
    model_config.transformer_ff_dim = 64
    model_config.max_seq_len = 64
    ecg = build_encoder(model_config, in_channels=1)
    ppg = build_encoder(model_config, in_channels=1)
    model = CrossModalMaskedAutoencoder(
        ecg,
        ppg,
        objective=objective,
        model_dim=32,
        heads=4,
        ff_dim=64,
        decoder_depth=1,
        max_patch_size=16,
    )
    return model, model_config


def test_anchor_mask_has_one_contiguous_visible_region():
    mask = make_patch_mask(8, 20, 0.8, "anchor", torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert torch.all(mask.sum(dim=1) == 16)
    for row in mask:
        visible = (~row).nonzero(as_tuple=False).flatten()
        assert torch.all(visible[1:] - visible[:-1] == 1)


def test_masked_patch_mse_ignores_visible_patches():
    target = torch.zeros(1, 3, 2)
    prediction = torch.tensor([[[9.0, 9.0], [1.0, 1.0], [3.0, 3.0]]])
    mask = torch.tensor([[False, True, False]])
    assert torch.isclose(masked_patch_mse(prediction, target, mask), torch.tensor(1.0))


def test_multimodal_mae_reconstructs_both_modalities():
    model, _ = tiny_model("multimodal_mae")
    ecg = torch.randn(2, 1, 128)
    ppg = torch.randn(2, 1, 128)
    tokens = model.token_count(128)
    ecg_mask = make_patch_mask(2, tokens, 0.5, "scatter", ecg.device)
    ppg_mask = make_patch_mask(2, tokens, 0.5, "scatter", ppg.device)
    output = model(ecg, ppg, ecg_mask, ppg_mask)
    losses = model.compute_loss(output)
    assert output["ecg_prediction"].shape == output["ecg_target"].shape
    assert output["ppg_prediction"].shape == output["ppg_target"].shape
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_xmae_objective_reconstructs_masked_ecg_only():
    model, _ = tiny_model("xmae_objective")
    ecg = torch.randn(2, 1, 128)
    ppg = torch.randn(2, 1, 128)
    tokens = model.token_count(128)
    ecg_mask = make_patch_mask(2, tokens, 0.75, "anchor", ecg.device)
    ppg_mask = torch.zeros_like(ecg_mask)
    output = model(ecg, ppg, ecg_mask, ppg_mask)
    losses = model.compute_loss(output)
    assert "ppg_prediction" not in output
    assert losses["ppg"].item() == 0.0
    assert torch.isclose(losses["total"], losses["ecg"])


def test_checkpoint_is_compatible_with_downstream_loader(tmp_path):
    model, model_config = tiny_model("xmae_objective")
    checkpoint = tmp_path / "xmae.pt"
    torch.save(
        {
            "context_encoder": model.ecg_encoder.state_dict(),
            "target_encoder": model.ppg_encoder.state_dict(),
        },
        checkpoint,
    )
    restored_ecg = load_pretrained_encoder(
        str(checkpoint),
        model_config,
        "context",
        torch.device("cpu"),
        in_channels=1,
    )
    restored_ppg = load_pretrained_encoder(
        str(checkpoint),
        model_config,
        "target",
        torch.device("cpu"),
        in_channels=1,
    )
    assert restored_ecg.state_dict().keys() == model.ecg_encoder.state_dict().keys()
    assert restored_ppg.state_dict().keys() == model.ppg_encoder.state_dict().keys()

