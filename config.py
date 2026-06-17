"""
Configuration for JEPA ECG-PPG pre-training and downstream fine-tuning.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Data paths and preprocessing parameters."""

    # Pre-training data (unlabeled 5-channel .pkl files)
    pretrain_dir: str = "/root/autodl-tmp/split"
    # Preprocessed pre-training data (.pt files, after channel extraction + Z-score)
    pretrain_processed_dir: str = "/root/autodl-tmp/split_processed"

    # Downstream data
    chd_ppg_dir: str = "/root/chd_ppg"
    chd_ecg_dir: str = "/root/autodl-tmp/chd_ecg"
    arrhythmia_dir: str = "/root/processed_dataset"

    # Preprocessing
    normalize: str = "zscore"  # zscore / iqr / minmax / none
    normalize_clip: float = 10.0  # clip value after zscore/iqr normalization
    pretrain_channels: List[int] = field(default_factory=lambda: [0, 4])  # ch0=ECG, ch4=PPG
    channel_names: List[str] = field(
        default_factory=lambda: ["ECG", "ACC_X", "ACC_Y", "ACC_Z", "PPG"]
    )

    # Augmentation (PhysioAugment — applied to ECG context signal)
    use_augment: bool = False
    augment_jitter_std: float = 0.02
    augment_scale_min: float = 0.85
    augment_scale_max: float = 1.15
    augment_max_shift: int = 50
    augment_wander_amp: float = 0.05
    augment_apply_prob: float = 0.8

    # Downstream label mapping
    num_classes: int = 2  # CHD: binary classification
    arrhythmia_num_classes: int = 6  # Arrhythmia: 6-class classification


@dataclass
class ModelConfig:
    """JEPA model architecture hyperparameters."""

    # Input
    in_channels: int = 1  # single-channel (ECG or PPG separately)
    signal_length: int = 3000  # pre-training: 30s @ 100Hz

    # CNN Stem
    cnn_channels: List[int] = field(
        default_factory=lambda: [128, 256, 512, 512]
    )
    cnn_kernel_sizes: List[int] = field(
        default_factory=lambda: [7, 5, 5, 3]
    )
    cnn_strides: List[int] = field(
        default_factory=lambda: [2, 2, 2, 2]
    )  # total stride = 16 → 3000→188

    # Transformer
    transformer_layers: int = 8
    transformer_dim: int = 512
    transformer_heads: int = 16
    transformer_ff_dim: int = 2048
    transformer_dropout: float = 0.1
    max_seq_len: int = 200  # enough for 3000/16=188

    # JEPA projection / predictor
    embedding_dim: int = 256  # final embedding for prediction target
    predictor_hidden: int = 256
    latent_dim: int = 64  # latent variable for multi-modal predictions
    num_latent_samples: int = 4  # number of z samples during training

    # EMA for target encoder
    # 低 → 高：LR 大时 target 快速追踪 context，LR 小时 target 稳定微调
    ema_momentum: float = 0.996  # initial: stable tracking (0.4% update per step)
    ema_end_momentum: float = 0.999  # final: slow fine-tuning (0.1% update per step)

    # Pooling
    pool_type: str = "adaptive_avg"  # adaptive average pooling → fixed output size

    # ── Auxiliary losses (from CWT-MAE v3) ──
    use_stats_loss: bool = False       # auxiliary statistics prediction
    stats_loss_weight: float = 0.1     # weight for stats loss
    use_contrast_loss: bool = False    # BYOL-style contrastive learning
    contrast_loss_weight: float = 0.1  # weight for contrastive loss
    contrast_decay: float = 0.999      # EMA decay for teacher projector

    # ── CWT Frontend (optional alternative to 1D CNN) ──
    use_cwt: bool = False              # use CWT 1D→2D frontend instead of CNN Stem
    cwt_scales: int = 64               # number of wavelet scales
    cwt_use_diff: bool = True          # include 1st/2nd derivative signals
    cwt_patch_freq: int = 8            # patch size in frequency dimension
    cwt_patch_time: int = 25           # patch size in time dimension

    # ── Downstream ──
    use_cot_head: bool = True          # Chain-of-Thought classification head
    cot_tokens: int = 16               # number of reasoning tokens
    use_layerwise_lr: bool = True      # layer-wise learning rate decay
    layer_decay: float = 0.75          # decay factor per layer


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Pre-training
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 310
    pretrain_lr: float = 5e-4
    pretrain_warmup_epochs: int = 5  # shorter warmup → earlier cosine decay
    pretrain_weight_decay: float = 0.05

    # Optimizer
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95

    # Scheduler
    lr_schedule: str = "cosine"

    # EMA schedule
    ema_schedule: str = "cosine"  # cosine schedule for target encoder momentum

    # Downstream
    downstream_epochs: int = 100
    downstream_batch_size: int = 128
    downstream_lr: float = 1e-3
    downstream_min_lr: float = 1e-6
    downstream_warmup_epochs: int = 5
    downstream_probe_epochs: int = 10  # linear probe only (frozen encoder)
    downstream_scheduler: str = "step"  # "epoch" or "step" (step-based for warmup+cosine)

    # Downstream loss
    loss_type: str = "focal"  # "ce" | "focal" | "asl" | "bce"
    focal_gamma: float = 2.0
    asl_gamma_neg: int = 4
    asl_gamma_pos: int = 1
    asl_clip: float = 0.05
    label_smoothing: float = 0.0
    auto_pos_weight: bool = True  # compute pos_weight from training data


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "/root/autodl-tmp/JEPA-PREDICT/outputs"


# Default config instance
config = Config()
