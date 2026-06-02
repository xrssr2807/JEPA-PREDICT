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
    chd_ppg_dir: str = "/root/autodl-tmp/chd_ppg"
    chd_ecg_dir: str = "/root/autodl-tmp/chd_ecg"
    arrhythmia_dir: str = "/root/autodl-tmp/processed_dataset"

    # Preprocessing
    normalize: str = "zscore"  # zscore per file
    pretrain_channels: List[int] = field(default_factory=lambda: [0, 4])  # ch0=ECG, ch4=PPG
    channel_names: List[str] = field(
        default_factory=lambda: ["ECG", "ACC_X", "ACC_Y", "ACC_Z", "PPG"]
    )

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
    ema_momentum: float = 0.996  # initial momentum
    ema_end_momentum: float = 1.0  # final momentum (1.0 = no update)

    # Pooling
    pool_type: str = "adaptive_avg"  # adaptive average pooling → fixed output size


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Pre-training
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 310
    pretrain_lr: float = 2.5e-3
    pretrain_warmup_epochs: int = 5
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
    downstream_epochs: int = 50
    downstream_batch_size: int = 128
    downstream_lr: float = 1e-3
    downstream_probe_epochs: int = 10  # linear probe only (frozen encoder)


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs"


# Default config instance
config = Config()
