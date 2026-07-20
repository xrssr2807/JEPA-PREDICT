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
    pretrain_val_split: float = 0.10  # patient-group holdout for Phase 0 validation

    # Downstream data
    chd_ppg_dir: str = "/root/chd_ppg"
    chd_ecg_dir: str = "/root/chd_ecg"
    chd_ecg_subdir: str = "ecg_chd"     # ECG数据子目录 (下含.pkl文件)
    arrhythmia_dir: str = "/root/processed_dataset"
    multidisease_dir: str = "/root/ppgchd/ppgchd/data_updated"
    multidisease_development_split: str = "splits/development_split.json"
    multidisease_split_file: str = "splits/multidisease_patient_split.json"
    multidisease_taskaware_split_file: str = "splits/multidisease_taskaware_split.json"

    # Preprocessing
    normalize: str = "zscore"  # zscore / iqr / minmax / none
    normalize_clip: float = 10.0  # clip value after zscore/iqr normalization
    pretrain_channels: List[int] = field(default_factory=lambda: [0, 4])  # ch0=ECG, ch4=PPG
    channel_names: List[str] = field(
        default_factory=lambda: ["ECG", "ACC_X", "ACC_Y", "ACC_Z", "PPG"]
    )

    # ★ 信号质量门控：过滤低质量PPG样本 (PPG BP综述: 使用SQA后精度提升19-24%)
    signal_quality_gate: float = 0.0  # 0=关闭 (SQI对CHD数据过滤过严)
    val_split: float = 0.15  # 训练集留出15%做验证集 (按标签分层)
    signal_align_to: int = 0  # 下游信号对齐到预训练长度 (0=不对齐)
    multidisease_channel: str = "both"  # "0" | "1" | "both"
    multidisease_dual_stream: bool = True
    multidisease_ppg_channel: int = 0
    multidisease_ecg_channel: int = 1
    multidisease_use_multiscale: bool = True
    multidisease_patient_mil: bool = True
    multidisease_mil_segments: int = 8
    multidisease_mil_encoder_chunk_size: int = 128

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
    multidisease_labels: List[str] = field(default_factory=lambda: [
        "高血压", "高血糖", "高血脂", "下肢动脉硬化闭塞症", "冠心病",
        "心律失常（房颤、频发早搏等）", "糖尿病", "脑卒中（中风）", "颈动脉斑块",
    ])


@dataclass
class ModelConfig:
    """JEPA model architecture hyperparameters."""

    # Phase 0 remains reproducible with ``python train_pretrain.py --phase 0``.
    pretrain_phase: int = 1
    # Phase 1: dual-online / dual-teacher masked-token JEPA (B2).
    phase1_mask_ratio: float = 0.60
    phase1_mask_block_tokens: int = 8
    phase1_bidirectional: bool = True
    phase1_token_loss_weight: float = 1.0
    phase1_use_stats_loss: bool = False

    # Phase 2: causal, positive-delay monotonic token transport (B3).
    phase2_sample_rate_hz: float = 100.0
    phase2_min_delay_ms: float = 80.0
    phase2_max_delay_ms: float = 800.0
    phase2_delay_prior_ms: float = 250.0
    phase2_delay_head_hidden: int = 128
    phase2_transport_temperature: float = 0.20
    phase2_unmatched_bias: float = -2.0
    phase2_transport_loss_weight: float = 1.0
    phase2_delay_prior_weight: float = 0.02
    phase2_monotonic_weight: float = 0.05
    phase2_delay_smoothness_weight: float = 0.01
    phase2_match_mass_weight: float = 0.01
    phase2_target_match_mass: float = 0.95
    phase2_variance_weight: float = 0.10
    phase2_covariance_weight: float = 0.01
    phase2_target_std: float = 0.10
    phase2_use_stats_loss: bool = False

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

    # ── JETS 式掩码策略 ──
    # 随机掩码信号patch，强制编码器从局部信息学习全局表征
    jets_mask_ratio: float = 0.6   # 0=关闭, 0.6=保留40%patch（从0.7调低）
    jets_mask_patch_size: int = 50 # 每个patch的采样点数 (3000/50=60个patch, 下游1000/50=20个patch)

    # ── Auxiliary losses (from CWT-MAE v3) ──
    use_stats_loss: bool = True        # auxiliary statistics prediction (Apple-style)
    stats_loss_weight: float = 0.1     # weight for stats loss
    # Token级对齐
    use_token_align: bool = True       # ★ Soft-DTW 弹性对齐
    token_align_weight: float = 0.1    # 辅助损失权重 (和 StatsLoss 同级)
    token_align_window: int = 3        # Soft-DTW 搜索窗口 (±3 token ≈ ±300ms)

    # ── CNN 增强 ──
    cnn_use_se: bool = True            # SE Block 通道注意力
    cnn_use_inception: bool = True     # Inception 残差多尺度 (alpha=0.2)

    # ── CWT Frontend (optional alternative to 1D CNN) ──
    use_cwt: bool = False              # use CWT 1D→2D frontend instead of CNN Stem
    cwt_scales: int = 64               # number of wavelet scales
    cwt_use_diff: bool = True          # include 1st/2nd derivative signals
    cwt_patch_freq: int = 8            # patch size in frequency dimension
    cwt_patch_time: int = 25           # patch size in time dimension

    # ── Downstream ──
    use_cot_head: bool = False         # CoT collapses with frozen encoder
    cot_tokens: int = 16               # number of reasoning tokens
    use_layerwise_lr: bool = False     # uniform LR (避免CoT坍塌)
    layer_decay: float = 0.85          # softened decay (was 0.75)
    # ★ XGBoost 替代微调 (M2AE: 冻结编码器+XGBoost → CVD AUROC 0.974)
    use_xgboost: bool = False          # 使用神经网络微调 + LayerDrop
    # ★ HuBERT-ECG LayerDrop (下游微调防过拟合: 随机丢20% transformer层)
    downstream_layerdrop: float = 0.3  # ECGFounder-PT: Stochastic Depth
    # ★ HiMAE 多尺度分类头 (不同疾病依赖不同时间尺度)
    use_multiscale: bool = False       # True=使用MultiScaleClassifier
    # ★ ECG+PPG 双通道融合 (CSFM: 多模态融合持续带来稳健提升)
    use_dual_channel: bool = False     # 单通道 PPG only
    use_ecg_distill: bool = False      # ECG蒸馏 (关闭, 先测纯PPG)
    use_cotrain: bool = True           # ★ ECG+PPG协同训练 (共享分类头, 部署仅需PPG)
    use_dual_channel: bool = True      # ★ ECG+PPG concat融合 (AUC 0.79)
    distill_lambda: float = 0.3


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Pre-training
    pretrain_epochs: int = 150
    pretrain_batch_size: int = 240
    pretrain_accum_steps: int = 2
    pretrain_lr: float = 5e-4
    pretrain_warmup_epochs: int = 5  # shorter warmup → earlier cosine decay
    pretrain_weight_decay: float = 0.05
    pretrain_val_every: int = 1

    # Phase 1 doubles online branches. AMP plus accumulation keeps the
    # effective batch near the Phase 0 value while fitting a 24 GB GPU.
    phase1_batch_size: int = 160
    phase1_accum_steps: int = 3
    phase1_lr: float = 3e-4
    phase1_warmup_epochs: int = 10
    phase1_use_amp: bool = True
    # Phase 2 uses the Phase 1 backbone from random initialization and ramps
    # transport in only after token representations become meaningful.
    phase2_batch_size: int = 128
    phase2_accum_steps: int = 3
    phase2_lr: float = 2e-4
    phase2_warmup_epochs: int = 10
    phase2_use_amp: bool = True
    phase2_transport_start_epoch: int = 10
    phase2_transport_ramp_epochs: int = 20
    # Count only full-transport, healthy validation epochs. A decrease smaller
    # than min_delta is treated as a plateau rather than a meaningful gain.
    phase2_early_stop_patience: int = 15
    phase2_early_stop_min_delta: float = 1e-4
    pretrain_dataloader_workers: int = 8
    pretrain_prefetch_factor: int = 4

    # Phase 3A: downstream-feedback-aware Phase 2 pre-training.
    taskaware_epochs: int = 30
    taskaware_feedback_interval: int = 20
    taskaware_feedback_batch_size: int = 8
    taskaware_feedback_segments: int = 4
    taskaware_feedback_encoder_chunk_size: int = 32
    taskaware_feedback_start_epoch: int = 5
    taskaware_head_warmup_steps: int = 50
    taskaware_head_lr: float = 5e-4
    taskaware_feedback_encoder_grad_ratio: float = 0.20
    taskaware_feedback_grad_clip: float = 1.0

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
    downstream_batch_size: int = 256
    multidisease_mil_batch_size: int = 64
    multidisease_probe_batch_size: int = 96  # frozen encoders allow a much larger patient batch
    multidisease_probe_encoder_chunk_size: int = 128
    dataloader_workers: int = 8
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    enable_tf32: bool = True
    downstream_use_amp: bool = True
    downstream_lr: float = 5e-4  # FT base=5e-5 (anti-overfit, 最优)
    downstream_min_lr: float = 1e-6
    downstream_warmup_epochs: int = 5
    downstream_probe_epochs: int = 20  # AUC在E20后不再提升
    downstream_scheduler: str = "step"  # "epoch" or "step" (step-based for warmup+cosine)

    # ★ Token 对齐续训练 (冻结 target, 训练 context 对齐到 target)
    token_align_epochs: int = 50       # 续训练epoch
    token_align_lr: float = 1e-4       # 续训练学习率 (比预训练小)
    use_mixup: bool = True          # True=启用MixUp
    mixup_alpha: float = 0.5        # Beta分布参数 (0.5=中等混合强度)

    # Downstream loss
    loss_type: str = "focal"  # "ce" | "focal" | "asl" | "bce"
    focal_gamma: float = 2.0
    asl_gamma_neg: int = 4
    asl_gamma_pos: int = 1
    asl_clip: float = 0.05
    label_smoothing: float = 0.0
    auto_pos_weight: bool = True  # compute pos_weight from training data
    multilabel_loss_type: str = "asl"  # "asl" | "bce"
    chd_label_index: int = 4  # 冠心病在 multidisease_labels 中的索引
    chd_focus_loss_weight: float = 0.5  # extra BCE loss weight for 冠心病
    chd_auc_loss_weight: float = 0.1  # patient-level pairwise ranking loss
    chd_auc_margin: float = 0.2
    best_metric: str = "hybrid"  # "chd_auc" | "macro_auc" | "hybrid"
    best_metric_chd_alpha: float = 0.7  # hybrid = alpha*CHD_AUC + (1-alpha)*macro_AUC
    threshold_strategy: str = "recall_floor"  # "fbeta" | "recall_floor"
    threshold_recall_floor_all_labels: bool = False  # keep the recall constraint focused on CHD
    threshold_beta: float = 0.75
    threshold_recall_floor: float = 0.60
    threshold_opt_metric: str = "f05"  # "accuracy" | "precision" | "f05" | "f1"


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "/root/autodl-tmp/JEPA-PREDICT/outputs"
    deterministic: bool = True


# Default config instance
config = Config()
