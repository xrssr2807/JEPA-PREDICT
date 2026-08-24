# 公开预训练权重基线实验

## 目的

在同一个八标签、患者级互斥的开发集协议下，比较公开基础模型的冻结表征能力。本实验回答的是：在不改动官方编码器的情况下，其 PPG 表征对 CHD 和八病种分类有多少可转移信息。

## 固定协议

- 数据划分：`multidisease_taskaware_downstream.json`
- 划分 SHA256：`e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716`
- 输入：PPG 单通道
- 表征：冻结官方编码器
- 下游头：相同的 disease-conditioned patient MIL
- 训练目标：相同 ASL 和 CHD focus loss
- 下游随机种子：`42, 3407, 2026`
- 模型选择：仅使用 validation
- 测试集：封存，所有结果必须记录 `test_set_used=false`

## 官方接入

| 模型 | 官方输入处理 | 使用表征 | 说明 |
|---|---|---|---|
| PhysioV2 PPG | 100 Hz 原始 PPG | 512 维 target encoder 池化输出 | 项目内部的同协议主模型对照 |
| MOMENT-small | 原始序列归一化 | 512 维 embedding | 官方 `momentfm` 管线 |
| PaPaGei-S | polyphase 重采样到 125 Hz | 512 维 backbone 输出 | 严格加载官方权重 |
| NormWear | 重采样到 64 Hz 后官方 CWT | 768 维块/通道均值 | 使用官方 `NormWearModel` |
| UniTS-x128 | 单变量 tokenization | 128 维共享 backbone 输出 | 不使用数据集特定 prompt |
| Pulse-PPG | polyphase 重采样到 50 Hz | 512 维 ResNet1D 池化输出 | 加载官方预训练 checkpoint 的 `net` |

## 证据边界

1. 本表是本项目统一协议下的复评，不能与各论文在其他数据集、标签或划分上的数字直接比较。
2. UniTS 使用共享 backbone，未训练专用任务 prompt；因此它是冻结表征基线，不是完整微调上限。
3. 当前结果用于开发阶段模型选择；确定最终模型后，才能对封存测试集进行一次性评估。

汇总见 `results/official_fm_validation_table.md`。

## 当前结果（2026-08-24）

- PhysioV2 PPG 的三种子 CHD AUROC 为 `0.7695 +/- 0.0012`，CHD AUPRC 为 `0.3370 +/- 0.0048`。
- 最强公开基线的 CHD AUROC 为 MOMENT-small `0.7217 +/- 0.0005`。
- 三种子患者概率集成后，PhysioV2 相对 MOMENT-small 的 CHD AUROC 差值为 `+0.0487`，患者级配对 Bootstrap 95% CI 为 `[+0.0070, +0.0919]`。
- PhysioV2 在该协议下的 Macro AUROC 为 `0.7290 +/- 0.0024`，同样高于五个公开基线。

这些结果支持“PhysioV2 的 PPG 表征在本开发队列上具有 CHD 特异转移优势”，但不应扩展为外部队列或所有任务的全面优越性声明。学术曲线见 `results/official_fm_chd_roc_pr.png`，统计比较见 `results/official_fm_chd_statistical_comparison.md`。
