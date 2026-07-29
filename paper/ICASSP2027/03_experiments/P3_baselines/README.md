# P3 对比模型实验

状态：尚未开始

所有基线必须使用相同的患者划分、标签体系、输入时长、验证集选模规则和
患者级评价方式。

候选模型：

- 监督式 1D CNN 或 ResNet。
- 从头训练的 Transformer。
- 重建式或掩码自编码器预训练。
- 对比表示学习。
- CWT-MAE Hybrid Fusion。
- xMAE（Physiology-Aware Masked Cross-Modal Reconstruction，2026）：与本文
  最接近的“ECG/PPG 有向生理时序”基线。若代码和预算允许，应使用相同冻结
  划分复现；否则至少在 Related Work 中逐项比较 directional masking、
  cross-attention 与本文概率单调 Transport 的差异。

## 已实现的一键内部基线

`scripts/run_icassp_remaining_experiments.sh` 会使用同一八标签患者级划分运行：

- 从头训练的 JEPA-Transformer 编码器（监督基线）。
- ResNet1D-18 编码器（监督卷积基线）。
- ECG/PPG 对称 InfoNCE 预训练后微调（自监督对比基线）。

每个下游模型运行 seed `42/3407/2026`，保存验证集患者级预测，并使用
`--seal_test` 保持测试集封存。流水线随后自动生成患者级 bootstrap 区间。

## 外部基线约束

CWT-MAE 与 xMAE 不在本仓库中。不得直接把其他划分、九标签版本或片段级
指标复制进主表。外部模型只有同时满足以下条件后才能加入：

1. 使用本项目冻结划分 SHA256。
2. 使用相同八标签定义和患者级 MIL/聚合评价。
3. 保存 1155 位验证患者的 UID、标签和概率。
4. 运行三个下游随机种子。
5. 日志明确测试集封存。

在外部仓库完成统一适配前，它们保持“待复现”，不阻塞本仓库的一键流水线。
