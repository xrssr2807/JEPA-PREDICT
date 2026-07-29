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
