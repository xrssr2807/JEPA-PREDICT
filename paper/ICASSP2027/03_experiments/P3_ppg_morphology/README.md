# PPG形态学残差头消融

## 目的

选择性双通道教师蒸馏未提高 PPG-only CHD AUC。本实验不再复制 ECG
教师表征，而是直接补充 PPG 可观测的脉搏形态信息，检验预训练 encoder
是否遗漏了对冠心病有用的血管形态线索。

## 严格对照

- `baseline`：原始 PPG encoder + multi-scale Patient-MIL。
- `morphology`：在完全相同配置上增加小型形态残差分支。
- 形态特征包括标准化振幅分位数、一/二阶差分、转折率和四个频带能量。
- 残差门初始约为 0.12，避免新分支覆盖预训练表示。
- 固定八标签患者级划分、Phase2 PhysioV2 权重、seed 42。
- 测试集全程封存。

## 决策门

只有 `morphology - baseline` 同时满足：

1. CHD AUC 增量不低于 `+0.005`；
2. Macro AUC 下降不超过 `0.005`；

才继续运行 seed 3407 和 2026。否则停止该方向，不继续搜索验证集。

## 运行

```bash
nohup bash scripts/run_ppg_morphology_ablation.sh \
  > logs/ppg_morphology_seed42.log 2>&1 &
```
