# 双向任务非对称权重实验

## 目的

验证“ECG→PPG 为生理主任务、PPG→ECG 为辅助表征任务”的非对称设计，是否比历史上的等权双向训练更适合冠心病识别。

Phase 2 的方向损失定义为：

`L_dir = (L_ECG→PPG + alpha * L_PPG→ECG) / (1 + alpha)`

- ECG→PPG 权重固定为 `1.0`。
- `alpha` 控制反向辅助任务，初筛取 `{0, 0.1, 0.25, 0.5, 1.0}`。
- 分母归一化保证不同 alpha 下的损失尺度可比。
- `alpha=1.0` 与历史对称双向目标严格等价。

## 固定条件

- 相同 Phase 2 Shared-Private 初始化权重。
- 相同 PhysioV2 Transport、训练轮数、学习率和随机种子。
- 相同八标签患者级划分，SHA256 为 `e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716`。
- 相同双通道 Patient-MIL、多尺度下游头，Shared-Private 下游头关闭。
- 测试集封存，仅使用验证集冠心病 AUC 做单种子初筛，Macro AUC 作为并列时的次要判据。

## 运行

```bash
bash scripts/run_phase2_direction_weight_ablation.sh
```

结果自动归档到：

```text
paper/ICASSP2027/03_experiments/P2_direction_weight/results/seed42
```

该实验只完成 alpha 的单种子筛选。确定候选 alpha 后，还需对候选值和 `alpha=1.0` 分别进行独立预训练种子复现，不能把下游随机种子重复误当作预训练复现。
