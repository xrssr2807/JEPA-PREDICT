# P2 核心消融实验

状态：代码已完成，等待远程服务器运行 seed 42 初筛。

计划比较：

1. 随机初始化 vs 预训练。
2. Phase 0 vs Phase 1 vs Phase 2。
3. 动态延迟/Transport 关闭 vs 开启。
4. Patient-level MIL 关闭 vs 开启。
5. Multi-scale 关闭 vs 开启。

先使用 seed 42 筛选。能够支持论文核心论点的比较再使用 seed 3407 和
seed 2026 复现。测试集始终保持封存。

## 已实现的实验开关

| 实验 ID | 唯一变化 | 对照 |
|---|---|---|
| `random_init` | 编码器随机初始化 | `phase2` |
| `phase0` | 使用 Phase 0 权重 | `phase2` |
| `phase1` | 使用 Phase 1 权重 | `phase2` |
| `phase2` | 完整 Phase 2 权重 | 主对照 |
| `transport_off` | 使用重新训练的无 Transport Phase 2 权重 | `phase2` |
| `mil_off` | 关闭 Patient-level MIL，保留双流编码与融合 | `phase2` |
| `multiscale_off` | 关闭下游多尺度头 | `phase2` |

所有组固定为八标签、患者级互斥划分、ECG+PPG 双通道、
Shared-Private 下游头关闭，并使用 `--seal_test`。

代码入口：

```text
scripts/run_p2_core_ablations.sh
scripts/run_p2_transport_ablation_pretrain.sh
scripts/summarize_p2_core_ablations.py
```

结果会自动归档到：

```text
paper/ICASSP2027/03_experiments/P2_core_ablations/results
```

详细命令见 `实验方案与运行说明.md`。
