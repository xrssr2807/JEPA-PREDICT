# Transport 约束组成消融

## 研究问题

该实验验证“生理约束的动态因果 Transport”究竟由哪些部分产生收益。
所有变体都进行独立 Phase 2 预训练，不能只在解释性分析阶段替换延迟。

| 模式 | 保留内容 | 被移除或破坏的内容 |
|---|---|---|
| `full` | token 动态延迟、正延迟、单调正则 | 无，论文完整模型 |
| `static_delay` | 每个样本一个延迟分布 | token 级动态性 |
| `fixed_prior` | 固定到最接近 250 ms 的延迟 bin | 数据驱动延迟学习 |
| `zero_delay` | 同时刻 ECG/PPG 对齐 | 正向生理传播延迟 |
| `no_monotonic` | 动态正延迟 Transport | 单调路径正则 |
| `token_shuffled` | 延迟分布及边际统计 | token 与时间位置的对应关系 |

`zero_delay` 在目标构造上接近直接同位置预测，因此应与已有
Transport-off 结果联合解释。`no_monotonic` 只将单调正则权重置零，
不会改变其余损失。

## 固定条件

- 八标签患者级互斥划分。
- 双通道 ECG+PPG。
- 下游 Shared-Private head 关闭。
- Patient-level MIL 与 multi-scale 均开启。
- 预训练数据划分 seed 固定为 42。
- 第一轮仅跑预训练 seed 42、下游 seed 42。
- 模型和阈值只根据验证集选择，测试集保持封存。

## 远程运行

```bash
cd /root/autodl-tmp/JEPA-PREDICT-priority1
conda activate JEPA

mkdir -p logs
nohup bash scripts/run_transport_constraint_ablation.sh \
  > logs/transport_constraint_ablation_seed42.log 2>&1 &

echo $! > logs/transport_constraint_ablation_seed42.pid
tail -f logs/transport_constraint_ablation_seed42.log
```

快速检查脚本和接口，不启动训练：

```bash
DRY_RUN=1 bash scripts/run_transport_constraint_ablation.sh
```

只运行部分模式：

```bash
MODES="full static_delay no_monotonic" \
bash scripts/run_transport_constraint_ablation.sh
```

初筛后对保留模式补跑三个独立预训练种子：

```bash
MODES="full 需要保留的模式" \
PRETRAIN_SEEDS="42 3407 2026" \
bash scripts/run_transport_constraint_ablation.sh
```

## 归档

脚本自动归档到：

```text
paper/ICASSP2027/03_experiments/P2_transport_constraint_ablation/results
```

归档包括命令、Git SHA、权重和划分 SHA256、验证集患者预测、日志末尾、
单次结果、聚合结果及相对 `full` 的配对差值。正式汇总缺少任一模式时
会直接失败，不会生成不完整的论文表格。
