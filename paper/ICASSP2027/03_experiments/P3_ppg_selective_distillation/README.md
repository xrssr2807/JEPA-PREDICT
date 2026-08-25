# PPG单通道选择性跨模态蒸馏

## 目的

在部署阶段只输入PPG的前提下，利用训练期ECG+PPG双通道教师补充冠心病相关信息，验证是否能稳定提高PPG单通道CHD AUC。

## 初筛矩阵

| 方案 | Logit KD | 可靠性门控 | 患者表征KD | 患者关系KD |
|---|---:|---:|---:|---:|
| baseline | 否 | 否 | 否 | 否 |
| logit | 是 | 否 | 否 | 否 |
| selective | 是 | 标签一致性 | 是 | 否 |
| selective_relation | 是 | 标签一致性 | 是 | 是 |

所有实验固定八标签患者级互斥划分、PhysioV2初始化、Patient-MIL、多尺度头、下游seed 42，并封存测试集。

## 决策规则

1. 主要指标为验证集CHD AUC，Macro AUC为约束指标。
2. 方案相对baseline的CHD AUC必须提高，且Macro AUC不能明显下降。
3. 只有满足前两条的方案才进入seed 3407和2026复现。
4. 本阶段禁止依据历史测试集结果调参，也不生成最终测试结论。

## 运行

```bash
nohup bash scripts/run_selective_dual_teacher_ppg.sh \
  > logs/selective_dual_teacher_ppg_seed42.log 2>&1 &
```

脚本会先训练同划分、测试集封存的双通道教师，再依次运行四组PPG学生，并自动归档验证集汇总。
