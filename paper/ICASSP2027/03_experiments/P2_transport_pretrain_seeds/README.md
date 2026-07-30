# Transport 跨预训练种子配对实验

目的：验证 Transport 的冠心病增益是否能跨独立预训练初始化重复，而不只是同一预训练
checkpoint 上的下游随机波动。

## 冻结设计

- 预训练优化种子：42、3407、2026。
- 预训练患者划分种子固定为 42。
- 每个预训练种子均运行 Transport on/off 配对。
- 下游微调种子固定为 42。
- 下游使用八标签、双通道、SP head off、Patient MIL on、Multi-scale on。
- 只访问冻结验证集，测试集保持封存。
- seed 42 复用已有 on/off 预训练权重；新增训练 seed 3407 和 2026。

## 一键入口

```bash
bash scripts/run_transport_pretrain_seed_study.sh
```

脚本支持断点续跑，并自动输出患者级配对 Bootstrap、seed 级均值与 95% t 区间。
checkpoint 不复制进论文目录，只归档哈希、配置、日志、患者级预测和统计结果。
