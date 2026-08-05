# P0 预训练目标公平对照

## 目的

比较 PhysioV2 与普通多模态 MAE、对称跨模态 InfoNCE、xMAE 训练目标，回答
PhysioV2 是否优于主流 ECG/PPG 跨模态预训练方式。

## 公平性约束

- 所有方法使用相同的预训练数据及固定患者级预训练划分种子 42。
- 所有方法使用同一套 ECG/PPG `SignalEncoder`，编码器参数量一致。
- 预训练默认 80 epochs、batch 128、梯度累积 3、相同优化步预算。
- 下游统一使用八标签患者级划分
  `e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716`。
- 下游统一双通道、Patient-MIL、多尺度头、SP-off。
- 验证集负责选模，测试集保持封存。

## 方法定义

| 方法 | 训练目标 |
|---|---|
| PhysioV2 | 生理有向概率单调 partial/unbalanced Transport-JEPA |
| Multimodal MAE | ECG 与 PPG 对称随机遮挡，双向交叉注意力重建两种波形 |
| Contrastive | 配对 ECG/PPG 的对称 InfoNCE |
| xMAE | PPG 完全可见，连续遮挡 ECG，PPG→ECG 定向重建遮挡区域 |

xMAE 对照是容量控制的 objective-level reproduction：核心遮挡与定向重建原则来自
官方 xMAE，但编码器替换为本项目统一骨干，目的是隔离训练目标而非比较模型规模。
实现会在进入 CNN 前遮挡原始连续 ECG 区域，并在 token 层再次替换为 mask token，
避免卷积前端直接看到被遮挡波形。

参考：<https://arxiv.org/abs/2605.00973>；官方代码：<https://github.com/hzhou3/xMAE>。

## 执行顺序

先执行 seed 42 初筛。确认所有检查点、验证集预测和归档文件完整后，再决定是否将
四种方法全部扩展到 seed 3407/2026。正式汇总缺少任何一组时会直接失败。

入口：`scripts/run_p0_pretraining_objectives.sh`
