# XAI-01 严格 Transport 验证：seed 42

运行日期：2026-07-29

数据：冻结验证集随机覆盖 512 个片段、457 位患者

统计：先按患者聚合，再进行 2,000 次患者级 bootstrap

测试集：封存，未访问

## 主要结果

| 问题 | 结果 | 判断 |
|---|---:|---|
| 单调违反率中位数 | 0.0000 | 路径满足顺序约束 |
| matched mass 中位数 | 0.8149 | 未退化到 dustbin |
| segment-static − dynamic | +0.000041，95% CI [0.000010, 0.000081] | token 动态有统计支持，但实际效应很小 |
| token-shuffled − dynamic | +0.01710，95% CI [0.01683, 0.01735] | 正确 token 时延顺序重要 |
| cross-patient policy − dynamic | +0.000065，95% CI [0.000027, 0.000113] | 输入特异性存在但较弱 |
| fixed prior − dynamic | +0.00482，95% CI [0.00406, 0.00566] | 学习策略优于固定先验 |
| zero delay − dynamic | +0.00870，95% CI [0.00759, 0.00990] | 正时延有贡献 |
| negative delay − dynamic | +0.01766，95% CI [0.01635, 0.01893] | 生理时间方向有贡献 |
| reversed PPG − dynamic | +0.27057，95% CI [0.26142, 0.28038] | 强支持时间顺序 |
| shuffled pair − dynamic | +0.08461，95% CI [0.07515, 0.09404] | 强支持跨模态配对特异性 |

## 生理数值一致性

质量门控后共有 430 个片段、388 位患者具有可用 PAT 代理值。模型平均时延约
357.8 ms，而波形 PAT 中位数为 85 ms。二者 Spearman 相关为 0.0224，MAE
为 256.2 ms。

因此，当前 delay head 不能解释为 PAT/PTT 估计器。模型时延应理解为编码器
token 空间中的最优跨模态搬运位置，而不是未经校准的物理传播时间。

## 冠心病关联

冠心病阳性与阴性患者的平均 Transport 时延差为 -0.45 ms，95% CI
[-0.99, 0.13]，没有显著关联。该结果不否定 Transport 对冠心病分类的贡献，
因为下游分类器使用预训练编码器，而不直接消费 delay head。

## 论文结论

当前结果支持：

> physiology-constrained dynamic probabilistic monotonic transport

其中“dynamic”的证据存在但效应较小；更强的证据来自方向反转、时间反转、
token 策略打乱和跨患者错配。论文不能声称 causal discovery，也不能声称
准确恢复患者级 PAT/PTT。

任务层已有 seed 42 Transport-on/off 结果：完整 Phase 2 相比 Transport-off
的冠心病 AUC 增加 0.0142、宏 F1 增加 0.0145。仍需补齐 seed 3407 和 2026
的相同消融，才能把任务增益写入正式主结论。
