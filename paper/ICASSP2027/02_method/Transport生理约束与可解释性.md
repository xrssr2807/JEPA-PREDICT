# 生理约束动态因果 Transport：方法定义与可解释性边界

## 1. 为什么 ECG 与 PPG 构成有方向的生理耦合

ECG 与 PPG 不是任意的两个传感器通道。

1. ECG 的 QRS 波群反映心室电激动；
2. 电激动之后发生机电耦合、心室射血；
3. 压力/血容量脉搏沿动脉传播到外周；
4. PPG 记录采集位置局部组织血容量随脉搏发生的光学变化。

因此，在同步采集条件下，每次有效心搏具有明确的时间方向：

```text
ECG R peak
    -> pre-ejection period
    -> ventricular ejection
    -> arterial pulse propagation
    -> peripheral PPG foot/upstroke
```

ECG R 峰到 PPG 足点的时间差应称为 pulse arrival time（PAT）代理值。
它同时包含射血前期和血管传播时间，不能直接等同于纯 PTT。

## 2. 模型中的“生理约束”具体在哪里

当前 Transport 不是普通的全局 cross-attention，而是显式限制的带状搬运算子。

### 2.1 正向时间支撑

源 token 为 ECG，目标 token 只能位于源 token 之后：

```math
j = i + d,\qquad d > 0.
```

这排除了 PPG 先于 ECG 的反生理映射。

### 2.2 生理时延范围

配置请求范围为 80–800 ms，并以 250 ms 为软先验。编码器总步长为 16，
采样率为 100 Hz，因此每个 token 对应 160 ms，实际离散支撑为：

```text
160, 320, 480, 640, 800 ms
```

论文必须报告实际 token 分辨率。硬范围只能证明模型遵守先验，不能证明模型
从数据中恢复了真实 PAT；后者必须与波形检测值比较。

### 2.3 顺序保持

若两个相邻 ECG token 的预期目标位置发生逆序，单调性损失会惩罚：

```math
L_mono = ReLU((i + E[d_i]) - (i+1 + E[d_{i+1}])).
```

该约束避免 Transport 路径交叉，符合脉搏传播的时间顺序。

### 2.4 局部平滑

相邻 token 的预测时延受到平滑约束，防止每个 token 独立跳到毫无连续性的
时间位置，同时仍允许时延随局部心动周期和形态变化。

### 2.5 未匹配质量

每个 ECG token 可以将质量送入 unmatched dustbin。运动伪影、低灌注或边界
token 不必被强制匹配到错误的 PPG 位置。

## 3. “动态”具体在哪里

固定时延方法对所有样本使用同一个位移。当前模型则由 delay head 对每个
ECG token 输出一组延迟概率：

```math
p(d\mid z_i^{ECG}),\qquad
E[d_i]=\sum_d p(d\mid z_i^{ECG})d.
```

所以动态性有三个层级：

- token 内：一个 token 对多个未来时延具有软分布；
- 片段内：不同 token 可以具有不同预期时延；
- 患者/片段间：不同输入可以产生不同的时延路径。

“动态”必须通过与 segment-static、token-shuffled、cross-patient delay policy
及固定时延对照共同验证。仅仅代码中存在 delay head，或 token 时延标准差非零，
不能证明输入依赖的动态路径带来实际收益。

## 4. “因果”具体在哪里

本文中的 causal 指：

- 生理机制给定的方向为 ECG→PPG；
- Transport 只能从 ECG 搬运到未来 PPG；
- 负时延、时间反转和错配应破坏表示对齐；
- 移除该方向约束后，下游冠心病判别应下降。

它属于 **causal-direction inductive bias** 或 **causal temporal consistency**。
由于数据是观察性的同步波形，没有干预，不能声称模型完成了 causal discovery。

当前稳妥的论文用语：

> physiology-constrained probabilistic monotonic transport

只有在严格动态对照成立后，才升级为：

> physiology-constrained dynamic causal-direction transport

避免：

> discovers the causal relationship between ECG and PPG

## 5. 可解释性证据链

### 5.1 波形层

- 在 ECG 上标出 R 峰；
- 在 PPG 上标出足点；
- 展示模型 Transport 热图和预期路径；
- 对比模型时延与 PAT。

### 5.2 表示层

比较以下条件的跨模态余弦距离：

- learned dynamic positive delay；
- segment-static delay；
- token-shuffled delay policy；
- cross-patient delay policy；
- fixed 250 ms；
- zero delay；
- negative delay；
- reversed PPG；
- shuffled ECG/PPG pair。

只有学习的正向动态 Transport 稳定优于所有对照，才支持方向性与配对特异性。

### 5.3 任务层

- Transport-on vs Transport-off 的患者级冠心病 AUC；
- 三随机种子均值、标准差及置信区间；
- 检查 Transport 增益是否集中于需要 ECG–PPG 耦合信息的病例。

### 5.4 证据与结论的对应关系

| 证据 | 可以支持 | 不能单独支持 |
|---|---|---|
| 正时延带 + 单调路径 | 生理方向先验被编码 | 模型发现了因果关系 |
| 波形 PAT 对照 | 延迟具有生理数值一致性 | 纯 PTT 或血压机制 |
| segment-static / token-shuffled | token 级动态路径有贡献 | 患者级个体化时延 |
| negative / reversed PPG | 时间方向有贡献 | 干预意义因果效应 |
| cross-patient delay / pair shuffle | 策略或波形配对具有个体特异性 | 冠心病由该时延直接造成 |
| Transport-on/off CHD AUC | 该归纳偏置改善任务表示 | delay head 是分类决策中介 |

## 6. seed 42 开发性验证结果

首版 512 片段运行暴露了 R 峰检测质量问题，PAT 部分已作废。修正 QRS 检测和
质量门控后，严格对照在 512 个片段、457 位患者上重新执行。该结果用于
机制验证，不进入下游性能主表。

| 证据 | 初步结果 | 判断 |
|---|---:|---|
| Transport-on vs off CHD AUC | +0.0142 | 支持任务贡献 |
| 单调违反率 | 0.0000 | 支持顺序约束 |
| matched mass 中位数 | 0.8149 | 未退化到 dustbin |
| 动态 vs segment-static | +0.000041，CI 不跨 0 | token 动态有统计支持，但效应极小 |
| 动态 vs token-shuffled policy | +0.01710，CI 不跨 0 | 正确 token 顺序有贡献 |
| 动态 vs cross-patient policy | +0.000065，CI 不跨 0 | 存在较弱输入特异性 |
| 动态 vs 固定先验 | +0.00482，CI 不跨 0 | 学习策略优于固定先验 |
| 动态 vs 负时延 | +0.01766，CI 不跨 0 | 支持时间方向 |
| 动态 vs PPG 反转 | +0.27057，CI 不跨 0 | 强支持时间顺序 |
| 动态 vs 跨患者 PPG 错配 | +0.08461，CI 不跨 0 | 支持配对特异性 |
| 模型时延与 PAT Spearman | 0.0224 | 不支持生理数值恢复 |
| PAT MAE | 256.2 ms | 不能称为 PAT/PTT 估计 |
| 患者间平均时延 SD | 1.98 ms | 患者级动态性很弱 |
| 片段内 token 时延 SD 中位数 | 30.87 ms | 路径变化主要位于 token 层 |

## 7. 当前可支持的论文结论

当前证据支持：

> Transport encodes a monotonic positive-delay ECG-to-PPG alignment that is
> sensitive to temporal direction and subject pairing, and improves downstream
> CHD discrimination.

当前证据不支持：

> Transport accurately recovers patient-specific physiological PAT/PTT.

严格 segment-static 对照在 457 位患者上统计显著，但效应量很小。因此，
“dynamic”可以作为算子属性和次级机制结论，标题级主价值仍应放在
“生理约束的概率单调 Transport”，而不是“动态 PAT 估计”。可解释性图用于
展示时间方向、顺序保持、拒配质量和扰动敏感性。

## 8. 后续模型优化触发条件

只有在三随机种子 Transport-on/off 完成后才决定是否修改模型。

若完整结果继续满足“方向对照显著、PAT 一致性弱”，下一版可采用：

- segment-level global PAT head；
- token-level local residual delay；
- ECG online token 与 stop-gradient PPG teacher token 共同条件化；
- global PAT + local residual 的分层 Transport；
- 对高质量 R-peak/PPG-foot 片段增加弱监督 PAT 校准损失。

该方案需要重新进行 Phase 2 预训练，不能与当前主结果混用。

## 9. 与近期生理时序跨模态方法的差异

近期工作已经开始利用 ECG 先于外周 PPG 的时间顺序进行跨模态预训练。因此，
“ECG 与 PPG 存在时间先后”本身不能单独构成本文创新。本文应突出 Transport
算子的可检验结构：

1. 延迟不是一个固定常数，而是每个 ECG token 上的条件概率分布；
2. 搬运仅允许进入生理正时延带；
3. 通过单调约束阻止跨心搏的时间路径反转；
4. 通过平滑项保持局部连续性；
5. 通过 unmatched dustbin 拒绝伪影或无可靠对应的 token；
6. 通过固定、零、负时延、时间反转和跨患者错配进行反事实式对照；
7. 通过 Transport-on/off 的下游冠心病 AUC 验证表示贡献。

因此论文中的差异化表述应是：

> We formulate ECG-to-PPG pre-training as a physiology-constrained,
> probabilistic monotonic transport problem, rather than generic temporal
> ordering or unconstrained cross-modal attention.

近期相关工作可纳入 Related Work 和基线讨论：

- xMAE: Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal
  Representation Learning, 2026. <https://arxiv.org/abs/2605.00973>

## 参考生理定义

- PAT 定义参考：MESA Sleep Study，ECG R-wave peak 到 PPG foot。
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8530459/>
- ECG、PCG 与 PPG 的多模态生理指标区分：ECG–PPG 对应 PAT，
  PCG–PPG 更接近去除射血前期后的 PTT。
  <https://pubmed.ncbi.nlm.nih.gov/41228931/>
