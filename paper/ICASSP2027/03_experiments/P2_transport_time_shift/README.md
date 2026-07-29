# P2 Transport 时间平移干预实验

状态：代码已完成，待远程运行。

## 目的

对 PPG 施加已知的非循环时间平移，验证 Phase 2 Transport 是否捕捉了
ECG 与 PPG 之间的时序几何关系，而不仅是在静态特征上完成跨模态预测。

主实验使用验证集，测试集继续封存。患者作为统计单位。

## 关键事实与正确口径

当前延迟策略 `phase2_delay_head` 只以 ECG token 为输入。因此，平移 PPG
不会也不应该改变延迟头本身的输出。

本实验验证三个可由现有架构支持的命题：

1. PPG 偏离观测时序后，原始 ECG 条件 Transport 下的对齐损失增加。
2. 将 Transport 目标列按已知干预量同步平移后，对齐损失得到恢复。
3. 在候选补偿量上搜索最小损失，可以恢复注入位移的方向和近似幅度。

不能据此声称延迟头观察了 PPG、恢复了物理 PAT/PTT，或完成了因果发现。

## 两种干预域

- `teacher_tokens`：直接平移 PPG teacher token。用于隔离验证 Transport
  的时序几何，是主要结果。
- `waveform`：在编码前平移原始 PPG 波形。更接近输入扰动，但同时包含
  编码器边界与上下文效应，作为补充结果。

所有平移均采用零填充和有效区域掩码，禁止循环回绕。

## 输出

每种干预域输出：

- `segment_shift_metrics.csv`
- `segment_shift_profiles.csv`
- `patient_shift_metrics.csv`
- `transport_time_shift_summary.json`
- `transport_time_shift_report.md`
- `shift_sensitivity.png`
- `shift_recovery.png`
- `shift_compensation.png`
- `command.txt`
- `console.log`

正式判断以患者级均值、95% bootstrap CI、位移恢复斜率和 MAE 为主。
