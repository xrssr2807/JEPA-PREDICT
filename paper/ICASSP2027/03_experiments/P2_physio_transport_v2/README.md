# 生理约束概率 Transport v2

## 目的

旧版 Transport 的延迟策略只读取 ECG token，因而“配对 PPG 波形改变时，路径应随之改变”的证据较弱。v2 将对齐代价改为同时依赖 ECG 与 PPG，并保留严格正延迟带和未匹配 dustbin。

## 模型变化

1. 双端内容条件：每个候选 ECG/PPG token 对都有内容相似度。
2. 全局与局部延迟：片段级延迟分布和逐 token 残差共同决定路径。
3. 非平衡 Sinkhorn：允许异常、缺搏和边界 token 进入 dustbin。
4. 反事实排序：正确患者配对的对齐分数必须高于批内错配。
5. 可选 PAT 弱监督：仅对高置信 PAT 目标启用，默认权重为 0。

旧 `full` 模式和既有消融保持不变，`physio_v2` 是独立模式。

## 推荐启动

```bash
cd /root/autodl-tmp/JEPA-PREDICT-priority1
conda activate JEPA

nohup bash scripts/run_phase2_physio_v2.sh \
  > logs/phase2_physio_v2_seed42.log 2>&1 &
```

默认从 `outputs_phase2_shared_private_seed42/jepa_best.pt` 迁移已有编码器与 Shared-Private 参数，仅随机初始化 v2 Transport 头，并以较小学习率训练 40 个 epoch。

## 论文判定标准

- 首要：固定患者划分下，双通道 CHD AUC 不低于旧 Phase 2。
- 机制：波形时间平移恢复斜率和正确配对识别率高于旧版。
- 稳健：Transport-on/off 独立预训练种子差异方向一致。
- 不把模型延迟直接称为 PAT，除非外部 PAT 相关性和误差达到预设标准。
