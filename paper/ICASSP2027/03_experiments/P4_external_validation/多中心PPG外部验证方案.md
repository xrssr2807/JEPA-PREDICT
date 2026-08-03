# 多中心 PPG 外部验证方案

## 目的

在完全独立的前瞻性多中心队列上评估冻结的 PPG 单通道模型，检验模型的跨中心泛化能力。外部队列只用于一次性评估，不用于训练、模型选择、早停或阈值搜索。

## 队列与处理结果

- 原始元数据患者：197 人。
- 诊断表：165 人。
- 安全匹配：157 人，其中精确匹配 154 人、唯一 `HR` 前缀归一化匹配 3 人。
- 未匹配：8 人，不进行模糊猜测。
- 有诊断文本且存在有效 PPG 窗口：132 人。
- 模型输入：1021 个 10 秒 PPG 窗口，每位患者最多 8 个窗口。
- 分析单位：患者；片段由 Patient-MIL 聚合。

阳性患者数：高血压 101、高血糖 0、高血脂 22、其他疾病 56、冠心病 43、心律失常 22、糖尿病 58、颈动脉斑块 5。由于高血糖没有阳性患者，其 AUROC 不定义，不得用 0.5 代替并纳入 Macro AUROC。

## 固定模型

- 输入：PPG 单通道。
- 预训练：Phase 2。
- 下游配置：Shared-Private head 关闭、Patient-MIL 开启、多尺度开启。
- 初始模型：seed 42；不得根据外部结果改选随机种子。
- 分类阈值：必须读取内部验证集确定并保存在下游 checkpoint 中的阈值。

## 数据安全

- 原始患者 ID 和诊断文本仅保存在本地 `private/` 目录。
- 上传服务器的公开包只包含匿名 UID、模型窗口、匿名标签表、窗口清单和数据摘要。
- `private/`、原始 Excel、原始 ZIP/TXT 不上传 GitHub，也不进入论文归档。

## 运行

```bash
PRETRAIN_CHECKPOINT=outputs_phase2_shared_private_seed42/jepa_best.pt \
DOWNSTREAM_CHECKPOINT=outputs_spv2_off_ppg_seed42/downstream_multidisease_best.pt \
EXTERNAL_DATA_DIR=/root/autodl-tmp/multicenter_external_model_ready/model_input \
bash scripts/run_multicenter_external_validation.sh
```

脚本会拒绝 ECG/双通道模式、缺少内部阈值的下游权重以及标签有效性不完整的患者。输出包括匿名患者级预测、每病种 AUROC/AUPRC/Brier/ECE、冠心病 ROC、PR 和校准图，以及输入文件哈希。

## 解释边界

诊断标签来自结构化病历表的规则映射，正式投稿前需由临床人员复核规则和 8 个未匹配 ID。该队列不含 ECG，因此只能验证 PPG 部署分支，不能直接验证双通道融合增益或 ECG 到 PPG Transport 的机制效果。
