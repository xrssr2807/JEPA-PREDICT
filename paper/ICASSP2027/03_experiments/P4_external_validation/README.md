# 外部验证与临床评价协议

## 必要输入

外部数据必须先按患者生成一行预测，字段与训练代码导出的 CSV 一致：

```text
uid,split,label::<疾病>,prob::<疾病>,pred::<疾病>
```

模型与 CNN 基线必须使用完全相同的 UID 和真实标签。数据只能按患者划分，不能把同一患者片段放入不同集合。

## 自动输出

`scripts/evaluate_clinical_predictions.py` 生成：

- 每疾病 AUROC 及 DeLong 95% CI；
- AUPRC、Brier score、10-bin ECE；
- 校准截距和斜率；
- 冠心病 ROC、PR 和校准图；
- 与参考模型的患者级配对 DeLong 检验。

示例：

```bash
python scripts/evaluate_clinical_predictions.py \
  --predictions jepa=external_jepa_predictions.csv \
  --predictions cnn=external_cnn_predictions.csv \
  --reference jepa \
  --focus_label 冠心病 \
  --output_dir paper/ICASSP2027/03_experiments/P4_external_validation/results
```

外部验证结果在真实外部数据和标签完成适配前不能预填，也不能用内部验证集替代。
