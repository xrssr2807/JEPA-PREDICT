# PPG专属继续自监督适配

## 动机

双通道教师蒸馏和直接形态残差均未提高 PPG-only CHD AUC。前者存在跨模态
负迁移，后者将粗粒度统计直接注入分类头，破坏了排序能力。本实验改为在冻结
患者划分的训练集 PPG 上继续自监督适配编码器，辅助头训练后全部丢弃。

## 训练目标

1. 学生编码器接收块遮挡、增益扰动和轻微噪声后的 PPG；
2. EMA 教师接收干净 PPG，约束 token 语义不漂移；
3. 学生 token 重建有顺序的局部波形和一/二阶导数包络；
4. pooled 表征预测四个 PPG 频带能量；
5. 仅保存适配后的 PPG encoder，不把辅助头带入下游。

## 协议

- 只读取固定 split 中的 train 文件；
- 不读取 val/test 波形开展自监督训练；
- baseline 与 adapted 使用相同下游头、MIL、多尺度和 seed42；
- 下游测试集保持封存；
- CHD AUC 增益至少 `0.005` 且 Macro AUC 下降不超过 `0.005`，才补三种子。

## 运行

```bash
nohup bash scripts/run_ppg_continued_ssl.sh \
  > logs/ppg_continued_ssl_seed42.log 2>&1 &
```
