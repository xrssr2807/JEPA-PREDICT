# Phase 3A 下游反馈预训练

Phase 3A 可从随机初始化开始，也可在完整 Phase 2 权重上继续训练。ECG/PPG 双在线编码器同时被 JEPA 和患者级多疾病任务共享；EMA 教师仍只由动量更新。默认从第 5 个 epoch 开始，每 20 个 JEPA 优化步插入 1 个监督反馈步。

## 1. 生成四路患者划分

```bash
cd ~/autodl-tmp/JEPA-PREDICT

python generate_multidisease_patient_split.py \
  --data_dir /root/ppgchd/ppgchd/data_updated \
  --taskaware \
  --feedback_train_ratio 0.55 \
  --feedback_meta_ratio 0.15 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --workers 8 \
  --representative_only \
  --output splits/multidisease_taskaware_split.json
```

四个集合按 UID 严格互斥：

- `feedback_train`：训练 Patient-MIL 反馈头，并在暖身结束后向共享编码器提供受限梯度。
- `feedback_meta`：监控下游反馈质量，不参与梯度更新。
- `val`：选择 `jepa_taskaware_best.pt`。
- `test`：训练期间只验证清单完整性，不创建 DataLoader、不计算指标。

## 2. 从头启动完整训练

```bash
mkdir -p outputs_taskaware_scratch

python -u train_taskaware_pretrain.py \
  --from_scratch \
  --split splits/multidisease_taskaware_split.json \
  --output_dir outputs_taskaware_scratch \
  --epochs 80 \
  --pretrain_batch_size 128 \
  --feedback_batch_size 8 \
  --feedback_segments 4 \
  --feedback_start_epoch 5 \
  --feedback_interval 20 \
  --head_warmup_steps 50 \
  --feedback_grad_ratio 0.20 \
  --workers 8 \
  --seed 42 \
  2>&1 | tee outputs_taskaware_scratch/console.log
```

前 10 个 epoch 使用 direct token JEPA；随后 transport 在 20 个 epoch 内逐渐升至完整权重。下游反馈从第 5 个 epoch 开始，反馈头暖身 50 次后才允许监督梯度进入在线编码器。

## 3. 从 Phase 2 权重继续

```bash
mkdir -p outputs_taskaware

python -u train_taskaware_pretrain.py \
  --checkpoint outputs_phase2/jepa_best.pt \
  --split splits/multidisease_taskaware_split.json \
  --output_dir outputs_taskaware \
  --epochs 30 \
  --pretrain_batch_size 128 \
  --feedback_batch_size 8 \
  --feedback_segments 4 \
  --feedback_interval 20 \
  --head_warmup_steps 50 \
  --feedback_grad_ratio 0.20 \
  --workers 8 \
  --seed 42 \
  2>&1 | tee outputs_taskaware/console.log
```

24 GB 显存不足时，先将 `--pretrain_batch_size` 降到 96 或 64，再将 `--feedback_batch_size` 降到 4。不要先减少 `feedback_segments`，患者级多片段信息对 CHD 更重要。

## 4. 恢复训练

```bash
python -u train_taskaware_pretrain.py \
  --checkpoint outputs_phase2/jepa_best.pt \
  --resume outputs_taskaware/jepa_taskaware_last.pt \
  --split splits/multidisease_taskaware_split.json \
  --output_dir outputs_taskaware \
  --epochs 30 \
  --workers 8 \
  --seed 42
```

`--epochs` 是总 epoch 数，不是额外 epoch 数。

## 5. 用最佳权重做常规下游微调

任务感知 checkpoint 保留 `context_encoder`、`ppg_encoder`、`target_encoder` 和 `model_state_dict`，兼容原下游入口：

```bash
python -u train_downstream.py \
  --checkpoint outputs_taskaware/jepa_taskaware_best.pt \
  --dataset multidisease \
  --multidisease_channel both \
  --multidisease_split splits/multidisease_patient_split.json \
  --output_dir outputs_multidisease_taskaware \
  --mil_batch_size 32 \
  --mil_chunk_size 64 \
  --workers 8 \
  --seed 42
```

正式对比实验必须让 baseline 与 Phase 3A 使用同一份下游患者划分和同一随机种子。报告 CHD AUC、macro AUC、CHD Precision/Recall/F1，并至少运行 3 个随机种子。
