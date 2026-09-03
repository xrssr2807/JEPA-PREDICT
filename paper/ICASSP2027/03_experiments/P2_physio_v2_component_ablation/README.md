# PhysioV2-v2 component ablation

This validation-only matrix isolates the current PhysioV2 Transport design:

- `full`: paired-content dynamic unbalanced monotonic Transport.
- `no_content`: removes the explicit ECG-PPG content score.
- `global_only`: removes token-local delay residuals.
- `fixed_delay`: uses the fixed physiological delay prior.
- `hard_delay`: replaces the soft delay distribution with argmax assignment.
- `no_monotonic`: removes the monotonic regularizer.
- `no_smoothness`: removes local delay smoothness.
- `no_dustbin`: forces every matchable ECG token to align without rejection.
- `no_counterfactual`: removes paired-versus-mismatched ranking.

All variants use the same parent checkpoint, frozen patient split, downstream
Patient-MIL head, multiscale setting, and sealed-test protocol. Seed 42 is a
screening matrix. The two largest CHD drops automatically advance to seeds
3407/2026; each seed also includes its matching full-model reference.

Run:

```bash
nohup bash scripts/run_physio_v2_evidence_pipeline.sh \
  > logs/physio_v2_evidence_pipeline.log 2>&1 &
```

Large ablation checkpoints are deleted only after metrics, hashes, logs, and
patient-level validation predictions have been captured in the paper archive.
Deletion is path-confined to the dedicated ablation output directory; source
parents and the three existing full-model checkpoints are never pruned.
