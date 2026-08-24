# Frozen PPG representation comparison

Curves use the mean patient probability across three downstream seeds. Confidence intervals use paired patient-level bootstrap on the development set; the test set remains sealed.

Two-sided p-values are exploratory and uncorrected for five comparisons.

| Comparator | PhysioV2 AUC | Comparator AUC | Delta | 95% CI | p (two-sided) |
|---|---:|---:|---:|---:|---:|
| MOMENT-small | 0.7706 | 0.7218 | +0.0487 | [+0.0070, +0.0919] | 0.0176 |
| UniTS-x128 | 0.7706 | 0.7208 | +0.0497 | [+0.0101, +0.0912] | 0.0128 |
| NormWear | 0.7706 | 0.7175 | +0.0531 | [+0.0146, +0.0944] | 0.005199 |
| PaPaGei-S | 0.7706 | 0.7060 | +0.0646 | [+0.0223, +0.1090] | 0.003599 |
| Pulse-PPG | 0.7706 | 0.7039 | +0.0667 | [+0.0257, +0.1083] | 0.0012 |
