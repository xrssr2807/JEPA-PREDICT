# Physiological Causal Transport Interpretability Report

## Scope

- Split: frozen development validation set; test set remains sealed.
- Segments/patients: 512/457.
- PAT quality-controlled segments: 430.
- Effective token delay support: 160, 320, 480, 640, 800 ms.
- Physiological reference: ECG R-peak to PPG foot PAT proxy. This includes pre-ejection period and must not be called pure PTT.
- Claim boundary: the controls test physiological temporal-direction consistency, not causal discovery from observational data.

## Physiological agreement

- Patients with valid PAT: 388.
- Model delay mean/median: 357.8 / 357.5 ms.
- PAT mean/median: 116.7 / 85.0 ms.
- Spearman(model delay, PAT): 0.0224
- MAE: 256.2 ms.
- Bias and Bland-Altman limits: 241.1 ms [47.8, 434.3].

## Dynamic and monotonic behavior

- Between-patient SD of patient mean delay: 2.0 ms.
- Median within-patient segment SD: 0.9 ms.
- Median within-segment token SD: 30.9 ms.
- Median monotonic violation rate: 0.0000

## Temporal and pairing controls

Positive values below mean that the control has higher cosine distance than learned dynamic causal transport.

| Control | Control - dynamic | 95% bootstrap CI |
|---|---:|---:|
| segment_static_delay | 0.0000 | [0.0000, 0.0001] |
| token_shuffled_delay | 0.0171 | [0.0168, 0.0174] |
| cross_patient_delay_policy | 0.0001 | [0.0000, 0.0001] |
| fixed_prior | 0.0048 | [0.0041, 0.0057] |
| zero_delay | 0.0087 | [0.0076, 0.0099] |
| negative_delay | 0.0177 | [0.0163, 0.0189] |
| reversed_ppg | 0.2706 | [0.2614, 0.2804] |
| shuffled_pair | 0.0846 | [0.0752, 0.0940] |

## CHD association

- CHD positive/negative mean delay: 357.4 / 357.9 ms.
- Difference (positive - negative): -0.5 ms [-1.0, 0.1].
- This is an association analysis. The downstream classifier uses the pretrained encoders, not the delay head itself.

## Paper decision rule

The Transport claim is supported only if all of the following hold:

1. learned delay has non-trivial agreement with waveform-derived PAT;
2. dynamic transport beats fixed-delay and zero-delay controls;
3. negative-delay, reversed-PPG and shuffled-pair controls are worse;
4. monotonic violations remain near zero with non-degenerate matched mass;
5. the downstream Transport-on advantage is reproduced across seeds.
