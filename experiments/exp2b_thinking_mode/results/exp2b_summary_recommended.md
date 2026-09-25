# Experiment 2, Tier 2 — Thinking ON (recommended sampling)

| Run | n | failed | chrF++ | COMET |
|---|---|---|---|---|
| Qwen3.5-9B C0 think-recommended | 997 | 116 | 44.78 | 0.8052 |
| Qwen3.5-9B C7 think-recommended | 997 | 131 | 44.09 | 0.7914 |
| Qwen3.5-4B C0 think-recommended | 997 | 79 | 44.33 | 0.8078 |
| Qwen3.5-4B C7 think-recommended | 997 | 467 | 28.46 | 0.6171 |
| Qwen3.5-27B C0 think-recommended | 997 | 11 | 52.84 | 0.8711 |
| Qwen3.5-27B C7 think-recommended | 997 | 5 | 51.56 | 0.8670 |

## Δ (gain over first run (see per-model table below))

| Run | Δ chrF++ | Δ COMET |
|---|---|---|
| Qwen3.5-9B C0 think-recommended | +0.00 | +0.0000 |
| Qwen3.5-9B C7 think-recommended | -0.69 | -0.0138 |
| Qwen3.5-4B C0 think-recommended | -0.45 | +0.0026 |
| Qwen3.5-4B C7 think-recommended | -16.32 | -0.1881 |
| Qwen3.5-27B C0 think-recommended | +8.06 | +0.0659 |
| Qwen3.5-27B C7 think-recommended | +6.78 | +0.0618 |

## Thinking-mode behavior (thinking ON runs)

| Run | answered % | length-capped | reasoning loops | answer loops | reasoning chars (mean/p95/max) |
|---|---|---|---|---|---|
| qwen3.5-9b_c0_baseline_think-recommended | 88.4% | 116 | 651 | 1 | 23953 / 50162 / 65950 |
| qwen3.5-9b_c7_full_think-recommended | 86.9% | 131 | 904 | 0 | 25499 / 49105 / 66251 |
| qwen3.5-4b_c0_baseline_think-recommended | 92.1% | 79 | 432 | 0 | 24323 / 47041 / 63270 |
| qwen3.5-4b_c7_full_think-recommended | 53.2% | 467 | 929 | 0 | 34661 / 56657 / 74124 |
| qwen3.5-27b_c0_baseline_think-recommended | 98.9% | 11 | 688 | 0 | 18284 / 27966 / 60992 |
| qwen3.5-27b_c7_full_think-recommended | 99.5% | 5 | 783 | 0 | 19058 / 27509 / 52952 |

## Δ thinking ON − thinking OFF (per model × condition)

| Run | chrF++ OFF→ON | Δ chrF++ | COMET OFF→ON | Δ COMET |
|---|---|---|---|---|
| Qwen3.5-9B C0 | 47.65 → 44.78 | -2.87 | 0.8543 → 0.8052 | -0.0491 |
| Qwen3.5-9B C7 | 47.71 → 44.09 | -3.62 | 0.8513 → 0.7914 | -0.0599 |
| Qwen3.5-4B C0 | 44.58 → 44.33 | -0.25 | 0.8355 → 0.8078 | -0.0277 |
| Qwen3.5-4B C7 | 45.48 → 28.46 | -17.02 | 0.8382 → 0.6171 | -0.2211 |
| Qwen3.5-27B C0 | 51.03 → 52.84 | +1.81 | 0.8723 → 0.8711 | -0.0012 |
| Qwen3.5-27B C7 | 51.95 → 51.56 | -0.39 | 0.8706 → 0.8670 | -0.0036 |
