# Experiment 2, Tier 2 — Thinking ON (greedy sampling)

| Run | n | failed | chrF++ | COMET |
|---|---|---|---|---|
| Qwen3.5-9B C0 think-greedy | 200 | 47 | 41.28 | 0.7439 |
| Qwen3.5-9B C7 think-greedy | 200 | 126 | 21.24 | 0.5428 |

## Δ (gain over first run (see per-model table below))

| Run | Δ chrF++ | Δ COMET |
|---|---|---|
| Qwen3.5-9B C0 think-greedy | +0.00 | +0.0000 |
| Qwen3.5-9B C7 think-greedy | -20.04 | -0.2011 |

## Thinking-mode behavior (thinking ON runs)

| Run | answered % | length-capped | reasoning loops | answer loops | reasoning chars (mean/p95/max) |
|---|---|---|---|---|---|
| qwen3.5-9b_c0_baseline_think-greedy_n200 | 76.5% | 47 | 137 | 0 | 27772 / 56530 / 65957 |
| qwen3.5-9b_c7_full_think-greedy_n200 | 37.0% | 126 | 198 | 0 | 38510 / 58100 / 70820 |

## Δ thinking ON − thinking OFF (per model × condition)

| Run | chrF++ OFF→ON | Δ chrF++ | COMET OFF→ON | Δ COMET |
|---|---|---|---|---|
| Qwen3.5-9B C0 | 47.65 → 41.28 | -6.37 | 0.8543 → 0.7439 | -0.1104 |
| Qwen3.5-9B C7 | 47.71 → 21.24 | -26.47 | 0.8513 → 0.5428 | -0.3085 |

## Matched-sample correction (thinking OFF scored on the same 200 sentences)

| Run | chrF++ OFF (n=200) | chrF++ ON (n=200) | Δ chrF++ |
|---|---|---|---|
| Qwen3.5-9B C0 | 46.69 | 41.28 | -5.41 |
| Qwen3.5-9B C7 | 45.45 | 21.24 | -24.21 |

Note: OFF-side COMET on the matched subset pending (GPU shared with the
recommended-arm generation at scoring time). The `Δ thinking ON − OFF` table
above used Tier-1 full-997 OFF scores; use this table for the greedy arm.

Matched-sample OFF scores (same 200 sentences): C0 chrF++ 46.69 / COMET 0.8488;
C7 chrF++ 45.45 / COMET 0.8422. Matched greedy-arm deltas therefore:

| Run | Δ chrF++ | Δ COMET |
|---|---|---|
| Qwen3.5-9B C0 | 46.69 → 41.28 (−5.41) | 0.8488 → 0.7439 (−0.1049) |
| Qwen3.5-9B C7 | 45.45 → 21.24 (−24.21) | 0.8422 → 0.5428 (−0.2994) |
