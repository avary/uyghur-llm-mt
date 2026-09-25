# Experiment 1 — Component Ablation (Qwen/Qwen3.5-9B, thinking off)

| Run | n | failed | chrF++ | COMET |
|---|---|---|---|---|
| c0_baseline_qwen3.5-9b.results.jsonl | 997 | 0 | 47.63 | 0.8544 |
| c1_syntax_qwen3.5-9b.results.jsonl | 997 | 0 | 48.05 | 0.8537 |
| c2_morph_qwen3.5-9b.results.jsonl | 997 | 0 | 48.18 | 0.8535 |
| c3_dict_qwen3.5-9b.results.jsonl | 997 | 0 | 48.34 | 0.8568 |
| c4_syntax_morph_qwen3.5-9b.results.jsonl | 997 | 0 | 47.82 | 0.8517 |
| c5_syntax_dict_qwen3.5-9b.results.jsonl | 997 | 0 | 47.86 | 0.8529 |
| c6_morph_dict_qwen3.5-9b.results.jsonl | 997 | 0 | 47.94 | 0.8536 |
| c7_full_qwen3.5-9b.results.jsonl | 997 | 0 | 47.71 | 0.8513 |

## Δ (gain over C0 baseline)

| Run | Δ chrF++ | Δ COMET |
|---|---|---|
| c0_baseline_qwen3.5-9b.results.jsonl | +0.00 | +0.0000 |
| c1_syntax_qwen3.5-9b.results.jsonl | +0.42 | -0.0007 |
| c2_morph_qwen3.5-9b.results.jsonl | +0.55 | -0.0009 |
| c3_dict_qwen3.5-9b.results.jsonl | +0.71 | +0.0024 |
| c4_syntax_morph_qwen3.5-9b.results.jsonl | +0.19 | -0.0027 |
| c5_syntax_dict_qwen3.5-9b.results.jsonl | +0.23 | -0.0015 |
| c6_morph_dict_qwen3.5-9b.results.jsonl | +0.31 | -0.0008 |
| c7_full_qwen3.5-9b.results.jsonl | +0.08 | -0.0031 |
