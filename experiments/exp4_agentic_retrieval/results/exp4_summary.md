
## Experiment 4 — agentic retrieval: results

### Quality

| Model | Cond | chrF++ | COMET |
|---|---|---|---|
| Qwen3.5-9B | C0 | 47.63 | 0.8544 |
| Qwen3.5-9B | C3L | 48.32 | 0.8575 |
| Qwen3.5-9B | E4-A | 43.56 | 0.8216 |
| Qwen3.5-9B | E4-B | 43.69 | 0.8322 |
| Gemma-4-E4B | C0 | 25.71 | 0.6341 |
| Gemma-4-E4B | C3L | 42.86 | 0.8078 |
| Gemma-4-E4B | E4-A | 38.12 | 0.7731 |
| Gemma-4-E4B | E4-B | 40.56 | 0.7876 |
| Ministral-3-8B | C0 | 16.70 | 0.4904 |
| Ministral-3-8B | C3L | 29.60 | 0.6863 |
| Ministral-3-8B | E4-A | 18.95 | 0.4929 |
| Ministral-3-8B | E4-B | 16.82 | 0.4810 |
| Llama-3.1-8B | C0 | 32.04 | 0.6403 |
| Llama-3.1-8B | C3L | 37.06 | 0.7078 |
| Llama-3.1-8B | E4-A | 31.01 | 0.6103 |
| Llama-3.1-8B | E4-B | 31.73 | 0.6252 |

### Contrasts (paired bootstrap, chrF++)

| Model | Contrast | Δ [95% CI] | p |
|---|---|---|---|
| Qwen3.5-9B | E4-A − C0 (agency vs nothing) | -4.07 [-5.27, -2.91] ✱ | 0.0000 |
| Qwen3.5-9B | E4-A − C3L (pull vs best push) | -4.76 [-5.89, -3.67] ✱ | 0.0000 |
| Qwen3.5-9B | E4-A − E4-B (offering more tools) | -0.13 [-1.57, +1.32] | 0.8560 |
| Gemma-4-E4B | E4-A − C0 (agency vs nothing) | +12.40 [+11.42, +13.36] ✱ | 0.0000 |
| Gemma-4-E4B | E4-A − C3L (pull vs best push) | -4.74 [-5.60, -3.91] ✱ | 0.0000 |
| Gemma-4-E4B | E4-A − E4-B (offering more tools) | -2.45 [-3.24, -1.63] ✱ | 0.0000 |
| Ministral-3-8B | E4-A − C0 (agency vs nothing) | +2.26 [+1.90, +2.60] ✱ | 0.0000 |
| Ministral-3-8B | E4-A − C3L (pull vs best push) | -10.65 [-11.32, -9.99] ✱ | 0.0000 |
| Ministral-3-8B | E4-A − E4-B (offering more tools) | +2.14 [+1.81, +2.47] ✱ | 0.0000 |
| Llama-3.1-8B | E4-A − C0 (agency vs nothing) | -1.03 [-2.00, -0.09] ✱ | 0.0328 |
| Llama-3.1-8B | E4-A − C3L (pull vs best push) | -6.04 [-6.88, -5.23] ✱ | 0.0000 |
| Llama-3.1-8B | E4-A − E4-B (offering more tools) | -0.71 [-1.57, +0.14] | 0.0948 |

### Agent behavior

| Model | Cond | ep w/ calls | calls/ep | rounds/ep | budget-forced | parse-fail | skipped | unresolved | tokens in/out |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | E4-A | 408/997 | 7.8 | 2.7 | 145 | 0 | 334 | 34 | 5524/287 |
| Qwen3.5-9B | E4-B | 397/997 | 4.7 | 1.9 | 8 | 0 | 27 | 23 | 1952/182 |
| Gemma-4-E4B | E4-A | 184/997 | 1.5 | 2.5 | 0 | 0 | 0 | 0 | 3103/70 |
| Gemma-4-E4B | E4-B | 2/997 | 0.0 | 1.0 | 0 | 0 | 0 | 0 | 443/28 |
| Ministral-3-8B | E4-A | 997/997 | 22.8 | 5.0 | 751 | 0 | 2645 | 563 | 10147/721 |
| Ministral-3-8B | E4-B | 997/997 | 19.4 | 4.7 | 340 | 68 | 1706 | 590 | 6174/783 |
| Llama-3.1-8B | E4-A | 319/997 | 0.3 | 3.5 | 0 | 2210 | 3504 | 6 | 8404/1368 |
| Llama-3.1-8B | E4-B | 241/997 | 0.2 | 3.4 | 0 | 2129 | 2561 | 4 | 6108/1287 |

Tool mix (E4-A):
  - Qwen3.5-9B: {'get_word_list': 391, 'get_gloss': 5712, 'get_lemma': 1112, 'get_near_matches': 321, 'get_features': 196, 'get_syntax': 67}  (addressing: {'token_id': 6915, 'word': 426})
    miss-path: {'explored_lemma': 0.733, 'accepted_unknown': 0.141, 'explored_near': 0.098, 'n_misses': 1363}
  - Gemma-4-E4B: {'get_word_list': 183, 'get_gloss': 1266, 'get_syntax': 5, 'get_near_matches': 6, 'get_lemma': 9}  (addressing: {'token_id': 1279, 'word': 2})
    miss-path: {'accepted_unknown': 0.949, 'explored_lemma': 0.033, 'explored_near': 0.015, 'n_misses': 274}
  - Ministral-3-8B: {'get_word_list': 998, 'get_gloss': 14367, 'get_lemma': 4178, 'get_features': 1383, 'get_syntax': 572, 'get_near_matches': 1234}  (addressing: {'word': 20833, 'token_id': 329})
    miss-path: {'accepted_unknown': 0.293, 'explored_lemma': 0.421, 'explored_near': 0.229, 'n_misses': 3286}
  - Llama-3.1-8B: {'get_word_list': 313, 'get_gloss': 14, 'get_syntax': 1, 'get_lemma': 1}  (addressing: {'word': 8, 'token_id': 7})
    miss-path: {'explored_lemma': 1.0, 'n_misses': 1}

### Metacognition calibration (H4, lexical heuristic)

| Model | Cond | gloss fetches | needed tokens | recall | precision |
|---|---|---|---|---|---|
| Qwen3.5-9B | E4-A | 5445 | 3504 | 0.32 | 0.206 |
| Qwen3.5-9B | E4-B | 4018 | 3504 | 0.24 | 0.209 |
| Gemma-4-E4B | E4-A | 1264 | 3085 | 0.077 | 0.189 |
| Gemma-4-E4B | E4-B | 14 | 3085 | 0.001 | 0.143 |
| Ministral-3-8B | E4-A | 13250 | 3495 | 0.829 | 0.219 |
| Ministral-3-8B | E4-B | 13213 | 3495 | 0.837 | 0.222 |
| Llama-3.1-8B | E4-A | 7 | 3373 | 0.0 | 0.143 |
| Llama-3.1-8B | E4-B | 1 | 3373 | 0.0 | 0.0 |
