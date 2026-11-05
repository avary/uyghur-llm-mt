
## Exp 4 follow-up — thinking mode in the agentic loop

Sampling note: thinking mode requires recommended sampling (t=0.6, top_p=0.95, top_k=20); E4-AS (thinking OFF, same sampling) deconfounds the sampling change.

### Quality

| Model | Cond | chrF++ | COMET |
|---|---|---|---|
| Qwen3.5-9B | C0 | 47.63 | 0.8544 |
| Qwen3.5-9B | C3L | 48.32 | 0.8575 |
| Qwen3.5-9B | E4-A | 43.56 | 0.8216 |
| Qwen3.5-9B | E4-AS | 42.86 | 0.8270 |
| Qwen3.5-9B | E4-AT | 15.84 | 0.5004 |
| Gemma-4-E4B | C0 | 25.71 | 0.6341 |
| Gemma-4-E4B | C3L | 42.86 | 0.8078 |
| Gemma-4-E4B | E4-A | 38.12 | 0.7731 |
| Gemma-4-E4B | E4-AS | 39.77 | 0.7799 |
| Gemma-4-E4B | E4-AT | 41.34 | 0.7926 |

### Contrasts (paired bootstrap, chrF++)

| Model | Contrast | Δ [95% CI] | p |
|---|---|---|---|
| Qwen3.5-9B | E4-AS − E4-A (sampling effect) | -0.69 [-2.38, +0.94] | 0.4400 |
| Qwen3.5-9B | E4-AT − E4-AS (thinking effect, clean) | -27.03 [-28.77, -25.29] ✱ | 0.0000 |
| Qwen3.5-9B | E4-AT − E4-A (thinking+sampling combined) | -27.72 [-28.98, -26.46] ✱ | 0.0000 |
| Qwen3.5-9B | E4-AT − C3L (thinking pull vs best push) | -32.48 [-33.22, -31.69] ✱ | 0.0000 |
| Gemma-4-E4B | E4-AS − E4-A (sampling effect) | +1.65 [+0.86, +2.47] ✱ | 0.0004 |
| Gemma-4-E4B | E4-AT − E4-AS (thinking effect, clean) | +1.57 [+1.05, +2.13] ✱ | 0.0000 |
| Gemma-4-E4B | E4-AT − E4-A (thinking+sampling combined) | +3.22 [+2.39, +4.09] ✱ | 0.0000 |
| Gemma-4-E4B | E4-AT − C3L (thinking pull vs best push) | -1.52 [-1.96, -1.06] ✱ | 0.0000 |

### Behavior

| Model | Cond | ep w/ calls | calls/ep | rounds/ep | budget-forced | think chars/ep | length-caps |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | E4-A | 408/997 | 7.8 | 2.7 | 145 | 0 | 0 |
| Qwen3.5-9B | E4-AS | 497/997 | 7.8 | 2.5 | 394 | 0 | 3 |
| Qwen3.5-9B | E4-AT | 997/997 | 14.2 | 4.3 | 475 | 0 | 3 |
| Gemma-4-E4B | E4-A | 184/997 | 1.5 | 2.5 | 0 | 0 | 0 |
| Gemma-4-E4B | E4-AS | 208/997 | 1.8 | 2.5 | 57 | 135 | 0 |
| Gemma-4-E4B | E4-AT | 967/997 | 7.8 | 5.6 | 204 | 8263 | 0 |

Tool mix:
  - Qwen3.5-9B E4-AS: {'get_word_list': 478, 'get_gloss': 6343, 'get_lemma': 660, 'get_near_matches': 129, 'get_features': 87, 'get_syntax': 46}
  - Qwen3.5-9B E4-AT: {'get_word_list': 996, 'get_gloss': 10987, 'get_lemma': 1536, 'get_near_matches': 221, 'get_syntax': 240, 'get_features': 150}
  - Gemma-4-E4B E4-AS: {'get_word_list': 206, 'get_gloss': 1596, 'get_near_matches': 7, 'get_lemma': 9, 'get_syntax': 2}
  - Gemma-4-E4B E4-AT: {'get_word_list': 944, 'get_syntax': 576, 'get_gloss': 5820, 'get_near_matches': 121, 'get_features': 36, 'get_lemma': 304}

### Calibration (H4)

| Model | Cond | gloss fetches | recall | precision |
|---|---|---|---|---|
| Qwen3.5-9B | E4-A | 5445 | 0.32 | 0.206 |
| Qwen3.5-9B | E4-AS | 6275 | 0.371 | 0.207 |
| Qwen3.5-9B | E4-AT | 10851 | 0.635 | 0.205 |
| Gemma-4-E4B | E4-A | 1264 | 0.077 | 0.189 |
| Gemma-4-E4B | E4-AS | 1596 | 0.089 | 0.172 |
| Gemma-4-E4B | E4-AT | 5778 | 0.366 | 0.195 |
