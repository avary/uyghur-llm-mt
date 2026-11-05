
## Exp 4 follow-up — draft-first protocol (E4-DF)

Instruction (system prompt): draft on your own → check ONLY uncertainties with tools → final translation. Thinking off, greedy, same caps as E4-A.

### Quality

| Model | Cond | chrF++ | COMET |
|---|---|---|---|
| Qwen3.5-9B | C0 | 47.63 | 0.8544 |
| Qwen3.5-9B | C3L | 48.32 | 0.8575 |
| Qwen3.5-9B | E4-A | 43.56 | 0.8216 |
| Qwen3.5-9B | E4-DF | 47.71 | 0.8423 |
| Gemma-4-E4B | C0 | 25.71 | 0.6341 |
| Gemma-4-E4B | C3L | 42.86 | 0.8078 |
| Gemma-4-E4B | E4-A | 38.12 | 0.7731 |
| Gemma-4-E4B | E4-DF | 32.12 | 0.6621 |

### Contrasts (paired bootstrap, chrF++)

| Model | Contrast | Δ [95% CI] | p |
|---|---|---|---|
| Qwen3.5-9B | E4-DF − E4-A (draft-first protocol) | +4.16 [+3.09, +5.27] ✱ | 0.0000 |
| Qwen3.5-9B | E4-DF − C0 | +0.09 [-0.44, +0.61] | 0.7484 |
| Qwen3.5-9B | E4-DF − C3L (vs best push) | -0.60 [-1.08, -0.12] ✱ | 0.0144 |
| Gemma-4-E4B | E4-DF − E4-A (draft-first protocol) | -6.00 [-6.98, -5.01] ✱ | 0.0000 |
| Gemma-4-E4B | E4-DF − C0 | +6.40 [+5.59, +7.21] ✱ | 0.0000 |
| Gemma-4-E4B | E4-DF − C3L (vs best push) | -10.74 [-11.50, -10.00] ✱ | 0.0000 |

### Draft analysis (per-episode, sentence chrF)

| Model | has draft | changed | draft chrF | final chrF | Δ(fin−dr) mean | P(better) | P(worse) |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 997/997 | 365 | 46.54 | 47.3 | 0.76 | 0.169 | 0.102 |
| Gemma-4-E4B | 997/997 | 611 | 39.35 | 34.16 | -5.19 | 0.089 | 0.491 |

### Calibration vs draft + behavior (E4-DF)

- **Qwen3.5-9B**: gloss fetches 6109, recall 0.35, precision 0.183; calls/ep 9.5, budget-forced 34, episodes w/ calls 976/997, parse-fail 0, tool mix {'get_gloss': 6651, 'get_near_matches': 778, 'get_lemma': 1276, 'get_word_list': 536, 'get_features': 215, 'get_syntax': 44}
- **Gemma-4-E4B**: gloss fetches 10494, recall 0.588, precision 0.194; calls/ep 11.0, budget-forced 13, episodes w/ calls 997/997, parse-fail 0, tool mix {'get_word_list': 296, 'get_gloss': 10529, 'get_lemma': 82, 'get_near_matches': 83, 'get_features': 13, 'get_syntax': 5}
