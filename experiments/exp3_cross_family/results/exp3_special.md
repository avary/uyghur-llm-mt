
## Follow-up studies — segmentation (S) and lemma-keyed dictionary (C3L)

### Ministral-3-8B

| Cond | chrF++ | COMET | len-cap | preamble | avg out tok |
|---|---|---|---|---|---|
| C0 | 16.70 | 0.4904 | 47 | 0 | 319 |
| C1 | 28.22 | 0.6585 | 2 | 1 | 60 |
| C3 | 27.74 | 0.6593 | 24 | 2 | 93 |
| S | 13.28 | 0.4146 | 531 | 200 | 458 |
| C3L | 29.60 | 0.6863 | 13 | 3 | 81 |

### Gemma-4-E2B

| Cond | chrF++ | COMET | len-cap | preamble | avg out tok |
|---|---|---|---|---|---|
| C0 | 16.48 | 0.4935 | 75 | 33 | 376 |
| C1 | 34.27 | 0.7124 | 0 | 0 | 28 |
| C3 | 35.84 | 0.7341 | 0 | 0 | 29 |
| S | 16.67 | 0.4423 | 110 | 991 | 264 |
| C3L | 36.79 | 0.7475 | 0 | 0 | 28 |

### Qwen3.5-9B

| Cond | chrF++ | COMET | len-cap | preamble | avg out tok |
|---|---|---|---|---|---|
| C0 | 47.63 | 0.8544 | 0 | 1 | 28 |
| C1 | 48.05 | 0.8537 | 0 | 1 | 28 |
| C3old | 48.34 | 0.8568 | 0 | 1 | 28 |
| C3 | 48.34 | 0.8568 | 0 | 1 | 28 |
| S | 38.16 | 0.7110 | 0 | 1 | 56 |
| C3L | 48.32 | 0.8575 | 0 | 1 | 28 |

### Llama-3.2-3B

| Cond | chrF++ | COMET | len-cap | preamble | avg out tok |
|---|---|---|---|---|---|
| C0 | 19.15 | 0.4523 | 17 | 697 | 235 |
| C1 | 23.19 | 0.6036 | 18 | 0 | 42 |
| C7 | 21.65 | 0.5784 | 72 | 10 | 116 |
| S | 13.13 | 0.4138 | 170 | 697 | 298 |

### Llama-3.2-1B

| Cond | chrF++ | COMET | len-cap | preamble | avg out tok |
|---|---|---|---|---|---|
| C0 | 13.58 | 0.4530 | 14 | 245 | 54 |
| C1 | 13.04 | 0.4233 | 163 | 0 | 262 |
| C7 | 11.70 | 0.3930 | 621 | 8 | 391 |
| S | 12.85 | 0.3860 | 78 | 0 | 234 |

### Paired bootstrap contrasts (chrF++)

| Model | Contrast | Δ [95% CI] | p |
|---|---|---|---|
| Ministral-3-8B | S − C0 (presentation only) | -3.41 [-3.63, -3.20] ✱ | 0.0000 |
| Gemma-4-E2B | S − C0 (presentation only) | +0.19 [-0.14, +0.51] | 0.2688 |
| Qwen3.5-9B | S − C0 (presentation only) | -9.46 [-10.19, -8.73] ✱ | 0.0000 |
| Llama-3.2-3B | S − C0 (presentation only) | -6.02 [-6.46, -5.59] ✱ | 0.0000 |
| Llama-3.2-1B | S − C0 (presentation only) | -0.74 [-1.33, -0.16] ✱ | 0.0128 |
| Ministral-3-8B | S − C1 (presentation vs roles) | -14.93 [-15.45, -14.41] ✱ | 0.0000 |
| Gemma-4-E2B | S − C1 (presentation vs roles) | -17.60 [-18.19, -17.03] ✱ | 0.0000 |
| Qwen3.5-9B | S − C1 (presentation vs roles) | -9.89 [-10.61, -9.13] ✱ | 0.0000 |
| Llama-3.2-3B | S − C1 (presentation vs roles) | -10.06 [-10.81, -9.28] ✱ | 0.0000 |
| Llama-3.2-1B | S − C1 (presentation vs roles) | -0.19 [-0.60, +0.22] | 0.3564 |
| Ministral-3-8B | C3L − C3 (lemma-keyed dict) | +1.86 [+1.13, +2.57] ✱ | 0.0000 |
| Ministral-3-8B | C3L − C1 (lemma-dict vs roles) | +1.39 [+0.68, +2.06] ✱ | 0.0000 |
| Gemma-4-E2B | C3L − C3 (lemma-keyed dict) | +0.95 [+0.60, +1.29] ✱ | 0.0000 |
| Gemma-4-E2B | C3L − C1 (lemma-dict vs roles) | +2.52 [+2.13, +2.91] ✱ | 0.0000 |
| Qwen3.5-9B | C3L − C3 (lemma-keyed dict) | -0.02 [-0.28, +0.23] | 0.8648 |
| Qwen3.5-9B | C3L − C1 (lemma-dict vs roles) | +0.26 [-0.08, +0.60] | 0.1380 |
| Qwen3.5-9B | C3(new gloss fmt) − C3(old fmt) | +0.00 [+0.00, +0.00] | 2.0000 |
| Llama-3.2-1B | C1 − C0 (floor: lean prompt) | -0.55 [-1.16, +0.06] | 0.0772 |
| Llama-3.2-1B | C7 − C1 (floor: full prompt) | -1.34 [-1.76, -0.93] ✱ | 0.0000 |

### Dictionary lookup coverage (FLORES dev, 17,159 content tokens)

- lemma-keyed hit rate: **80.4%** (near-advisory extra 9.2%)
- surface-keyed hit rate: **44.0%**
- every surface hit is also a lemma hit (surface-only: 0.0%; lemma-only: +36.4 points)

Lemma-only examples (surface lookup missed, lemma lookup hit):
  - `كۈنى` → lemma `كۈن` → "kün (n.): 1. [astronomy] Sun / 2. Day / 3. Life; existence"
  - `ئۇنىۋېرسىتېتى` → lemma `ئۇنىۋېرسىتېت` → "uniwërsitët (n.): University"
  - `ئالىملىرى` → lemma `ئالىم` → "alim (n.): 1. Scholar; learned person / 2. Male name"
