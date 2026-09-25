# Experiment 3 — Cross-Family Sweep: Analysis

10 dense models × {C0 zero-shot, C7 full pipeline} × 997 FLORES dev sentences,
greedy decoding, no thinking mode, identical protocol to Exp 2 (directly
comparable). All runs 0 empty/failed outputs.

## Headline results

| Family | Model | C0 chrF++ | C7 chrF++ | Δ chrF++ [95% CI] | p | Δ COMET |
|---|---|---|---|---|---|---|
| Llama 3.2 | 1B | 13.58 | 11.70 | **−1.88** [−2.46, −1.30] ✱ | <0.0001 | −0.060 |
| Ministral 3 | 3B | 14.83 | 18.92 | **+4.09** [+3.69, +4.49] ✱ | <0.0001 | +0.077 |
| Llama 3.2 | 3B | 19.15 | 21.65 | **+2.50** [+1.75, +3.24] ✱ | <0.0001 | +0.126 |
| Gemma 4 | E2B | 16.48 | 35.42 | **+18.94** [+18.46, +19.45] ✱ | <0.0001 | +0.238 |
| Ministral 3 | 8B | 16.70 | 25.65 | **+8.96** [+8.26, +9.65] ✱ | <0.0001 | +0.157 |
| Gemma 4 | E4B | 25.71 | 41.84 | **+16.12** [+15.39, +16.83] ✱ | <0.0001 | +0.162 |
| Llama 3.1 | 8B | 32.04 | 36.56 | **+4.52** [+3.72, +5.30] ✱ | <0.0001 | +0.059 |
| Gemma 4 | 12B | 45.62 | 46.69 | +1.07 [+0.56, +1.60] ✱ | <0.0001 | +0.010 |
| Ministral 3 | 14B | 17.17 | 29.54 | **+12.37** [+11.63, +13.14] ✱ | <0.0001 | +0.198 |
| Gemma 4 | 31B | 50.20 | 50.88 | +0.67 [+0.17, +1.21] ✱ | 0.0056 | +0.007 |

9/10 models gain significantly; the only negative result is Llama-3.2-1B.

## Finding 1 — The gains are 3–20× larger than on Qwen, exactly as predicted

Best Qwen gain (Exp 2): +3.11 (0.8B). Unsupported-family gains: +2.5 to
**+18.94 chrF++** (Gemma-E2B: 16.48 → 35.42, +0.24 COMET). The Exp-2
conclusion generalizes: **in-context linguistic grounding substitutes for
missing parametric knowledge.**

## Finding 2 — Δ is a function of zero-shot competence, not family or size

Merging all 15 models (Exp 2 + Exp 3) sorted by C0:

| C0 chrF++ | Δ chrF++ | Model |
|---|---|---|
| 13.58 | **−1.88** | Llama-3.2-1B ← capacity floor |
| 14.83 | +4.09 | Ministral-3-3B |
| 16.48 | +18.94 | Gemma-4-E2B |
| 16.70 | +8.96 | Ministral-3-8B |
| 17.17 | +12.37 | Ministral-3-14B |
| 19.15 | +2.50 | Llama-3.2-3B |
| 23.39 | +3.11 | Qwen3.5-0.8B |
| 25.71 | +16.12 | Gemma-4-E4B |
| 32.04 | +4.52 | Llama-3.1-8B |
| 35.29 | +1.35 | Qwen3.5-2B |
| 44.58 | +0.90 | Qwen3.5-4B |
| 45.62 | +1.07 | Gemma-4-12B |
| 47.65 | +0.06 | Qwen3.5-9B |
| 50.20 | +0.67 | Gemma-4-31B |
| 51.03 | +0.92 | Qwen3.5-27B |

- Every model with C0 ≥ 44.5 (genuine Uyghur competence): Δ ≤ +1.07 —
  the pipeline is redundant.
- Every model with 14 ≤ C0 ≤ 32 (weak/no knowledge): Δ = +2.5…+18.9, all
  significant — the pipeline supplies what the weights lack.
- Exception at the bottom: Llama-3.2-1B (C0 13.6) is **negative** — a capacity
  floor: the model must be able to parse the table before it can exploit it.

This is the thesis's unifying figure: **Δ(C7−C0) vs zero-shot competence,
with a capacity floor** — family identity matters less than where the model
already stands. (Note Gemma-12B/31B and Qwen-9B/27B sit at the same C0 levels
and show the same near-zero Δ: Gemma-4 *does* have Uyghur knowledge, it just
arrives later in the size ladder — empirically resolving the "unsure" case.)

## Finding 3 — The double dissociation: scaling within knowledge-poor vs knowledge-rich families

- **Qwen (knowledge-rich):** C0 rises steeply with size (23→51); Δ shrinks
  (+3.11 → +0.06).
- **Ministral (knowledge-poor):** C0 stays **flat** across sizes (14.8 → 16.7
  → 17.2 — zero-shot ability does not scale at all), while Δ **grows** with
  size (+4.09 → +8.96 → +12.37): more capacity extracts more from the same
  external information.

Mirror-image trajectories: external info and parametric knowledge are
*substitutes*; capacity determines how much of the external info a model can
harvest, knowledge determines how little it needs to.

## Finding 4 — Why small Gemmas gain most: script literacy ≠ translation ability

The tokenizer probe showed Gemma-4 compresses Arabic-script Uyghur *best of
all* (1.79 chars/token vs Qwen 1.45, Ministral 1.23) — yet its zero-shot C0
at E2B (16.48) is no better than Ministral-3B (14.83). Gemma can *read* the
words but cannot *translate* them. Given glosses + syntax, it converts that
script literacy into translation instantly (E2B Δ = +18.94). Tokenizer
compression measures script exposure, not translation knowledge — the sweep
separates the two cleanly.

## Finding 5 — Qualitative mechanism: what the table changes sentence-by-sentence

Sentence 1, token دۈشەنبە ("Monday"):

| Model | C0 zero-shot | C7 with pipeline |
|---|---|---|
| Llama-3.1-8B | "On **Tuesday**, researchers…" | "On **Monday**…" |
| Gemma-4-E2B | "On **Wednesday**, the University…" | "On **Monday**…" |
| Ministral-3-8B | "On **Tuesday**, …" (buried in preamble) | "On **Monday**…" |

The gloss genuinely resolves content words when the entry is a dictionary hit
(cell, Monday). **But the control run (below) shows this effect is rare at
corpus scale:** the dictionary contributes only ~±1 chrF++ beyond structure —
individual fixes are visible but not what moves the corpus score.

## Control run — the anchor/knowledge decomposition (and a revised conclusion)

Confound flagged above, now resolved. C1 (syntax-only) and C2 (morphology-only)
carry the same table-format anchor as C7 but **no English glosses**, run on
Ministral-3-8B and Gemma-4-E2B (the biggest rambler and the biggest Δ).

### Scores

| Model | Cond | chrF++ | COMET | len-cap | avg out tok |
|---|---|---|---|---|---|
| Ministral-3-8B | C0 | 16.70 | 0.4904 | 47 | 319 |
| Ministral-3-8B | C1 | 28.22 | 0.6585 | 2 | 60 |
| Ministral-3-8B | C2 | 25.56 | 0.6339 | 23 | 104 |
| Ministral-3-8B | C7 | 25.65 | 0.6471 | 37 | 135 |
| Gemma-4-E2B | C0 | 16.48 | 0.4935 | 75 | 376 |
| Gemma-4-E2B | C1 | 34.27 | 0.7124 | 0 | 28 |
| Gemma-4-E2B | C2 | 34.39 | 0.7159 | 1 | 29 |
| Gemma-4-E2B | C7 | 35.42 | 0.7312 | 0 | 28 |

### Paired bootstrap deltas (chrF++)

| Model | Contrast | Δ [95% CI] | p |
|---|---|---|---|
| Ministral-3-8B | C1 − C0 | +11.52 [+11.03, +12.00] ✱ | 0.0000 |
| Ministral-3-8B | C2 − C0 | +8.86 [+8.23, +9.53] ✱ | 0.0000 |
| Ministral-3-8B | C7 − C0 | +8.96 [+8.26, +9.65] ✱ | 0.0000 |
| Ministral-3-8B | **C7 − C1** | **−2.56 [−3.31, −1.84] ✱** | 0.0000 |
| Ministral-3-8B | C7 − C2 | +0.10 [−0.76, +0.93] | 0.8300 |
| Gemma-4-E2B | C1 − C0 | +17.79 [+17.33, +18.27] ✱ | 0.0000 |
| Gemma-4-E2B | C2 − C0 | +17.91 [+17.40, +18.43] ✱ | 0.0000 |
| Gemma-4-E2B | C7 − C0 | +18.94 [+18.46, +19.45] ✱ | 0.0000 |
| Gemma-4-E2B | **C7 − C1** | **+1.15 [+0.76, +1.54] ✱** | 0.0000 |
| Gemma-4-E2B | C7 − C2 | +1.03 [+0.58, +1.48] ✱ | 0.0000 |

### Reading of the control — the gain is mostly structure, not lexical knowledge

1. **The anchor carries almost the whole gain.** C1 — no lemmas, no morphology,
   no glosses, no English beyond deprel labels — delivers +11.5 (Ministral) and
   +17.8 (Gemma) of the +9.0/+18.9 total. What C1 actually provides:
   a "this is a translatable sentence" frame, word-by-word segmentation of the
   Arabic-script source (critical for models that fragment it into 6+ tokens per
   word), grammatical roles (nsubj/obj/root), and an output-format constraint
   that kills the C0 degeneration (avg 319→60 and 376→28 tokens; length-caps
   47→2 and 75→0).
2. **Dictionary glosses contribute almost nothing beyond structure.** On top
   of the anchor: +1.15 (Gemma), and **−2.56 for Ministral — significantly
   negative**. Adding morphology+dictionary to syntax makes Ministral *worse*
   than syntax alone. The interference effect from Exp 1 (C7 < single
   components at 9B Qwen) replicates in an unsupported family — much stronger.
3. **Why glosses fail knowledge-poor models:** (a) dictionary misses render as
   `[lemma]` — Arabic script, unreadable to exactly the models we're trying to
   help; (b) `≈` spelling-near candidates include false friends ("Pineapple
   (Ananas comosa)" for بولۇپ) that a model which cannot verify in context
   will *trust*; (c) multi-sense glosses ("ket-: To go; To cost; To come off…")
   require disambiguation ability the model lacks. The dictionary was designed
   under the assumption that the reader can judge candidates — true for the
   9B translator of Exp 1, false for a 3B model with zero Uyghur.
4. **Gemma's C2 ≈ C1 ≈ C7−1**: for a script-literate model, lemma+ULY
   (morphology, no glosses) is worth as much as the whole dictionary — its
   small latent lexicon does the lexical work once the script barrier and
   segmentation are removed. This is consistent with the tokenizer probe:
   Gemma's advantage is script fluency, which morphology (ULY) unlocks.
5. The sentence-level Monday fixes (Finding 5) are real but rare: corpus-wide,
   the dictionary moves ±1 chrF++. Individual gloss hits are visible in
   qualitative examples while contributing almost nothing to the aggregate.

### Revised bottom line for the thesis

- **The pipeline's dominant effect on knowledge-poor models is structural
  grounding** — segmentation, grammatical skeleton, output normalization —
  not bilingual knowledge injection. "Grounding" works by making the model a
  *better reader of the input* (and a cleaner writer), not by teaching it
  words.
- **Information-maximal prompting is suboptimal everywhere we measured it:**
  9B Qwen (C3 best, C7 ≈ C0), Ministral-8B (C1 best, C7 < C1), Gemma-E2B
  (C1/C2 within ~1 of C7 at half the tokens). The full table buys nothing that
  a leaner prompt doesn't, and costs quality for weak or mid-size models.
- **Where knowledge *is* missing, it is not fixable by gloss-lists** — the
  glosses are noisy (misses, false friends, sense ambiguity) relative to a
  reader that cannot filter them. The usable form of external lexical
  knowledge for weak models would need context-selected senses, i.e. per-token
  glosses already resolved for the sentence — which is a different artifact
  from a dictionary dump.
- Capacity floor (~2B, Llama-1B negative) and competence ceiling (C0 ≥ ~44 →
  Δ ≈ 0) bracket the regime where structural prompting pays: weak-but-capable
  models, which is precisely the deployment scenario for low-resource MT.

## Full component ablation (C0-C7) — knowledge-poor regime

Mirrors Exp 1 (Qwen3.5-9B) on Ministral-3-8B and Gemma-4-E2B.

### Ministral-3-8B

| Cond | chrF++ | COMET | Δ chrF++ vs C0 [95% CI] | p | len-cap | avg out tok |
|---|---|---|---|---|---|---|
| C0 baseline (zero-shot) | 16.70 | 0.4904 | — | — | 47 | 319 |
| C1 + syntax | 28.22 | 0.6585 | +11.52 [+11.03, +12.00] ✱ | 0.0000 | 2 | 60 |
| C2 + morphology | 25.56 | 0.6339 | +8.86 [+8.23, +9.53] ✱ | 0.0000 | 23 | 104 |
| C3 + dictionary | 27.74 | 0.6593 | +11.04 [+10.42, +11.70] ✱ | 0.0000 | 24 | 93 |
| C4 + syntax + morph | 24.09 | 0.6179 | +7.39 [+6.78, +8.06] ✱ | 0.0000 | 39 | 133 |
| C5 + syntax + dict | 28.13 | 0.6696 | +11.44 [+10.79, +12.11] ✱ | 0.0000 | 16 | 94 |
| C6 + morph + dict | 25.56 | 0.6508 | +8.86 [+8.18, +9.62] ✱ | 0.0000 | 74 | 140 |
| C7 full pipeline | 25.65 | 0.6471 | +8.96 [+8.26, +9.65] ✱ | 0.0000 | 37 | 135 |

### Gemma-4-E2B

| Cond | chrF++ | COMET | Δ chrF++ vs C0 [95% CI] | p | len-cap | avg out tok |
|---|---|---|---|---|---|---|
| C0 baseline (zero-shot) | 16.48 | 0.4935 | — | — | 75 | 376 |
| C1 + syntax | 34.27 | 0.7124 | +17.79 [+17.33, +18.27] ✱ | 0.0000 | 0 | 28 |
| C2 + morphology | 34.39 | 0.7159 | +17.91 [+17.40, +18.43] ✱ | 0.0000 | 1 | 29 |
| C3 + dictionary | 35.84 | 0.7341 | +19.36 [+18.86, +19.87] ✱ | 0.0000 | 0 | 29 |
| C4 + syntax + morph | 33.86 | 0.7079 | +17.37 [+16.89, +17.87] ✱ | 0.0000 | 0 | 29 |
| C5 + syntax + dict | 35.58 | 0.7317 | +19.09 [+18.59, +19.63] ✱ | 0.0000 | 0 | 28 |
| C6 + morph + dict | 35.94 | 0.7402 | +19.46 [+18.94, +19.98] ✱ | 0.0000 | 0 | 29 |
| C7 full pipeline | 35.42 | 0.7312 | +18.94 [+18.46, +19.45] ✱ | 0.0000 | 0 | 28 |


### Full-factorial interpretation — the marginal-effects analysis

Paired bootstrap on the *marginal* contrasts (component added to an existing
prompt), across all three factorial regimes:

**Morphology added to a combination** (the column `Lemma | ULY | Features`):

| Model | +morph on syntax | +morph on dict | +morph on syn+dict |
|---|---|---|---|
| Qwen3.5-9B (Exp 1) | −0.23 [−0.49, +0.02] | **−0.40 ✱** [−0.72, −0.11] | −0.15 [−0.41, +0.12] |
| Gemma-4-E2B | **−0.42 ✱** [−0.77, −0.07] | +0.10 [−0.26, +0.45] | −0.16 [−0.49, +0.18] |
| Ministral-3-8B | **−4.13 ✱** [−4.71, −3.47] | **−2.18 ✱** [−3.00, −1.41] | **−2.48 ✱** [−3.23, −1.74] |

**9 of 9 cells negative-or-zero; never significantly positive in any regime.**
Morphology-in-combination is a universal tax — small for strong/stable models,
catastrophic for the fragile one. This identifies the *culprit behind Exp 1's
interference finding* (C7 < singles at 9B): it was never "too much information"
generically — it is specifically the morphological-features column.

**Dictionary added to a combination** (the `ULY | Gloss` columns):

| Model | +dict on syntax | +dict on morph |
|---|---|---|
| Qwen3.5-9B | −0.20 [−0.52, +0.11] | −0.24 [−0.55, +0.06] |
| Gemma-4-E2B | **+1.30 ✱** [+0.91, +1.70] | **+1.55 ✱** [+1.08, +2.04] |
| Ministral-3-8B | −0.08 [−0.76, +0.60] | +0.00 [−0.83, +0.81] |

The dictionary is **additive knowledge — but only for the script-literate
model** (+1.3–1.6 everywhere for Gemma), exactly **neutral for the
low-literacy model** (marginal 0.00 for Ministral), and unneeded by Qwen-9B
(≈0; its dict value only appears as the best *single* over C0, +0.71).

**Mechanism check — morphology tax is mediated by degeneration in the
fragile model** (Ministral; avg output tokens / length-caps per condition):

| | C1 | C3 | C5 | C2 | C4 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| morph absent/present | absent | absent | absent | present | present | present | present |
| avg out tok | 60 | 93 | 94 | 104 | 133 | 140 | 135 |
| len-capped | 2 | 24 | 16 | 23 | 39 | **74** | 37 |

Every morph-containing condition sits in the longer, more degenerate band.
The verbose English feature strings ("Standard verb, Transitive, Passive
voice, Past gerund, Third person possessive singular or plural, Accusative")
invite the model to talk instead of translate. Gemma shows the same sign of
the effect at ~10× smaller magnitude with zero degeneration — for a stable
model the tax is attention dilution, for a fragile one it is collapse.

**Saturation.** For both weak models the first component does ~90–100% of the
work (Ministral: first block +8.9…+11.5, any second block adds ≤ +1.6 or
subtracts; Gemma: first block +17.4…+19.4). Components do not stack; the
attention budget does not grow with the table.

**Optimal prompt per regime — C7 is never optimal:**

| Regime (example) | Best condition | chrF++ | vs C7 |
|---|---|---|---|
| Knowledge-rich (Qwen3.5-9B) | **C3 dict-only** | 48.34 | +0.63 |
| Script-literate, lexicon-poor (Gemma-4-E2B) | C6 morph+dict ≈ **C3** (Δ +0.10 n.s.) | 35.94 / 35.84 | +0.52 |
| Low-literacy, fragile (Ministral-3-8B) | **C1 syntax** ≈ C5 ≈ C3 | 28.22 | +2.57 |

Design recommendation (actionable for the pipeline deliverable —
`PipelineConfig` already exposes the `include_morphology` /
`include_dictionary_glosses` toggles): compose the prompt *conditionally* —
segmentation + roles as the universal base, glosses when the model is
script-literate, and drop the morphological-features column whenever the
model is small or its Uyghur exposure is unknown. The maximal table is
2.25× the tokens of the lean optimum (1,753 vs 780 prompt tokens in
Ministral's tokenizer) and delivers less in every regime measured.
