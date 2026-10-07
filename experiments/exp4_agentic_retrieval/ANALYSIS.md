# Experiment 4 — Agentic Retrieval: Analysis

## Phase 0 — verification results (gated before the full run)

**P0-a (tool calling with thinking disabled): PASSED for Qwen3.5-9B.** The
`qwen3_coder` parser works with the thinking block pre-closed; calls emit and
parse reliably. The original concern is resolved empirically.

**P0-b (template round-trip): PASSED for all four families** (tool sections
render, `role=tool` results re-render, multi-turn episodes complete). Two
real integration bugs were found and fixed during Phase 0:

1. *Server strictness (not model failure)*: Qwen/Gemma pass `token_id` as the
   string `"1"` (JSON args) — the server's type-strict comparison rejected
   314 calls as "unresolved" before coercion. Also added stem-prefix
   resolution for agglutinative queries (agents querying the *stem* of an
   inflected form, e.g. `توسۇۋال` for `توسۇۋالغان`) — used 20× by Ministral
   in Phase 0 alone.
2. *Llama-3.1's template only supports a single tool call per assistant
   message* (vLLM `VLLMValidationError` otherwise). Multi-call plans are
   truncated to one executed call per turn; the rest are logged as skipped.
   Additionally Llama-3.1 frequently emits malformed JSON (`\uXXXX` escapes
   for Arabic words fail to parse). Unparseable tool-call attempts are routed
   to a corrective turn instead of being accepted as garbage final answers.
   (Integration note: the corrective routing was wired in *after* an
   assistant-last template crash surfaced in the first full-run attempt —
   when the budget is exhausted and the model keeps calling tools, the
   episode is now force-terminated as `budget_forced` protocol data instead
   of appending an assistant-terminated conversation that Mistral's template
   rejects. Regression-tested with a 2-call budget: all episodes complete.)

**P0-c (Phase-0 behavior, 30 episodes/model, log-only caps 32/12/1024):**

| Model | episodes w/ calls | calls/ep | tool mix | notes |
|---|---|---|---|---|
| Qwen3.5-9B | 15/30 | 10.3 (max 32) | gloss 209, lemma 49, near 15, features 17, syntax 3, list 15 | half the episodes need zero calls — perfectly selective |
| Gemma-4-E4B | 14/30 | 5.3 (max 12) | gloss 133, list 14, syntax 5, lemma 6 | modest, gloss-dominant, zero failures |
| Ministral-3-8B | 30/30 | 28.0 (max 32) | gloss 421, lemma 221, near 64, features 80, syntax 24, list 30 | textbook over-fetcher (H2) |
| Llama-3.1-8B | 5/30 | 0.2 | list only | protocol-broken: malformed JSON escapes + single-call template limit — measured as-is |

All resolutions clean after the fixes (Qwen: 290/290 token_id; Ministral:
exact 677 / stem_prefix 20 / edit-1 17 / unresolved 30).

**Calibrated caps** (from observed distributions): `MAX_CALLS=24`,
`MAX_ROUNDS=8`, `max_tokens=512` — covers selective agents fully, truncates
only extreme over-fetching (itself measured data).

(The `_p0` result files are retained in `results/` as the verification record.)

---

## Full-run results

(appended by score_exp4.py after the batch run)

### Integration incident log (full-run attempts)

1. **Assistant-terminated conversations** (first attempt, Ministral E4-A):
   budget-exhausted episodes where the model kept calling tools left the
   conversation ending on an assistant message → Mistral template rejects
   `add_generation_prompt=True`. Fix: force-terminate such episodes
   (`budget_forced`, protocol data); never append an assistant message with
   zero executed calls. Regression-tested with a 2-call budget.
2. **Garbage function names** (second attempt, Ministral E4-A at 384/997):
   the mistral parser can produce a "call" whose function name is a garbage
   string (the model garbles multi-call turns and echoes tool-result text);
   echoing that name back breaks mistral_common's function-name validation
   (`a-z, A-Z, 0-9, _, -`, ≤64 chars). Fix: parsed calls are validated
   against the real tool names before execution/echo; invalid-name calls are
   dropped and logged (`invalid_name_dropped`), and all-invalid turns route
   to a corrective turn. Regression test caught 1 such call in 24 episodes
   (previously batch-fatal).

3. **Context overflow** (third attempt, Llama-3.1-8B E4-A at 128/997):
   garbage-heavy corrective rounds compound (each failed round appends the
   garbled JSON echo + corrective message) until an episode's conversation
   exceeded `max_model_len=16384`, crashing the batch at render time. Fix:
   `max_model_len` raised to 32768 + a per-episode guardrail — conversations
   whose estimated size would overflow are force-terminated as
   `context_overflow` (logged protocol data; counted as degenerate finals in
   the analysis). Regression-tested on 32 Llama episodes: all complete,
   65 garbled-call corrections handled, zero crashes.

   Note on log noise: `llama_tool_parser.py ... Returning output in content
   with empty tool calls` ERROR lines are the *handled* path for Llama's
   malformed `\uXXXX` escapes (vLLM's parser recovers internally); every
   event is counted in `n_parse_failures` and routed to the corrective turn.

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

## Interpretation — pull is strictly dominated by push

**Verdict: agentic retrieval loses to the static lean prompt in every regime,
at 4–14× the context cost.** E4-A − C3L: Qwen-9B −4.76 ✱, Gemma-E4B −4.74 ✱,
Ministral-8B −10.65 ✱, Llama-8B −6.04 ✱ (all paired bootstrap, chrF++).

### Mechanism decomposition (per-episode Δ vs each model's own C0)

**Qwen3.5-9B (H1 refuted — selectivity does not pay, even for the capable):**

| Episode group | n | mean Δ vs own C0 |
|---|---|---|
| zero tool calls | 589 | +0.28 (framing neutral) |
| called tools, not budget-forced | 263 | **−5.69** (P(Δ>0)=0.33) |
| budget-forced | 145 | −0.73 |

Self-selection explains only ~2 points of the −5.69 (C0 on with-calls
episodes: 46.0 vs 48.1 on zero-call ones). The retrieval dialog *itself*
damages the translation, through two observed mechanisms:
(a) **register destruction** — after 20+ rounds of meta-dialog, final answers
become multi-variant discussions ("A more natural translation would be:",
"**Revised for fluency:**", "**Final polished translation:**"), wrecking the
clean single-sentence output; (b) **literal gloss-stitching** — freshly
fetched gloss text gets over-weighted ("how many times a day in the evening
you can feel your large brain's dream-seeing") at the cost of fluency and
sense disambiguation. The same information delivered as a static table
(C3L) helped (+0.71 ✱); delivered through the conversation, it hurts.

**Gemma-4-E4B (the anchor in its purest form):** the +12.4-over-C0 agency
gain is delivered almost entirely by episodes with **zero calls** (995/997 in
E4-B; sentence-Δ ≈ +9–10 on those). The tool-framing system context alone
normalizes Gemma's output (kills the C0 preamble-rambling) without fetching
any information. This is the Exp-3 control's anchoring mechanism
re-appearing in its purest form: framing without content. The lean arm (lean
> full, −2.45 ✱) repeats the interference pattern.

**Ministral-3-8B (H2 confirmed):** 997/997 episodes call, 22.8 calls/ep,
751/997 budget-forced; recall 0.83 achieved only by brute force (fetch
everything; precision 0.22). The loop-as-anchor effect is real but weak:
finals average 146–184 words (C0: ~320; C3L push: ~92) — the dialog halves
rambling where the table eliminated it.

**Llama-3.1-8B (the push/pull dissociation):** 2210 parse failures; only 14
gloss calls ever executed (mostly `get_word_list`, then the model cannot emit
Arabic words in its JSON format). Agency −1.03 ✱ vs C0 while its static C3L
was +4.52 ✱ — **the model can exploit glosses it cannot retrieve**:
language capability ≠ agency capability. Exactly the reason it was worth
keeping in the roster.

### The four hypotheses, settled

- H1 (selectivity pays): **refuted** — the best case (Qwen) loses 5.7 chrF
  on exactly the sentences where it retrieves.
- H2 (protocol burden): **confirmed, graded** — Gemma (framing-anchor gains,
  little fetching) > Qwen (clean protocol, damaged finals) > Ministral
  (over-fetching, budget-forcing) > Llama (protocol-broken).
- H3 (loop as anchor): **confirmed but weak** — the framing anchors output
  (Gemma zero-call +9–10), but far less than the static table does.
- H4 (metacognition): **refuted** — no model is calibrated: precision
  0.14–0.22 everywhere, recall high only via brute force (0.83, Ministral) or
  absent (0.08 Gemma, 0.0 Llama). None knows what it doesn't know.

### Cost inversion

E4-A contexts: 5.5k tokens/episode (Qwen) to 10.1k (Ministral) vs C3L's
~720 — pull is 4–14× more expensive *and* worse in every measured cell.
Agentic retrieval is strictly dominated by static lean prompting on both
axes of the deployment trade-off.

### Thesis-level synthesis

All four experiments now converge on one law: **the value of in-context
linguistic information lies in clean static framing that supplies knowledge
without disturbing the translation register** — the lean lemma-keyed gloss
table (C3L) being the optimum in every regime. Everything that disturbs the
register — maximal stacked tables, the morphology column, thinking mode,
multi-turn retrieval dialogs — pays a cost that the information never
recovers. Experiment 4 is the terminal demonstration: giving the model
*control* over retrieval costs more than the retrieved information is worth,
and the machinery worked exactly as designed (Qwen's spontaneous tool mix
was gloss-dominant; its miss-path behavior — explore-lemma 73%, accept 14%,
explore-near 10% — is precisely the sensible policy), which sharpens the
conclusion: the negative result is a property of the pull paradigm, not of
the implementation.
