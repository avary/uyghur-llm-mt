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

## Thinking-mode follow-up — design notes & verified format behavior

Both Qwen3.5 and Gemma-4 templates accept `enable_thinking` (rendered live:
Qwen opens the `uns` block; Gemma inserts a `<|think|>` system marker and
splits output with `<|channel>thought ... <channel|>`).

**Verified empirically (live single-turn + multi-turn round-trip tests):**

- **Qwen3.5-9B with thinking ON + tools emits NO `**END**` marker**: the
  reasoning prose flows directly into the qwen3-coder tool call (verified
  over three separate generations; multi-turn history with `tool_calls`
  re-renders correctly). Consequence: per-turn reasoning cannot be cleanly
  separated from content — reasoning text bleeds into echoed assistant
  messages and finals (smoke test: all 8 episodes' finals contained
  reasoning-bleed, 285-560 words). This is *measured model behavior*
  (register destruction in thinking mode, consistent with Exp 2b), not an
  implementation artifact. The loop treats marker-less thinking output as
  content and lets the corrective/final paths handle it.
- Qwen3.5's chat template requires `tool_calls` arguments as **dicts** in
  history (string arguments raise `Can only get item pairs from a mapping`
  with thinking on); the loop now echoes dict arguments.
- Gemma-4 splits cleanly (`<|channel>thought ... <channel|>`), so its
  reasoning is separable and `think_chars` will be meaningful.

**Budget policy (both thinking arms):** proportional — `budget = alpha x
content tokens` (alpha=1, floor 4; i.e. enough to gloss every word once),
identical in E4-AT and E4-AS so the thinking contrast stays clean. The
flat-24 budget previously penalized long sentences (35-token sentences hit
the cap with legitimate full-table fetching) and subsidized short ones.

**Token budgets loosened:** 16384 per assistant turn in thinking arms
(reasoning runs long even for single tool-call decisions; mid-reasoning
exhaustion would be our artifact, not model behavior), max_model_len 65536.
Context-overflow guardrail (from the Llama incident) remains as backstop.

**Sampling deconfounding:** thinking requires recommended sampling
(t=0.6, top_p=0.95, top_k=20). E4-AS (thinking OFF, same sampling) isolates
the sampling change; the clean thinking contrast is E4-AT − E4-AS.

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

### Thinking-mode follow-up — interpretation

**Headline: thinking splits the two families in opposite directions.**

| Contrast (paired bootstrap) | Qwen3.5-9B | Gemma-4-E4B |
|---|---|---|
| E4-AS − E4-A (sampling) | −0.69 n.s. | +1.65 ✱ |
| **E4-AT − E4-AS (thinking, clean)** | **−27.03 ✱** | **+1.57 ✱** |
| E4-AT − C3L (vs best push) | −32.48 ✱ | −1.52 ✱ |

**Qwen3.5-9B: catastrophic (43.6 → 15.8 chrF++).** The collapse is *not* a
protocol failure — it is register destruction at extreme scale: finals average
**305 words** (p95: 593) of reasoning-bleed ("Based on the lemmas and
context, I can now translate the sentence: - دۈشەنبە كۈنى = Monday …"), a
word-by-word gloss worksheet instead of a translation. With thinking enabled
(and no reasoning-end marker in tool mode — verified during design), Qwen
cannot stop *showing its work*: 997/997 episodes fetch (vs 497 in the control
arm), 14.2 calls/episode, 475 budget-forced. Consistent with Exp 2b (thinking
harmed its translation under greedy AND recommended sampling), now amplified
by the retrieval dialog: each round of reasoning gives the register
destruction more surface to latch onto.

**Gemma-4-E4B: the first-ever positive thinking result in this thesis
(+1.57 ✱, E4-A 38.1 → E4-AT 41.3, closing to within 1.5 of its C3L push).**
The behavior profile explains it: thinking makes Gemma *actually use* the
tools (967/997 episodes fetch, 7.8 calls/ep vs 1.5 thinking-off — recall
0.077 → 0.366), with clean 9k-char reasoning blocks and **compact finals** (no
register bleed). Gemma's thinking solves the same problem the static table
solved for it: it turns passive competence into active information use. But
the ceiling shows: still below its static C3L push (42.9) — even perfect
agency under-performs handing the model the right information.

**H4 update — reasoning improves recall, never precision.** Qwen recall
0.32 → 0.64, Gemma 0.08 → 0.37 under thinking; precision stays ~0.2 in all
six cells. Thinking makes models fetch *more*, not *better*: it activates
retrieval (Qwen's 997/997, Gemma's 967/997) without improving which words are
chosen. The metacognitive gap is not closed by reasoning tokens.

**Sampling deconfound (E4-AS):** near-zero for Qwen (−0.69 n.s.), mildly
positive for Gemma (+1.65 ✱ — stochasticity slightly helps its zero-shot-ish
anchored mode). Neither explains the thinking effects.

**Synthesis for the thesis:** the register-preservation law survives its
final stress test. Thinking mode is the strongest register destroyer we have
measured for Qwen (−27 in-loop, worse than its −6.4 static-greedy from Exp
2b, amplified by the retrieval dialog), while for a model whose failure mode
is *under*-engagement (Gemma's zero-call anchoring), reasoning tokens act as
an activator — buying real gains (+1.6) but never surpassing the static
optimum. The best information-delivery mechanism remains the lean static
gloss table; thinking in the loop is at best a partial substitute for it,
paid for in 8-16k-token reasoning traces, and at worst catastrophic.

## Draft-first protocol (E4-DF) — design notes

**Protocol** (system-prompt instruction, thinking OFF, greedy, E4-A-matched
caps — so E4-DF − E4-A is the clean protocol contrast):

> 1. In your first reply, translate the Uyghur sentence as well as you can on
>    your own. Output only the draft translation.
> 2. Identify the specific words or constructions you were uncertain about,
>    and call the available tools to check exactly those. Do not look up words
>    you were already sure of.
> 3. When done, reply with the improved English translation only.

Loop mechanics: the first no-tool-call turn is captured as the draft (with a
standardized nudge to check uncertainties — identical across episodes); the
next no-call turn is the final. Drafting + calling tools in one turn also
captures the draft (content part).

**Why this design**: it attacks the register-destruction mechanism diagnosed
in E4-A (retrieval before translation → worksheet mode) by making translation
the *primary* register and retrieval a verification step. It also unlocks two
measurements impossible before:
- **Δ(final − draft)** per episode: the value of retrieval given a draft,
  cleanly separated from framing/anchor effects (draft shares the framing,
  lacks the information);
- **Calibration against the draft** (fetched words vs words the draft itself
  got wrong) — a same-episode, same-framing needed-words signal, far better
  grounded than the C0-based heuristic used for H4 so far.

Smoke test (Qwen-9B, 8 episodes): 8/8 drafts produced, selective retrieval
(101 calls total vs ~240 at E4-A rates), visible draft→final corrections
("Pattaw" → "Pattawi" after 13 lookups). Full run: 4 models × 997.

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

### Draft-first results — interpretation

| Contrast | Qwen3.5-9B | Gemma-4-E4B |
|---|---|---|
| E4-DF − E4-A (protocol effect) | **+4.16 ✱** | **−6.00 ✱** |
| E4-DF − C0 | +0.09 n.s. | +6.40 ✱ |
| E4-DF − C3L (vs best push) | −0.60 ✱ | −10.74 ✱ |
| Δ(final − draft), per-episode | **+0.76** (P(better)=0.17 vs P(worse)=0.10) | **−5.19** (P(worse)=0.49!) |
| Draft chrF (same framing) | 46.54 | 39.35 |

**Qwen: the draft-first protocol nearly repairs agentic retrieval.** E4-A's
register destruction (43.56) is almost fully reversed (47.71, +4.16 ✱) —
translation restored as the primary register (finals: 23 words avg, zero
verification commentary). And the decomposition shows *where* the value lies:
drafts alone score **46.54** — within 1 point of C0 (47.63) — while retrieval
adds +0.76. The protocol's gain over E4-A is mostly the *framing* (the
draft-instruction context anchors the output), not the fetched information;
retrieval-on-a-draft is worth less than a static gloss table (+0.76 vs C3L's
+0.71-over-C0 — same ballpark, at ~10x the tokens). Qwen ends at −0.60 ✱
below C3L: draft-first agency converges back to the static optimum without
reaching it.

**Gemma: the same protocol is actively destructive (−6.00 ✱).** Mechanism,
quantified: 459/997 finals are **verification commentary** ("The initial
translation seems accurate based on the dictionary lookups…") — 22→55 avg
words, Δ(fin−dr) = −5.19 with P(worse) = 0.49. Restricted to the 538 episodes
where Gemma *did* output a translation, Δ(fin−dr) = **−0.01** — retrieval
neither helped nor hurt. So Gemma's damage is *pure register*: it answers the
instruction as a discussion prompt instead of a translation prompt, and its
drafts (39.35 — strong, near its E4-A zero-call anchor level) get abandoned
in favor of commentary.

**Why the protocol splits the families this way:** it demands *two* skills —
(a) produce a first-pass translation, (b) treat retrieval as internal
verification, invisible in the final output. Qwen has both; its only prior
failure was that retrieval-before-drafting dragged it into worksheet mode,
which the draft-first ordering removes. Gemma's compliance profile is the
opposite: its instruction-following binds to the *verification* framing
(weak models over-comply — echoing the E4-A zero-call anchoring result from
the other side), and it cannot suppress the meta-discourse. The same
instruction is a fix for one model and a trigger for the other: **prompt
protocols interact with model capability in both directions.**

**Calibration note (draft-based, better-grounded than the C0 heuristic):**
recall improves when retrieval is draft-anchored (Qwen 0.35, Gemma 0.59 — its
best ever) — but precision stays ~0.19 everywhere. A concrete artifact of
its own attempt does not teach either model *which* words it got wrong;
fetching remains poorly aimed even when anchored to a draft.

**Thesis-level summary of the agentic chapter:** across three protocols —
retrieve-then-translate (E4-A), thinking-enabled retrieval (E4-AT), and
draft-then-verify (E4-DF) — pull-based retrieval never beats the static lean
push (C3L) for any model, on any measure. The best agentic configuration
(Qwen E4-DF, 47.71) exactly reproduces the information value of a static
gloss table at 10x the token cost; every other configuration is worse. The
register-preservation law explains all of it: any protocol step that
interposes discourse between the model and the translation output (tool
dialog, reasoning, verification discussion) taxes quality by more than the
retrieved information pays.
