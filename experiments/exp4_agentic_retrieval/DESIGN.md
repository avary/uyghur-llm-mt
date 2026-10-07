# Experiment 4 — Agentic Retrieval: Design

**Research question**: after Exps 1–3 established *which* information helps
(the lemma-keyed dictionary, in a lean table) and *which* hurts (everything
stacked together, morph features especially), can the LLM itself decide what
it needs — fetching information on demand instead of receiving a fixed prompt?

## 1. Motivation (what the previous experiments tell us)

- The best static prompt is the same in every regime: the lean lemma-keyed
  gloss table (C3L). The maximal prompt (C7) never pays for its extra sections;
  the morphological-features column is a tax in combinations.
- What a model *needs* depends on what it already knows (Qwen-9B needs ~3
  glosses per sentence; Gemma-E2B needs most of the table; Ministral needs the
  whole anchor). A static prompt cannot adapt per sentence or per model.
- Push prompts are one-size-fits-all at the *sentence* level too: a sentence
  full of loanwords needs different help than one full of morphology.

So the natural next question: **is the LLM a good information manager for
itself?** — selecting which words to look up, and whether that selectivity
beats (or at least matches) the best static push at lower context cost.

## 2. Hypotheses

- **H1 (selectivity pays where knowledge exists)**: Qwen-9B will fetch few,
  well-chosen glosses → equal-or-better quality than C3L push at a fraction of
  the context tokens.
- **H2 (protocol burden where it doesn't)**: fragile models (Ministral-8B)
  will misdrive the loop (malformed calls, over-fetching, degeneration) →
  worse than C3L push. The Exp-2/3 crossover (capability ↔ grounding value)
  should reappear as *capability ↔ agency value*.
- **H3 (loop as anchor)**: the multi-turn protocol forces short, functional
  messages each round — it may itself discipline weak models' rambling (their
  static C0 failure mode).
- **H4 (metacognition calibration)**: the overlap between what a model
  *fetched* and what it *needed* (measured via its own C0 errors) quantifies
  whether LLMs know what they don't know — per model competence level.

## 3. Architecture

### 3.1 Round-synchronous agent loop (batch-friendly)

Each FLORES sentence is an independent single-agent episode:

```
initial: system-lite + "Translate this Uyghur sentence." + raw sentence + tool schemas
            │
   ┌────────▼─────────┐
   │ assistant turn N  │── no tool call ──► final translation; episode ends
   │  (0..k calls)    │
   └────────┬─────────┘
            │ tool calls
   execute all calls of this turn against the precomputed pipeline analysis
            │
   append tool results (role=tool) ──► next round
```

- **Batching**: episodes are run round-synchronously — at each round, *all*
  still-active episodes are dispatched in ONE `llm.chat()` call (vLLM
  continuous batching). Sentences finish at different rounds; finished
  episodes drop out of the pool. This answers the classic "agentic breaks
  batching" objection: total rounds ≈ max episode length, not sum.
- **Budget & turn cap — provisional, calibrated in Phase 0**: guardrail
  defaults `MAX_TOOL_CALLS = 12`, `MAX_ROUNDS = 6`, `max_tokens = 512` per
  assistant turn (uniform). All three are **verified empirically in Phase 0**
  (unconstrained log-only smoke episodes; caps then set at ~p95 + margin) —
  see §7. On budget exhaustion, inject: *"Tool budget exhausted. Provide the
  final English translation now."* — the next assistant message is the
  translation.
- **Decoding**: greedy, thinking disabled — identical regime to all prior
  experiments.

### 3.2 Tool-call handling (validated feasibility)

- `llm.chat(..., tools=[...])` renders each family's native tool section
  (validated: Qwen3.5, Gemma-4, Llama-3.1, Ministral-3 templates all accept
  tools + `role=tool` result turns).
- Assistant text is parsed with vLLM's own per-family parsers
  (registry names: **`qwen3_coder`** for Qwen3.5, `llama3_json`,
  `mistral`, `gemma4`) → structured calls.
- Multiple calls per turn are executed together and appended together
  (the templates support consecutive tool-result turns).

## 4. The tool set — essential functions and return formats

**Design principles**:

1. **Minimal units**: each tool returns exactly one information type; the
   agent composes. (Informed by the static experiments: stacked information
   interferes, the lean C3L table is optimal.)
2. **Purity invariant**: the free tier (initial prompt, `get_word_list`)
   carries identity only — no analysis bytes outside a tool call. Otherwise
   pull collapses into push.
3. **Fallbacks are opt-in**: when information is missing or uncertain, the
   tool says so cleanly and lists the options — the DECISION to explore
   stays with the agent.
4. **Static-prompt parity**: every return is byte-identical to the fragments
   the static prompts contain (same rendering functions) — if an agent
   fetched everything it would reconstruct the static prompts exactly.
5. **Uniform envelope**: every per-word tool anchors its return with the same
   `Token #N: form (ULY)` header so the agent always knows which word it is
   looking at; parallel calls in one turn are supported (budget counts
   calls, not turns).
6. **No compromises for weak models**: malformed/mangled queries get honest
   error returns and are logged; nothing is silently repaired. (Server-side
   fuzzy resolution guards against Arabic-script echo corruption — NFC/NFD,
   zero-width joiners, confusables — at every capability level; every
   non-exact resolution is recorded.)

### The tool set (v2)

| Tool | Returns | Availability |
|---|---|---|
| `get_word_list()` | numbered token list (address space) | — |
| `get_lemma(word)` | stem + ULY | **100%** of analyzed tokens |
| `get_gloss(word)` | lemma line + numbered senses, or clean miss | 80.4% hit rate |
| `get_near_matches(word)` | `≈` spelling-near candidates + advisory | on misses |
| `get_features(word)` | POS + grammatical features (the demonstrated tax — included to test agent judgment) | 100% |
| `get_syntax()` | clause skeleton (sentence-level digest) | — |

Every per-word tool accepts `word: string` (primary: what the model can see)
and optional `token_id: int`; whichever is given is resolved against the
sentence's pipeline tokens (exact → normalized → edit-distance-1, all
logged). Tools are sentence-scoped: only tokens of the current sentence can
be queried.

### 4.1 `get_word_list()` — the address space

```
1. دۈشەنبە  2. كۈنى  3. ،  4. ستانفورد  5. ئۇنىۋېرسىتېتى …
```

Bare forms only (purity invariant: no ULY, no analysis). Optional for the
agent — not infrastructure.

### 4.2 `get_lemma(word)` — the stem, separated from the gloss

Works even when the dictionary has no entry (stem + transliteration are
computable for every analyzed token — 100% availability):

```
Token #2: كۈنى → lemma كۈن (kün)
```

Value even without a gloss: ULY is the script bridge (cognate recognition:
uniwërsitët → university), and the stem is the dictionary key. A capable
agent may find the lemma sufficient and skip the gloss entirely.

### 4.3 `get_gloss(word)` — the C3L row, without fallbacks attached

On hit (byte-identical to the C3L dictionary column):

```
Token #2: كۈنى (küni)
Lemma: كۈن (kün) — noun
Senses: 1. [astronomy] Sun / 2. Day / 3. Life; existence
```

On miss — **clean, with discoverability hints, no auto-exploration**:

```
Token #4: ستانفورد (stanford) — no dictionary entry.
(Options: get_lemma for the stem/transliteration, get_near_matches for
spelling-near candidates.)
```

### 4.4 `get_near_matches(word)` — the `≈` advisory, now a decision

Only when the agent asks (the pipeline's advisory caveat included):

```
Token #6: تېببىي (tëbbiy) — no exact entry. Spelling-near candidates
(verify meaning and part of speech in context):
≈ tebiy (adj.): 1. Natural / 2. Only natural; of course
≈ tërbiy (n.): Terbium (Tb)
```

The gloss-miss response now creates a **measurable trichotomy**: on a miss,
the agent (a) accepts the unknown and moves on, (b) pulls the lemma
(script-bridge guess), or (c) explores near matches (with their false-friend
risk — Exp 3 showed near-match noise misleads weak models; does the agent
know better?).

### 4.5 `get_features(word)` — POS + morphological features (renamed from
`get_word_analysis` after offloading the lemma)

```
Token #2: كۈنى — Noun, Third person possessive singular or plural, Nominative
```

Deliberately retained despite being the demonstrated tax in combination:
   agency is precisely the test of whether the model avoids it.

### 4.6 `get_syntax()` — clause skeleton (the C1 section, unchanged)

```
- Main Predicate / Action: كېتىدۇ (root)
- Subject(s): ئالىملىرى (nsubj -> #16 قىلدى), …
- Object(s): … / Adverbial / Obliques: …
```

### Considered and rejected

- `get_pos(word)` — redundant; POS rides on the gloss line ("kün (n.)").
- `get_transliteration(word)` — covered by `get_lemma` (lemma ULY); a
  surface-ULY tool would invite script-bridging without normalization.
- Per-word `get_dependency(word)` — syntax value is sentence-structural; our
  evidence says it is second-order; the skeleton digest suffices.
- Sense-picking (`get_sense(word, i)`) — over-decomposition; sense selection
  is in-context work the model should do itself.
- `get_full_table()` / bulk glosses — rejected earlier (re-measures push).

### NOT included (by design, unchanged)

No translation-suggestion tool, no web/knowledge tool — strictly retrieval
selectivity over OUR pipeline's information.

## 5. Experimental conditions

| Condition | Tools available | Budget |
|---|---|---|
| **E4-A** (main) | full tool set v2 (word_list, lemma, gloss, near_matches, features, syntax) | Phase-0-calibrated |
| **E4-B** (lean) | gloss only (plus word_list) | same |

Models: **Qwen3.5-9B** (knowledge-rich; needs little — H1/H4),
**Gemma-4-E4B** (script-literate, gloss-hungry; mid-competence point),
**Ministral-3-8B** (fragile rambler — H2/H3), **Llama-3.1-8B** (fourth
family for generality). All in the ~8–9B total-parameter class.

Baselines for comparison (already on disk except where noted): C0, C7
(everywhere); **C3L needs cheap static runs for Gemma-E4B and Llama-8B**
(folded into the same batch). Primary contrasts: **E4-A − C0** (does agency
beat nothing), **E4-A − C3L** (does pull beat the best push), plus
**E4-B − E4-A** (does offering the demonstrated-harmful tools hurt via
temptation — the agency-judgment test).

## 6. Metrics

**Quality**: chrF++, COMET, paired bootstrap on every contrast (as before).

**Agentic behavior** (per model × condition):
- calls/episode, rounds/episode, budget-exhaustion rate
- tool mix (gloss vs analysis vs syntax vs list)
- coverage: % of content tokens whose gloss was fetched (over/under-fetch
  relative to C3L's 80.4% availability)
- protocol failures: unparseable calls, non-sentence words, loops
- context cost: total tokens (initial + tool traffic) vs C3L/C7 prompt sizes

**Metacognition calibration (H4)** — the novel analysis:
- "Needed words" per sentence ≈ content tokens whose C3L gloss keyword is
  absent from the model's own C0 translation (lexical heuristic; documented
  as such) and/or sentences where per-sentence chrF(C0) ≪ chrF(C3L).
- Then: precision/recall of the fetched set against the needed set, by model —
  *do LLMs know what they don't know, and does that calibration scale with
  competence?*

## 7. Implementation plan

| File | Responsibility |
|---|---|
| `experiments/exp4_agentic_retrieval/tool_server.py` | Resolves word queries against the persisted per-sentence analysis (fuzzy match), renders tool returns via the **same functions as `prompt_variants`** |
| `experiments/exp4_agentic_retrieval/agent_loop.py` | Round-synchronous episode state machine over `llm.chat`; per-family tool-call parsing (vLLM parsers); budget enforcement; JSONL episode logs (every turn, every call, every return) |
| `experiments/exp4_agentic_retrieval/run_exp4.py` | Driver: models × conditions, checkpointed per-episode (resumable at episode granularity) |
| `experiments/exp4_agentic_retrieval/score_exp4.py` | Quality + behavior + calibration analysis, appending to its own `ANALYSIS.md` |
| `experiments/exp4_agentic_retrieval/run_exp4.sh` | Batch-safe wrapper (patch + scoring venv + run + score) |

**Phase 0 — gated verification (must pass before the full run)**:

1. **P0-a: tool-calling reliability with thinking disabled.** Qwen3.5's
   template closes the reasoning block (`öffnung\n\n**END**\n\n`) when
   thinking is off; tool-call behavior may be trained in the
   reasoning-then-call pattern. Live test, Qwen3.5-9B first: tools
   declared, ~10 sentences × 2 rounds, thinking OFF. Pass: ≥90% of first
   turns contain a parseable `qwen3_coder` call with valid arguments.
   On failure: test a thinking-ON arm; if the agentic condition requires
   thinking, that is a documented decoding-regime deviation — and a matched
   static arm (C3L with thinking on) is added for fairness, so the tension
   (Tier-2: thinking hurts MT; agentic stacks assume thinking) becomes a
   measured trade-off rather than a confound.
2. **P0-b: template round-trip per family.** All four families: tool section
   renders; calls parse; `role=tool` results re-render correctly in round 2.
   Includes the Mistral `tokenize=False` path end-to-end.
3. **P0-c: budget & turn-cap calibration.** Run ~30 unconstrained episodes
   per model (log-only, no caps beyond a safety ceiling of 32 calls / 12
   rounds / 1024 tokens): observe the natural distributions of calls per
   episode, calls per turn, and assistant-turn lengths (are multi-call plans
   common? do fragile models write verbose pre-call text that a 512 cap
   would truncate mid-call?). Set `MAX_TOOL_CALLS`, `MAX_ROUNDS`, and
   `max_tokens` at ~p95 + margin — caps must never manufacture protocol
   failures that are our artifact, not model behavior.
4. Weak-model failure handling is *data*, not noise: log everything, never
   repair silently.

## 8. Cost estimate

≈ 2 conditions × 4 models × 997 episodes × ~3–5 assistant turns ≈ 30–40k
generations of small contexts — comparable to the Exp-3 factorial. GPU time
~4–6 h for E4-A/E4-B; +1–2 h if E4-C is added. Scoring + behavior analysis
~1 h.

## 9. Open design decisions

Resolved in review (2026-10-02): word addressing = both `word` + optional
`token_id`; E4-C (bulk fetch) dropped; turn cap = 512 uniform (provisional,
with budget — both calibrated in Phase 0-c); models = Qwen-9B, Gemma-E4B,
Ministral-8B, Llama-8B; Qwen parser = `qwen3_coder`; thinking-off tool
reliability gated by Phase 0-a.

Remaining open:
1. Phase-0 outcomes (reliability thresholds, calibrated caps) may adjust
   the condition set — e.g., if thinking-off tool calling fails for Qwen3.5,
   decide between (a) thinking-ON agentic arm + matched static arm, or
   (b) dropping Qwen from E4.
2. E4-B tool set: gloss-only vs gloss+lemma — the lean condition's exact
   composition (does offering `get_lemma` in the lean arm change what we
   attribute to 'selectivity'?). Default: gloss + word_list only.
