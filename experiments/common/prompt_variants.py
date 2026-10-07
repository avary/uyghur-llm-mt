"""
prompt_variants.py - Build the 2^3 component-ablation prompt variants (C0-C7)
from the persisted FLORES dev analysis (`stats/flores_dev/dev.results.jsonl`).

Conditions (full factorial over the three pipeline components):
    C0  baseline        : raw sentence only (zero-shot)
    C1  syntax          : clause skeleton + dependency roles
    C2  morph           : lemma + ULY + morphological features
    C3  dict            : surface-form dictionary glosses (+ surface ULY)
    C4  syntax_morph    : skeleton + lemma/features
    C5  syntax_dict     : skeleton + surface-form glosses
    C6  morph_dict      : lemma/features + glosses
    C7  full            : everything (must byte-match the live pipeline prompt)

C7-parity notes (verified against `pipeline/prompt.py`):
- `morph_summary` is rebuilt from the persisted `features` list: the pipeline
  stores enclitic readings as entries starting with "Lemma: " inside `features`
  (see `Morphology.parse_reading`), and `TokenAnnotation.morph_summary` splits
  on exactly that prefix.
- `dict_gloss` mirrors `TokenAnnotation.dict_gloss`: short gloss on hit, "-" for
  punctuation, "≈ gloss" (max 2) for spelling-near candidates, "[lemma]" on miss.
- Dictionary-only conditions (C3, C5) deliberately *cannot* use the
  disambiguated lemma (that would leak the morphology component). They query the
  Schwarz dictionary with the raw surface form through the real
  `UyghurDictionary.generate_stem_variants` logic, and skip the pipeline's
  proper-noun exclusion for near-matches (POS is part of morphology). ULY in
  these conditions is the surface-form transliteration.
"""

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Standalone import of pipeline/dictionary.py (avoids pipeline/__init__.py,
# which pulls in torch/diaparser). Only needs sqlite3 + stdlib.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DICT_PY = os.path.join(_REPO_ROOT, "pipeline", "dictionary.py")

_spec = importlib.util.spec_from_file_location("uyghur_dictionary_standalone", _DICT_PY)
_dict_mod = importlib.util.module_from_spec(_spec)
sys.modules["uyghur_dictionary_standalone"] = _dict_mod
_spec.loader.exec_module(_dict_mod)

UyghurDictionary = _dict_mod.UyghurDictionary
uey_to_uly = _dict_mod.uey_to_uly

FLORES_RESULTS_PATH = os.path.join(_REPO_ROOT, "stats", "flores_dev", "dev.results.jsonl")
DICTIONARY_DB = os.path.join(_REPO_ROOT, "dictionary", "uig_eng_dict_complete.sqlite")
_SURFACE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".surface_dict_cache.json")


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    cid: str          # "C0" .. "C7"
    name: str         # file-safe slug, e.g. "c3_dict"
    syntax: bool
    morph: bool
    dictionary: bool
    description: str
    lemma_keyed: bool = False  # dictionary column from persisted lemma lookups
                               # (C3L) instead of surface lookups (C3)


CONDITIONS: List[Condition] = [
    Condition("C0", "c0_baseline",    False, False, False, "Zero-shot baseline (raw sentence only)"),
    Condition("C1", "c1_syntax",      True,  False, False, "+ Dependency syntax & clause skeleton"),
    Condition("C2", "c2_morph",       False, True,  False, "+ Disambiguated morphology (lemma, ULY, features)"),
    Condition("C3", "c3_dict",        False, False, True,  "+ Surface-form dictionary glosses"),
    Condition("C4", "c4_syntax_morph",True,  True,  False, "+ Syntax + Morphology"),
    Condition("C5", "c5_syntax_dict", True,  False, True,  "+ Syntax + Dictionary"),
    Condition("C6", "c6_morph_dict",  False, True,  True,  "+ Morphology + Dictionary"),
    Condition("C7", "c7_full",        True,  True,  True,  "Full pipeline (Syntax + Morphology + Dictionary)"),
]

# Special follow-up conditions (Exp 3 studies). Kept OUT of `CONDITIONS` so the
# exp1/exp2 drivers' default condition lists are unchanged.
#   S   : presentation-only — the exact C0 prompt with the sentence as a
#         numbered token list (no roles/lemmas/glosses/guidance). Tests the
#         "clean word separation" hypothesis.
#   C3L : dict-only table identical to C3, but the dictionary column comes
#         from the pipeline's LEMMA-keyed lookups (80.4% coverage) instead of
#         surface-form lookups (44.0%). Tests the lemma-vs-surface ablation.
SPECIAL_CONDITIONS: List[Condition] = [
    Condition("S", "s_segmented", False, False, False,
              "Zero-shot, words in a numbered list (presentation-only)"),
    Condition("C3L", "c3_dict_lemma", False, False, True,
              "+ Dictionary (lemma-keyed lookups)", lemma_keyed=True),
]

CONDITIONS_BY_ID = {c.cid: c for c in CONDITIONS + SPECIAL_CONDITIONS}


def get_condition(cid_or_name: str) -> Condition:
    """Accept 'C7', 'c7_full', 'S', 'C3L', or 'full'."""
    key = cid_or_name.strip()
    if key in CONDITIONS_BY_ID:
        return CONDITIONS_BY_ID[key]
    for c in CONDITIONS + SPECIAL_CONDITIONS:
        if c.name == key or c.name.endswith("_" + key):
            return c
    raise KeyError(f"Unknown condition: {cid_or_name!r}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_flores_records(results_path: str = FLORES_RESULTS_PATH) -> List[dict]:
    """Load the persisted per-token analysis, one record per FLORES dev sentence."""
    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_references(ref_path: str) -> List[str]:
    with open(ref_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Token-level renderers (C7-parity with pipeline/prompt.py)
# ---------------------------------------------------------------------------
def _morph_summary(tok: dict) -> str:
    """Mirror TokenAnnotation.morph_summary from a persisted per_token record."""
    if tok["pos"] == "Punctuation":
        return "Punctuation"
    feats = tok["features"]
    feats_clean = [f for f in feats if not f.startswith("Lemma:")]
    enclitics = [f for f in feats if f.startswith("Lemma:")]
    base = ", ".join(feats_clean) if feats_clean else tok["pos"]
    if enclitics:
        return f"{base} ({' + '.join(enclitics)})"
    return base


def _definition_from_gloss(gloss: str) -> Optional[str]:
    """'kün (n.): Sun; Day; Life' -> 'Sun; Day; Life', else None.

    Mirrors the live pipeline's clause skeleton, which renders definitions[0]
    only when the entry has definitions. Synthetic universal entries (digits,
    Latin tokens) have short_gloss like '864 (num.)' with no ': ' part and
    therefore no definition string.
    """
    if ": " in gloss:
        return gloss.split(": ", 1)[1].strip()
    return None


class GlossProvider:
    """Per-condition dictionary glosses for one sentence's token list.

    - morph conditions: reuse the pipeline's lemma-based glosses/near-matches
      exactly as persisted in dev.results.jsonl (zero divergence from C7).
    - dict-only conditions: query the real Schwarz dictionary by surface form.
      Surface lookups (esp. the edit-distance-1 near search over long inflected
      forms) cost seconds per new form, so results are memoized in a
      class-level cache AND persisted to `.surface_dict_cache.json` next to
      this file — shared across conditions and across runs.
    """

    _disk_cache: Optional[Dict] = None   # form -> {"gloss": str|None, "near": [...]}
    _dirty_since_save = 0

    @classmethod
    def _cache(cls) -> Dict:
        if cls._disk_cache is None:
            path = _SURFACE_CACHE_PATH
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cls._disk_cache = json.load(f)
            else:
                cls._disk_cache = {}
        return cls._disk_cache

    @classmethod
    def _save_cache(cls, force: bool = False) -> None:
        if cls._disk_cache is None:
            return
        if not force and cls._dirty_since_save < 500:
            return
        tmp = _SURFACE_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cls._disk_cache, f, ensure_ascii=False)
        os.replace(tmp, _SURFACE_CACHE_PATH)
        cls._dirty_since_save = 0

    def __init__(self, use_dictionary: bool, use_morph: bool,
                 db_path: str = DICTIONARY_DB):
        self.use_dictionary = use_dictionary
        self.use_morph = use_morph
        self._dict = None
        if use_dictionary and not use_morph:
            self._dict = UyghurDictionary(db_path)

    def lookup_surface(self, form: str) -> tuple[Optional[str], List[dict]]:
        """Surface-form lookup for dict-only conditions.

        Returns (gloss_or_None, near_entries) — memoized on disk.
        """
        cache = self._cache()
        if form in cache:
            hit = cache[form]
            return hit["gloss"], hit["near"]

        entry = self._dict.lookup(form, is_verb=False, surface=form)
        gloss = entry.short_gloss if entry is not None else None
        near: List[dict] = []
        if entry is None:
            near = [
                {"headword": e.headword_uey, "uly": e.headword_uly,
                 "gloss": e.short_gloss}
                for e in self._dict.lookup_near(form, surface=form)[:2]
            ]
        cache[form] = {"gloss": gloss, "near": near}
        GlossProvider._dirty_since_save += 1
        GlossProvider._save_cache()
        return gloss, near

    def close(self):
        if self._dict is not None:
            self._dict.close()
            self._dict = None
        GlossProvider._save_cache(force=True)


def _dict_gloss(tok: dict, gloss: Optional[str], near: List[dict]) -> str:
    """Mirror TokenAnnotation.dict_gloss."""
    if tok["pos"] == "Punctuation":
        return "-"
    if gloss is not None:
        return gloss
    if near:
        return "; ".join("≈ " + n["gloss"] for n in near[:2])
    return f"[{tok['lemma'] if tok['lemma'] else tok['form']}]"


def _uly(tok: dict, use_morph: bool) -> str:
    if use_morph:
        return tok["uly"]  # persisted lemma-based ULY (falls back to form ULY)
    return uey_to_uly(tok["form"]) or tok["uly"]


# ---------------------------------------------------------------------------
# Clause skeleton (mirrors PromptBuilder._extract_clause_structure)
# ---------------------------------------------------------------------------
def _clause_lines(tokens: List[dict], glosses: List[str], use_morph: bool,
                  use_dict: bool) -> List[str]:
    id_to_form = {t["id"]: t["form"] for t in tokens}
    id_to_form[0] = "ROOT"
    roots = [t for t in tokens if t["deprel"] == "root" or t["head_id"] == 0]
    subjects = [t for t in tokens if "subj" in t["deprel"]]
    objects = [t for t in tokens if t["deprel"] in ("obj", "iobj")]
    obliques = [t for t in tokens if t["deprel"] in ("obl", "advmod", "case", "case:dat")]

    def entry_gloss(t):
        if use_dict and t["has_dict"]:
            d = _definition_from_gloss(t["gloss"])
            if d is not None:
                return f" ('{d}')"
        return ""

    root_summary = []
    for r in roots:
        if use_morph:
            root_summary.append(
                f"{r['form']} [Lemma: {r['lemma']}{entry_gloss(r)}, "
                f"Features: {_morph_summary(r)}]"
            )
        else:
            root_summary.append(f"{r['form']}{entry_gloss(r)}")

    def fmt(t):
        return (f"{t['form']}{entry_gloss(t)} "
                f"({t['deprel']} -> #{t['head_id']} {id_to_form.get(t['head_id'], 'ROOT')})")

    lines = []
    if root_summary:
        lines.append(f"- **Main Predicate / Action**: {' | '.join(root_summary)}")
    if subjects:
        lines.append(f"- **Subject(s)**: {', '.join(fmt(t) for t in subjects)}")
    if objects:
        lines.append(f"- **Object(s)**: {', '.join(fmt(t) for t in objects)}")
    if obliques:
        lines.append(f"- **Adverbial / Obliques**: {', '.join(fmt(t) for t in obliques)}")
    return lines or ["- (Simple clause)"]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(cond: Condition, sentence: str, tokens: List[dict],
                 gloss_provider: Optional[GlossProvider] = None) -> str:
    """Render one ablation-condition prompt for one sentence.

    `tokens` = the per_token list from dev.results.jsonl for this sentence.
    """
    # -- S: presentation-only (C0 text, sentence as numbered token list) --------
    if cond.cid == "S":
        tok_lines = "\n".join(f"{t['id']}. {t['form']}" for t in tokens)
        return (
            "You are an expert Uyghur-to-English translator.\n"
            "Translate the following Uyghur sentence into fluent, accurate English.\n\n"
            "### Uyghur Source Sentence:\n"
            f"{tok_lines}\n\n"
            "### English Translation:"
        )

    # -- C0: zero-shot -------------------------------------------------------
    if not (cond.syntax or cond.morph or cond.dictionary):
        return (
            "You are an expert Uyghur-to-English translator.\n"
            "Translate the following Uyghur sentence into fluent, accurate English.\n\n"
            "### Uyghur Source Sentence:\n"
            f"{sentence}\n\n"
            "### English Translation:"
        )

    # -- Dictionary glosses per token ----------------------------------------
    glosses: List[Optional[str]] = [None] * len(tokens)
    nears: List[List[dict]] = [[] for _ in tokens]
    if cond.dictionary:
        if cond.morph or cond.lemma_keyed:
            # Pipeline's own lemma-based hits (C6/C7/C3L parity).
            for i, t in enumerate(tokens):
                glosses[i] = t["gloss"] if t["has_dict"] else None
                nears[i] = t["near"] or []
        else:
            assert gloss_provider is not None, \
                "dict-only conditions need a GlossProvider (surface-form lookups)"
            for i, t in enumerate(tokens):
                if t["pos"] == "Punctuation":
                    continue
                gloss, near = gloss_provider.lookup_surface(t["form"])
                glosses[i] = gloss
                nears[i] = near

    gloss_strs = [_dict_gloss(t, glosses[i], nears[i]) for i, t in enumerate(tokens)]

    # -- Table ---------------------------------------------------------------
    id_to_form = {t["id"]: t["form"] for t in tokens}
    id_to_form[0] = "ROOT"
    header_cells = ["ID", "Word"]
    if cond.morph:
        header_cells += ["Lemma", "ULY", "Morphological Features"]
    elif cond.dictionary:
        header_cells += ["ULY"]
    if cond.syntax:
        header_cells += ["Syntactic Role (Dependency)"]
    if cond.dictionary:
        header_cells += ["Dictionary Translation"]

    rows = ["| " + " | ".join(header_cells) + " |",
            "|" + "|".join(["---"] * len(header_cells)) + "|"]
    for i, t in enumerate(tokens):
        cells = [str(t["id"]), t["form"]]
        if cond.morph:
            cells += [t["lemma"], _uly(t, True), _morph_summary(t)]
        elif cond.dictionary:
            cells += [_uly(t, False)]
        if cond.syntax:
            head_form = id_to_form.get(t["head_id"], "ROOT")
            cells.append(f"{t['deprel']} -> #{t['head_id']} ({head_form})")
        if cond.dictionary:
            cells.append(gloss_strs[i])
        rows.append("| " + " | ".join(cells) + " |")
    table_str = "\n".join(rows)

    # -- Clause skeleton ------------------------------------------------------
    clause_str = "\n".join(_clause_lines(tokens, gloss_strs, cond.morph, cond.dictionary))

    # -- Guidance -------------------------------------------------------------
    guidance = []
    if cond.morph:
        guidance += [
            "1. Preserve all grammatical inflections (tense, mood, voice, polarity, person/number agreements).",
            "2. Resolve pro-drop (omitted subjects) using verbal agreement suffixes where applicable.",
        ]
    if cond.dictionary:
        guidance.append(
            f"{len(guidance) + 1}. Use the word-level dictionary definitions in proper contextual syntax."
        )
    guidance.append(
        f"{len(guidance) + 1}. Provide only the final fluent English translation without conversational preamble."
    )
    if cond.dictionary and any(nears):
        guidance.append(
            f"{len(guidance) + 1}. Tokens marked '≈' have no exact dictionary entry; the listed candidates are "
            "spelling-near matches - verify meaning and part of speech against the "
            "sentence context before using them."
        )
    guidance_str = "\n".join(guidance)

    # -- Assemble -------------------------------------------------------------
    sections = [
        "You are an expert Uyghur-to-English translator and computational linguist.",
        "Translate the following Uyghur sentence into fluent, accurate English "
        "using the provided linguistic analysis.",
        "",
        "### Uyghur Source Sentence:",
        sentence,
    ]
    if cond.syntax:
        sections += ["", "### Clause Architecture & Syntactic Skeleton:", clause_str]
    sections += [
        "",
        "### Word-by-Word Morphosyntactic Breakdown & Dictionary Entries:",
        table_str,
        "",
        "### Translation Guidance:",
        guidance_str,
        "",
        "### English Translation:",
    ]
    return "\n".join(sections)


def build_prompts_for_condition(cond: Condition, records: List[dict]) -> List[str]:
    """Render the prompt list for one condition across all sentences."""
    provider = GlossProvider(cond.dictionary, cond.morph)
    try:
        return [build_prompt(cond, rec["sentence"], rec["per_token"], provider)
                for rec in records]
    finally:
        provider.close()
