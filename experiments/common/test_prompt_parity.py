"""
test_prompt_parity.py - Verify that the C7 prompt rebuilt from the persisted
dev.results.jsonl is byte-identical to the live pipeline's PromptBuilder output.

Works by loading pipeline/prompt.py standalone (stubbing diaparser, which is
not installed in the vLLM env), reconstructing ParsedSentence / MorphToken /
DictEntry objects from the persisted per-token records, and diffing
PromptBuilder._render_markdown_table_prompt against
prompt_variants.build_prompt(C7).

Run (vLLM env):
  conda run -p "$TMPDIR/vllm" python experiments/common/test_prompt_parity.py
"""

import importlib.util
import json
import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

# --- stub diaparser so pipeline.parser imports cleanly ----------------------
dia = types.ModuleType("diaparser")
parsers = types.ModuleType("diaparser.parsers")
parsers.Parser = object
dia.parsers = parsers
sys.modules.setdefault("diaparser", dia)
sys.modules.setdefault("diaparser.parsers", parsers)

# --- load pipeline modules as a synthetic package ---------------------------
pkg = types.ModuleType("pl_live")
pkg.__path__ = [os.path.join(_REPO_ROOT, "pipeline")]
sys.modules["pl_live"] = pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"pl_live.{name}", os.path.join(_REPO_ROOT, "pipeline", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"pl_live.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


dictionary_mod = _load("dictionary")
parser_mod = _load("parser")
morphology_mod = _load("morphology")
prompt_mod = _load("prompt")

from experiments.common.prompt_variants import (  # noqa: E402
    CONDITIONS_BY_ID, load_flores_records, build_prompt,
)

ParsedSentence = parser_mod.ParsedSentence
ParsedToken = parser_mod.ParsedToken
MorphToken = morphology_mod.MorphToken
DictEntry = dictionary_mod.DictEntry
TokenAnnotation = prompt_mod.TokenAnnotation
PromptBuilder = prompt_mod.PromptBuilder


def entry_from_gloss(gloss: str) -> DictEntry:
    """'kün (n.): Sun; Day; Life' -> DictEntry.

    The defs string is kept as ONE definition (DB definitions can themselves
    contain semicolons); short_gloss then reproduces the stored gloss exactly,
    and definitions[0] matches what the live clause skeleton renders.
    """
    head, _, defs = gloss.partition(": ")
    uly, _, pos = head.partition(" (")
    pos = pos.rstrip(")") or None
    return DictEntry(
        headword_uey="",
        headword_uly=uly,
        pos=pos,
        definitions=[defs] if defs else [],
    )


def rebuild_pipeline_objects(rec: dict):
    """Persisted per_token -> (ParsedSentence, [MorphToken], [DictEntry], near)."""
    tokens = []
    morphs = []
    entries = []
    nears = []
    for t in rec["per_token"]:
        tokens.append(ParsedToken(
            id=t["id"], form=t["form"], head=t["head_id"], deprel=t["deprel"],
            upos=t["upos"],
        ))
        feats = list(t["features"])
        if t["pos"] == "Punctuation":
            lemma, pos = t["form"], "Punctuation"
        else:
            lemma, pos = t["lemma"], t["pos"]
        morphs.append(MorphToken(
            id=t["id"], form=t["form"], lemma=lemma, pos=pos, features=feats,
            enclitics=[f for f in feats if f.startswith("Lemma:")],
        ))
        if t["has_dict"]:
            entries.append(entry_from_gloss(t["gloss"]))
        else:
            entries.append(None)
        nears.append([entry_from_gloss(n["gloss"]) for n in (t["near"] or [])])

    ps = ParsedSentence(
        text=rec["sentence"], tokens=tokens,
        # to_conllu() is not used by the prompt renderer
    )
    return ps, morphs, entries, nears


def live_prompt(rec: dict) -> str:
    ps, morphs, entries, nears = rebuild_pipeline_objects(rec)
    analysis = PromptBuilder.build_analysis(ps, morphs, entries, near_entries=nears)
    return PromptBuilder.build_prompt(analysis, format_type="markdown_table")


def main(n_sentences: int = 25):
    records = load_flores_records()
    # deterministic diverse sample: first N + stride across the set
    stride = max(1, len(records) // n_sentences)
    sample = [records[i] for i in range(0, len(records), stride)][:n_sentences]

    mismatches = 0
    for idx, rec in enumerate(sample):
        ours = build_prompt(CONDITIONS_BY_ID["C7"], rec["sentence"], rec["per_token"])
        theirs = live_prompt(rec)
        if ours != theirs:
            mismatches += 1
            print(f"MISMATCH at sample {idx} (sentence id {rec['sentence'][:30]}...)")
            a, b = ours.splitlines(), theirs.splitlines()
            for la, lb in zip(a, b):
                if la != lb:
                    print(f"  ours  : {la}")
                    print(f"  live  : {lb}")
            if len(a) != len(b):
                print(f"  length differs: {len(a)} vs {len(b)}")
    total = len(sample)
    print(f"\nparity check: {total - mismatches}/{total} C7 prompts byte-identical "
          f"to live PromptBuilder output")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
