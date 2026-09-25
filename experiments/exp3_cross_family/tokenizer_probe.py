"""
tokenizer_probe.py - Quantify Arabic-script Uyghur exposure per model family.

Proxy: tokenizer compression on FLORES dev Uyghur (chars/token, tokens/word)
vs English, plus a single-token test for common Uyghur function words.
Byte-fallback tokenizers would sit near ~0.5 chars/token (2 UTF-8 bytes per
Arabic char); values near 1.5+ indicate real Arabic-script merges in vocab.

Run (vLLM env, CPU only):
  "$TMPDIR/vllm/bin/python" experiments/exp3_cross_family/tokenizer_probe.py
Writes tokenizer_probe.json next to this script.
"""

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer
from experiments.common.prompt_variants import uey_to_uly

MODELS = [
    ("Qwen", "qwen3.5-2b", "Qwen/Qwen3.5-2B"),
    ("Ministral", "ministral-3-3b", "mistralai/Ministral-3-3B-Instruct-2512"),
    ("Ministral", "ministral-3-14b", "mistralai/Ministral-3-14B-Instruct-2512"),
    ("Llama", "llama-3.2-1b", "unsloth/Llama-3.2-1B-Instruct"),
    ("Llama", "llama-3.2-3b", "unsloth/Llama-3.2-3B-Instruct"),
    ("Gemma", "gemma-4-e2b", "google/gemma-4-E2B-it"),
    ("Gemma", "gemma-4-12b", "google/gemma-4-12B-it"),
]

PROBES = ["ۋە", "بىر", "بۇ", "ھەم", "بار", "يوق", "كىشى", "كۈن", "سۇ"]


def main():
    uig = [l.strip() for l in open(
        os.path.join(_REPO_ROOT, "flores200_dataset", "uig_Arab.dev"), encoding="utf-8")]
    eng = [l.strip() for l in open(
        os.path.join(_REPO_ROOT, "flores200_dataset", "eng_Latn.dev"), encoding="utf-8")]
    uly = [uey_to_uly(l) for l in uig]

    out = []
    for family, tag, mid in MODELS:
        tok = AutoTokenizer.from_pretrained(mid)

        def stats(lines):
            n_c = sum(len(l) for l in lines)
            n_t = sum(len(tok(l, add_special_tokens=False)["input_ids"]) for l in lines)
            n_w = sum(len(l.split()) for l in lines)
            return round(n_c / n_t, 2), round(n_t / n_w, 2)

        en_ct, _ = stats(eng)
        ui_ct, ui_tw = stats(uig)
        ul_ct, _ = stats(uly)
        single = [w for w in PROBES
                  if len(tok(w, add_special_tokens=False)["input_ids"]) == 1]
        out.append({
            "family": family, "tag": tag, "model": mid,
            "en_chars_per_token": en_ct,
            "uig_chars_per_token": ui_ct, "uig_tokens_per_word": ui_tw,
            "uly_chars_per_token": ul_ct,
            "single_token_function_words": single,
        })
        print(f"{family:<10} {tag:<16} uig c/t: {ui_ct:>5}  uig t/w: {ui_tw:>5}  "
              f"1-tok probes: {len(single)}/{len(PROBES)} {single}")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
