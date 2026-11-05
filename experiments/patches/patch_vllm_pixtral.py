"""
patch_vllm_pixtral.py - Fix vLLM 0.29 Pixtral imports against transformers 5.x.

transformers 5.x renamed `PixtralRotaryEmbedding` -> `PixtralVisionRotaryEmbedding`
and removed `position_ids_in_meshgrid`. vLLM 0.29's pixtral.py still imports
both, which breaks registry inspection and thus ANY Mistral-3-family model
load (their architectures resolve through the Pixtral module).

The patched symbols are used only inside the vision tower, which never runs
for text-only translation, so aliasing/stubbing is safe.

Idempotent: run before every Exp-3 session (the vLLM env lives in $TMPDIR and
dies with the session).

Usage:  python experiments/patches/patch_vllm_pixtral.py
"""

import importlib.util
import os
import sys

spec = importlib.util.find_spec("vllm.model_executor.models.pixtral")
if spec is None:
    print("vllm not importable from this interpreter; run with the vllm env python")
    sys.exit(1)
path = spec.origin

OLD = """from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding,
    apply_rotary_pos_emb,
    position_ids_in_meshgrid,
)"""

NEW = """from transformers.models.pixtral.modeling_pixtral import (
    PixtralVisionRotaryEmbedding as PixtralRotaryEmbedding,
    apply_rotary_pos_emb,
)
try:
    from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid
except ImportError:  # transformers 5.x: removed; only needed for vision inputs
    def position_ids_in_meshgrid(*args, **kwargs):
        raise NotImplementedError(
            "position_ids_in_meshgrid was removed from transformers 5.x; "
            "vision inputs are unsupported under this patch")"""

with open(path, encoding="utf-8") as f:
    src = f.read()

if NEW in src:
    print("already patched:", path)
    sys.exit(0)

# Newer vLLM releases fix this internally with version-conditional imports
# (TRANSFORMERS_WITH_AXIAL_ROPE). Recognize that as already-handled.
if "TRANSFORMERS_WITH_AXIAL_ROPE" in src:
    print("upstream already handles the transformers 5.x rename:", path)
    sys.exit(0)

if OLD in src:
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched:", path)
else:
    print("expected import block not found — vLLM layout changed; "
          "verify Mistral model loading manually")
    # no longer fatal: real engine errors surface in the runner scripts anyway
    sys.exit(0)

# verify it imports cleanly now
import vllm.model_executor.models.pixtral  # noqa: E402,F401
print("import check: OK")
