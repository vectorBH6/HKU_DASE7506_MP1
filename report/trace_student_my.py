"""Generate interactive torchvista traces of `student_final`.

This script produces three complementary HTML views:

1. ``fig7_torchvista_compressed.html`` — full-model trace with the seven
   transformer Blocks collapsed into one ``Modules (×7)`` container that the
   user can expand on demand to inspect a single Block's internals.
2. ``fig7_torchvista_block_detail.html`` — trace of a single Block; the
   attention sub-module is fully expanded so you can see QK-norm, partial
   RoPE, per-head log-temperature, and the per-token output gate.
3. ``fig7_torchvista_attention_detail.html`` — trace of one Attention module
   alone, showing every operator from ``q_proj`` / ``k_norm`` to the gated
   output projection.

The static, embeddable end-to-end overview lives in
``figures/fig7_end_to_end.png`` and is produced by ``make_figures.py``.

All numerical hyperparameters are read from ``code/configs/student_final.json``:
``depth=7``, ``hidden_dim=640``, ``rope_fraction=0.7``, ``width=288``,
``n_heads=6``, ``n_kv_heads=3``, ``context=256``, ``vocab=2048``. No
GlobalMemoryTokens — the submission is purely a stack of causal pre-norm
blocks.
"""
import json
import sys
from pathlib import Path

import torch
from torchvista import trace_model

CODE_DIR = Path("/home/rvector/MP1_student_starter/code")
sys.path.insert(0, str(CODE_DIR / "student_work"))
sys.path.insert(0, str(CODE_DIR))

from student_final import build_model, Block, Attention  # noqa

OUT_DIR = Path("/home/rvector/MP1_student_starter/report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _head_dim(cfg):
    return cfg["width"] // cfg["heads"]


# ---- Load config and trained weights ------------------------------------
with open(CODE_DIR / "configs" / "student_final.json") as f:
    cfg = json.load(f)

depth = cfg["depth"]
hidden_dim = cfg.get("hidden_dim", 768)
rope_fraction = cfg.get("rope_fraction", 0.5)
n_heads = cfg["heads"]
n_kv_heads = cfg.get("kv_heads", n_heads // 2)
head_dim = _head_dim(cfg)

model = build_model(cfg)
ckpt = torch.load(
    CODE_DIR / "runs" / "student_final" / "checkpoint_best.pt",
    map_location="cpu", weights_only=False,
)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

ids = torch.randint(0, cfg["vocab"], (1, cfg["context"]), dtype=torch.long)


# ---- (1) Full-model trace: the seven Blocks folded into one node -------
top_path = OUT_DIR / "fig7_torchvista_compressed.html"
trace_model(
    model, ids,
    collapse_modules_after_depth=1,    # fold low-level submodules
    forced_module_tracing_depth=3,     # dig into compressed block enough to expand it
    show_compressed_view=True,         # collapse the 7 Blocks into one node
    height=1200,
    export_format="html",
    export_path=str(top_path),
)
print(f"[ok] {top_path.name}  (depth={depth}, width={cfg['width']})")


# ---- (2) Single Block: full attention internals visible -----------------
single_block = Block(
    width=cfg["width"], n_heads=n_heads,
    n_kv_heads=n_kv_heads, head_dim=head_dim,
    context=cfg["context"], dropout=0.0,
    rope_base=cfg.get("rope_base", 10000.0),
    rope_fraction=rope_fraction,
    hidden_dim=hidden_dim,
    residual_dropout=0.0,
).eval()

block_in = torch.randn(1, cfg["context"], cfg["width"])

block_path = OUT_DIR / "fig7_torchvista_block_detail.html"
trace_model(
    single_block, block_in,
    collapse_modules_after_depth=0,    # show full attention internals
    height=1300,
    export_format="html",
    export_path=str(block_path),
)
print(f"[ok] {block_path.name}  (hidden_dim={hidden_dim}, rope_fraction={rope_fraction})")


# ---- (3) Attention alone: every operator visible ------------------------
attn = Attention(
    width=cfg["width"], n_heads=n_heads,
    n_kv_heads=n_kv_heads, head_dim=head_dim,
    context=cfg["context"], dropout=0.0,
    rope_base=cfg.get("rope_base", 10000.0),
    rope_fraction=rope_fraction,
).eval()

attn_path = OUT_DIR / "fig7_torchvista_attention_detail.html"
trace_model(
    attn, block_in,
    collapse_modules_after_depth=0,
    height=1200,
    export_format="html",
    export_path=str(attn_path),
)
print(f"[ok] {attn_path.name}  (n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
      f"head_dim={head_dim})")