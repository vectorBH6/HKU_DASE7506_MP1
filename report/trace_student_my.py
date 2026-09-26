"""Trace student_my with compressed_view=True so the 8 transformer Blocks
fold into one 'Modules (×8)' container that the user can expand on demand
to inspect a single Block's internals.
"""
import json
import sys
from pathlib import Path

import torch
from torchvista import trace_model

CODE_DIR = Path("/home/rvector/MP1_student_starter/code")
sys.path.insert(0, str(CODE_DIR / "student_work"))
sys.path.insert(0, str(CODE_DIR))

from student_my import build_model, Block, Attention, GlobalMemoryTokens  # noqa

OUT_DIR = Path("/home/rvector/MP1_student_starter/report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load model ---------------------------------------------------------
with open(CODE_DIR / "configs" / "student_my.json") as f:
    cfg = json.load(f)
model = build_model(cfg)
ckpt = torch.load(
    CODE_DIR / "runs" / "student_my" / "checkpoint_best.pt",
    map_location="cpu", weights_only=False,
)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

ids = torch.randint(0, cfg["vocab"], (1, 256), dtype=torch.long)

# ---- Main view: 8 Blocks collapsed into one repeat container ------------
top_path = OUT_DIR / "fig5_torchvista_compressed.html"
trace_model(
    model, ids,
    collapse_modules_after_depth=1,    # fold low-level submodules
    forced_module_tracing_depth=3,     # dig into compressed block enough to expand it
    show_compressed_view=True,         # ← this collapses the 8 Blocks into one node
    height=1200,
    export_format="html",
    export_path=str(top_path),
)
print(f"[ok] {top_path.name}")

# ---- Single-Block detail: just one Block, so no repeats are needed ------
single_block = Block(
    width=cfg["width"], n_heads=cfg["heads"],
    n_kv_heads=cfg.get("kv_heads", cfg["heads"] // 2),
    head_dim=cfg["width"] // cfg["heads"],
    context=cfg["context"], dropout=0.0,
    rope_base=cfg.get("rope_base", 10000.0),
    rope_fraction=cfg.get("rope_fraction", 0.5),
    hidden_dim=cfg.get("hidden_dim", 768),
).eval()

block_in = torch.randn(1, 256, cfg["width"])

block_path = OUT_DIR / "fig5_torchvista_block_detail.html"
trace_model(
    single_block, block_in,
    collapse_modules_after_depth=0,    # show full attention internals
    height=1300,
    export_format="html",
    export_path=str(block_path),
)
print(f"[ok] {block_path.name}")

# ---- Attention detail: QK-norm / gate / log_temperature / RoPE ----------
attn = Attention(
    width=cfg["width"], n_heads=cfg["heads"],
    n_kv_heads=cfg.get("kv_heads", cfg["heads"] // 2),
    head_dim=cfg["width"] // cfg["heads"],
    context=cfg["context"], dropout=0.0,
    rope_base=cfg.get("rope_base", 10000.0),
    rope_fraction=cfg.get("rope_fraction", 0.5),
).eval()

attn_path = OUT_DIR / "fig5_torchvista_attention_detail.html"
trace_model(
    attn, block_in,
    collapse_modules_after_depth=0,
    height=1200,
    export_format="html",
    export_path=str(attn_path),
)
print(f"[ok] {attn_path.name}")

# ---- Global memory tokens detail ---------------------------------------
gmt = GlobalMemoryTokens(
    width=cfg["width"],
    n_global_tokens=cfg.get("n_global_tokens", 12),
).eval()

gmt_path = OUT_DIR / "fig5_torchvista_gmt_detail.html"
trace_model(
    gmt, block_in,
    collapse_modules_after_depth=0,
    height=1200,
    export_format="html",
    export_path=str(gmt_path),
)
print(f"[ok] {gmt_path.name}")
