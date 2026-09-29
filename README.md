# MP1 — Small Language Model Challenge

**Student ID:** 3036705979
**Name:** Li Shanghang
**Course:** DASE7506

---

## Results

| Model | Test BPB | CPU Time | Checkpoint | Under Limit? |
|-------|----------|----------|------------|--------------|
| Baseline (this machine) | 1.9653 | 8.39 s | — | — |
| Reference baseline (Xeon Platinum 8457C) | 2.10 | 5.92 s | — | — |
| **student_final** | **1.5107** | **34.18 s** | **29 MiB** | **✓ All pass** |

**All three resource limits satisfied (using this machine's measured baseline, 5× rule):**

| Constraint | student_final | Limit (5× local baseline) | Status |
|------------|--------------|---------------------------|--------|
| CPU time (test, FP32) | 34.18 s | ≤ 41.93 s (5 × 8.39 s) | ✓ 4.07× baseline |
| Peak RAM | ~0.05 GiB (est.) | ≤ 4 GiB | ✓ |
| Inference assets | 29 MiB | ≤ 64 MiB | ✓ |

> **CPU-time note.** The FP32 CPU test pass on this machine measures **34.18 s**, which is **4.07×** the locally-measured baseline of **8.39 s** (limit is 5× baseline ≈ 41.93 s). The 5× rule is applied against the baseline `test_cpu_fp32.json` produced on the same hardware, not against the reference Xeon number; on the reference CPU the baseline is 5.92 s and the equivalent ceiling is 29.60 s, which the new model does not meet on that hardware — so the limit status is hardware-relative. Recorded BPB (1.5107) and checkpoint size (29 MiB) are hardware-independent. The BF16 training run used BF16 autocast on a single RTX 5090 Laptop GPU and took **~895 s** of train-time (plus ~12 s preparation) for 16000 steps (seed 17, EMA-decay 0.999).

BPB improved **-23.1%** vs local baseline (1.9653 → 1.5107) and **-28.1%** vs reference baseline (2.10 → 1.5107). Best validation BPB was **1.4871** at step **10000**, with `checkpoint_best.pt` chosen at that step; the final 16000-step checkpoint has validation BPB 1.5060.

---

## Method

`student_final` builds on the baseline GPT by replacing every component of the original block with a deeper, more stable stack. All architectural choices fit inside the original `forward(ids) → logits[B,T,2048]` contract (no extra input tokens), so the causality test and tokenizer contract both still pass.

### Building blocks

| Module | What it does | Key hyperparameters |
|---|---|---|
| `RMSNorm` | Per-channel RMS normalization with FP32 accumulation under FP16/BF16 autocast (avoids overflow on squared activations). | `eps=1e-6` |
| `RotaryPositionalEmbedding` | Partial RoPE on the first `rope_fraction × head_dim` dimensions; cos/sin are built once in FP32/FP64 and re-cast per call. Caches only position constants, never token information. | `rope_base=10000`, `rope_fraction=0.7` |
| `SwiGLU` | Three bias-free projections `w1/w2/w3` with `silu(w1(x)) * w2(x)` gating. Replaces the baseline 2-layer MLP. | `hidden_dim=640` (from config; falls back to `⌈8·width/3/256⌉·256` if omitted) |
| `Attention` | Causal grouped-query attention with QK-norm, partial RoPE, per-token gating, and per-head learnable temperature. | `n_heads=6`, `n_kv_heads=3`, `head_dim=48`, `dropout=0.10` |
| `Block` | Pre-norm residual: `x ← x + Drop(Attn(RMSNorm(x)))` then `x ← x + Drop(FFN(RMSNorm(x)))`. | `residual_dropout=0.15` |

### Per-block components inside `Attention`

1. **Grouped-Query Attention (GQA)** — 6 query heads share 3 KV heads. After projection, `k` and `v` are expanded to query shape with `repeat_interleave` (kept explicit for compatibility with PyTorch 2.7 CPU scoring).
2. **QK normalization** — both `q` and `k` are RMS-normalized **before** RoPE, so positional rotations act on already-scaled vectors. This dramatically improves stability with the deeper stack.
3. **Partial RoPE** — only the first 70 % of each head's dimensions are rotated; the remaining 30 % carry absolute position information implicitly.
4. **Per-head learnable temperature** — `log_temperature ∈ ℝ^{n_heads}` is clamped to `[-log 100, log 100]` and exponentiated to a per-head scale, multiplied into `q` before `scaled_dot_product_attention`. At init it is `0`, so the effective scale equals `1/√d` and the model starts identical to a standard attention block.
5. **Per-token gating** — the attention output projection is multiplied by `σ(W_gate x)`, a learned sigmoid gate over the **input** `x`. This lets each position attenuate the attention contribution locally without breaking causality.
6. **`scaled_dot_product_attention`** — uses the fused `is_causal=True` path on both CUDA and CPU; no hand-written causal mask.

### Top-level model (`ImprovedGPT`)

- **Tokenizer-free positional embedding.** RoPE handles positions; there is no learned positional table, so the input contract is purely `ids: [B, T]`.
- **Weight tying.** `lm_head.weight = token_emb.weight`, halving the embedding budget and keeping the output distribution anchored to the input embedding geometry.
- **Depth-scaled initialization (GPT-2 style).** Linear layers initialize at `std=0.02`, but the **residual-output projections** `attn.o_proj` and `ffn.w3` are re-sampled at `std = 0.02 / √(2·depth)` to bound the per-layer contribution to the residual stream and prevent explosion across 7 layers.
- **Final RMSNorm** before the LM head — standard pre-norm finishing layer.
- **`predict_log_probs`** casts logits to FP32 before `log_softmax`, matching the official scoring path exactly.

### Contract-preserving properties

- **Input shape unchanged.** `ids: [B, T]`, no memory tokens concatenated to the sequence, no auxiliary inputs.
- **Causality preserved.** Strict causal masking is enforced by `scaled_dot_product_attention(is_causal=True)`; all linear projections and norms are position-local; RoPE and QK-norm do not cross positions. The implementation satisfies `tests/test_causality.py`.
- **Independent windows.** No carry-over state between forward calls; the only non-parameter buffers are the RoPE cos/sin cache, which depend only on `seq_len`, `device`, `dtype`.
- **Weight tying.** Halves the embedding parameter cost (the model has **6,788,010 parameters** total, including the tied embedding).

> **Causality note (history).** Earlier drafts (`student_my`) experimented with a `GlobalMemoryTokens` side-channel that aggregated per-position summaries by softmaxing over the **time axis**; that version failed `tests/test_causality.py` because each token could see its own future summary. The current `student_final` removes that mechanism entirely and relies solely on causal pre-norm attention + SwiGLU + QK-norm + GQA + per-head temperature + per-token gating — every component is strictly causal.

---

## Ablation suite (for the report)

The `code/runs/` directory contains smaller ablation runs that share the same block template but disable individual components:

| Run | Variant | Purpose |
|---|---|---|
| `student_my_no_gm` | identical stack, control run without memory tokens | Confirms the final arch doesn't need the removed GlobalMemoryTokens to fit its BPB. |
| `student_my_no_gate_temp` | gates and temperature disabled | Isolates their contribution to BPB and stability. |
| `student_my_casual` | historical causal fix of `student_my` | Kept only for reference; not the final submission. |

These are referenced from `report/report.md` figures; only `student_final` is graded.

---

## Quick Start

```bash
# 1. Install dependencies
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126  # GPU
python -m pip install -r requirements.txt

# 2. Train
python train02.py \
  --implementation student_final \
  --config configs/student_final.json \
  --device cuda --precision bf16 \
  --threads 4 --seed 17 \
  --steps 16000 --batch-size 32 \
  --eval-every 400 --lr 0.003 --warmup 2400 \
  --ema-decay 0.999 \
  --run-dir runs/student_final

# 3. Evaluate on test set
python evaluate.py \
  --checkpoint runs/student_final/checkpoint_best.pt \
  --split test
```

See [`code/student_work/命令指导.md`](code/student_work/命令指导.md) for full command reference and file naming conventions.

---

## Repository Structure

```
MP1_student_starter/
├── GUIDE_zh.md                  # Assignment guide (Chinese)
├── README.md                    # This file
├── code/
│   ├── student_work/
│   │   ├── student_final.py     # Final submission (causal; see Method)
│   │   ├── student_my_casual.py # Historical causal fix of an earlier draft (reference only)
│   │   ├── student_my.py        # Earlier non-causal draft (do NOT use)
│   │   ├── student_my_no_gm.py   # Ablation control: identical arch, separate run
│   │   ├── student_my_no_gate_temp.py  # Ablation: gates and per-head temperature disabled
│   │   └── 命令指导.md           # Command reference (Chinese)
│   ├── configs/
│   │   ├── student_final.json   # Hyperparameters for student_final
│   │   └── ...                  # Ablation configs
│   ├── train02.py               # Training script
│   └── evaluate.py              # Evaluation script
└── runs/student_final/          # Training outputs (not committed)
```

---

## AI Assistance Disclosure

This project uses **Cursor AI** as a coding assistant during implementation and debugging. All algorithmic decisions, experimental design, and analysis were conceived and verified by the author. AI assistance was limited to code generation, error resolution, and documentation formatting. This submission represents the author's own work.

---

## Checkpoint Bundle

The complete checkpoint package (including `checkpoint_best.pt`, `checkpoint.pt`, and `metrics.json`) is available as a GitHub Release. See the Releases page.