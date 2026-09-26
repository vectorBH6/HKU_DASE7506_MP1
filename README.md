# MP1 — Small Language Model Challenge

**Student ID:** 3036705979  
**Name:** Li Shanghang  
**Course:** DASE7506

---

## Results

| Model | Test BPB | CPU Time | Checkpoint | Under Limit? |
|-------|----------|----------|------------|--------------|
| Baseline | 2.10 | 5.92 s | — | — |
| **student_my** | **1.337** | **21.90 s** | **36 MiB** | **✓ All pass** |

**All three resource limits satisfied:**

| Constraint | student_my | Limit |
|------------|-----------|-------|
| CPU time | 21.90 s | ≤ 29.60 s (5× baseline) |
| Peak RAM | ~0.05 GiB (est.) | ≤ 4 GiB |
| Inference assets | 36 MiB | ≤ 64 MiB |

BPB improved **-36.3%** vs baseline. Training time ~805 s on GPU (BF16).

---

## Method

`student_my` builds on the baseline GPT with the following architectural improvements:

1. **RMSNorm + SwiGLU blocks** — Pre-norm residual structure with Swish-Gated Linear Unit replacing the original MLP, paired with grouped-query attention (GQA), QK normalization, and partial rotary positional embedding (RoPE).
2. **Per-token gating** — A lightweight linear gate modulates each attention block's output, allowing selective information flow.
3. **Learnable attention temperature** — A per-head log-temperature parameter adaptively scales query/key dot products before softmax.
4. **GlobalMemoryTokens** — A small set of learnable memory tokens aggregates global context via cross-attention over the sequence and injects a gated summary back into the residual stream. They live outside the input sequence and therefore preserve the original model contract (`ids` shape stays `[B, T]`).

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
  --implementation student_my \
  --config configs/student_my.json \
  --device cuda --precision bf16 \
  --threads 4 --seed 17 \
  --steps 24000 --batch-size 32 \
  --eval-every 400 --lr 0.003 --warmup 2400 \
  --ema-decay 0.999 \
  --run-dir runs/student_my

# 3. Evaluate on test set
python evaluate.py \
  --checkpoint runs/student_my/checkpoint_best.pt \
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
│   │   ├── student_my.py        # Final submission implementation
│   │   ├── student_my_no_gm.py   # Ablation: no GlobalMemoryTokens
│   │   ├── student_my_no_gate_temp.py  # Ablation: no gate + temperature
│   │   └── 命令指导.md           # Command reference (Chinese)
│   ├── configs/
│   │   ├── student_my.json      # Hyperparameters for student_my
│   │   └── ...                  # Ablation configs
│   ├── train02.py               # Training script
│   └── evaluate.py              # Evaluation script
└── runs/student_my/             # Training outputs (not committed)
```

---

## AI Assistance Disclosure

This project uses **Cursor AI** as a coding assistant during implementation and debugging. All algorithmic decisions, experimental design, and analysis were conceived and verified by the author. AI assistance was limited to code generation, error resolution, and documentation formatting. This submission represents the author's own work.

---

## Checkpoint Bundle

The complete checkpoint package (including `checkpoint_best.pt`, `checkpoint.pt`, and `metrics.json`) is available as a GitHub Release. See the Releases page.
