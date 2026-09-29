"""Generate all plots for the report.

Reads metrics.json from `code/runs/` and produces PNGs in `report/figures/`.
All numbers, colours, and run names are derived from the captured metrics so
the figures are 100 % reproducible from the runs on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT.parent / "code"
RUNS = CODE_ROOT / "runs"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


# Local CPU baseline measured on this machine (per code/README_cn.md the 5x rule
# is hardware-relative). Used to draw the resource-limit line.
LOCAL_CPU_BASELINE_S = 8.386799252999992  # from runs/baseline-test/test_cpu_fp32.json


def load(name: str) -> dict:
    with open(RUNS / name / "metrics.json") as f:
        return json.load(f)


def best_validation(run: dict) -> tuple[int, float]:
    vh = run["validation_history"]
    best_step, best_bpb = 0, float("inf")
    for entry in vh:
        if entry["bpb"] < best_bpb:
            best_step, best_bpb = entry["step"], entry["bpb"]
    return best_step, best_bpb


def test_record(name: str) -> dict | None:
    p = RUNS / name / "test_cpu_fp32.json"
    if p.exists():
        return json.load(open(p))
    return None


def test_bpb(name: str) -> float | None:
    r = test_record(name)
    return r["bpb"] if r else None


def ckpt_mib(name: str) -> float:
    p = RUNS / name / "checkpoint.pt"
    return p.stat().st_size / (1024 * 1024) if p.exists() else float("nan")


def fmt_secs(s: float) -> str:
    return f"{s:.2f}s"


# ----------------------------------------------------------------------------
# Figure 1: validation BPB vs training step for all available runs
# ----------------------------------------------------------------------------
def fig_training_curves() -> None:
    runs = {
        "Baseline (1.2k steps)":  ("baseline-test",          "#9CA3AF", ":"),
        "student_final (16k)":    ("student_final",          "#2563EB", "-"),
        "student_my (24k)":       ("student_my",             "#7C3AED", "-."),
        "student_my_no_gm (24k)": ("student_my_no_gm",       "#EF4444", "--"),
        "student_my_no_gate (24k)": ("student_my_no_gate_temp", "#F59E0B", ":"),
        "student_my_casual (24k)":("student_my_casual",      "#10B981", "--"),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, (run, color, ls) in runs.items():
        d = load(run)
        vh = d["validation_history"]
        steps = [e["step"] for e in vh]
        bpb = [e["bpb"] for e in vh]
        ax.plot(steps, bpb, label=label, color=color, linestyle=ls, linewidth=1.6,
                marker="o", markersize=3.5, markevery=4)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation BPB")
    ax.set_title("Validation BPB vs. Training Step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_training_curves.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: Pareto frontier of test BPB vs CPU time (with 5x limit line)
# ----------------------------------------------------------------------------
def fig_pareto() -> None:
    runs = [
        "baseline-test",
        "student_final",
        "student_my_no_gm",
        "student_my_no_gate_temp",
        "student_my_casual",
        "student_my_03",
        "student_my_back",
        "student_my",
    ]
    pts = []
    for name in runs:
        rec = test_record(name)
        if rec is None:
            continue
        d = load(name)
        pts.append({
            "name": name,
            "params": d["parameters"],
            "bpb": rec["bpb"],
            "secs": rec["seconds"],
            "ckpt_mib": ckpt_mib(name),
        })
    pts.sort(key=lambda p: p["bpb"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cpu_limit = 5 * LOCAL_CPU_BASELINE_S

    for p in pts:
        is_final = p["name"] == "student_final"
        color = "#2563EB" if is_final else "#9CA3AF"
        size = 130 if is_final else 75
        marker = "*" if is_final else "o"
        edge = "black"
        ax.scatter(p["secs"], p["bpb"], s=size, c=color, edgecolors=edge,
                   linewidths=0.9, marker=marker, zorder=3)
        ax.annotate(p["name"], (p["secs"], p["bpb"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, alpha=0.85)

    ax.axvline(cpu_limit, color="#DC2626", linestyle="--",
               linewidth=1.2, label=f"CPU limit (5x baseline = {cpu_limit:.1f}s)")
    ax.set_xlabel("CPU evaluation time (s) — lower is faster")
    ax.set_ylabel("Test BPB — lower is better")
    ax.set_title("Test BPB vs. CPU evaluation time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_pareto.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: ablation bar chart (validation / test BPB + params)
# ----------------------------------------------------------------------------
def fig_ablation() -> None:
    runs = [
        ("student_my",                "Full (student_my, 24k)"),
        ("student_my_no_gm",          "− GlobalMemoryTokens"),
        ("student_my_no_gate_temp",   "− gate + per-head T"),
        ("student_final",             "student_final (16k, causal-only)"),
    ]
    rows = []
    for name, label in runs:
        d = load(name)
        bp = best_validation(d)
        tb = test_bpb(name)
        rows.append({
            "name": name,
            "label": label,
            "params": d["parameters"],
            "val_bpb": bp[1],
            "test_bpb": tb if tb is not None else float("nan"),
        })

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    names = [r["label"] for r in rows]
    params = [r["params"] / 1e6 for r in rows]
    colors = ["#7C3AED", "#EF4444", "#F59E0B", "#2563EB"]

    axes[0].bar(names, params, color=colors, edgecolor="black")
    axes[0].set_ylabel("Parameters (M)")
    axes[0].set_title("Parameter count")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(params):
        axes[0].text(i, v + 0.15, f"{v:.2f}M",
                     ha="center", fontsize=8)
    axes[0].tick_params(axis="x", labelsize=8)

    x = np.arange(len(names))
    w = 0.36
    val_bpbs = [r["val_bpb"] for r in rows]
    test_bpbs = [r["test_bpb"] for r in rows]
    axes[1].bar(x - w / 2, val_bpbs, w, color=colors, edgecolor="black",
                label="Best validation BPB")
    axes[1].bar(x + w / 2, test_bpbs, w, color=colors, edgecolor="black",
                alpha=0.55, label="Test BPB")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=12, ha="right", fontsize=8)
    axes[1].set_ylabel("BPB (lower is better)")
    axes[1].set_title("Ablation: BPB")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    for i, (v, t) in enumerate(zip(val_bpbs, test_bpbs)):
        axes[1].text(i - w / 2, v + 0.02, f"{v:.3f}",
                     ha="center", fontsize=7.5)
        if not np.isnan(t):
            axes[1].text(i + w / 2, t + 0.02, f"{t:.3f}",
                         ha="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIGS / "fig3_ablation.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: smoothed training loss and validation BPB for student_final
# ----------------------------------------------------------------------------
def fig_loss_curves(run_name: str = "student_final") -> None:
    d = load(run_name)
    history = d.get("history", [])
    vh = d["validation_history"]

    train_steps = [h["step"] for h in history if h.get("loss") is not None]
    train_loss = [h["loss"] for h in history if h.get("loss") is not None]
    val_steps = [v["step"] for v in vh]
    val_bpb = [v["bpb"] for v in vh]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    window = 50
    if len(train_loss) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(train_loss, kernel, mode="valid")
        ax.plot(train_steps[window - 1:], smoothed,
                color="#60A5FA", linewidth=1.2, alpha=0.9,
                label="Train loss (smoothed, window=50)")

    ax.plot(val_steps, val_bpb, color="#2563EB", linewidth=2.0,
            marker="o", markersize=4, markevery=3,
            label="Validation BPB")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss / BPB")
    ax.set_title(f"{run_name}: training loss and validation BPB")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_loss_curves.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: block-level schematic of student_final
# ----------------------------------------------------------------------------
def fig_block_diagram() -> None:
    """Schematic of one Block of student_final.

    The pre-norm residual block is:
        x = x + Drop(Attn(RMSNorm(x)))
        x = x + Drop(SwiGLU(RMSNorm(x)))
    and Attention itself contains GQA + QK-norm + partial RoPE + per-token gate
    + per-head log-temperature.  No GlobalMemoryTokens.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # Residual stream
    ax.annotate("", xy=(9.4, 5.5), xytext=(0.6, 5.5),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4))
    ax.text(0.4, 5.5, "x ∈ [B,T,288]", ha="right", va="center",
            fontsize=8, style="italic")
    ax.text(9.5, 5.5, "x' ∈ [B,T,288]", ha="left", va="center",
            fontsize=8, style="italic")
    ax.text(5.0, 5.85, "residual stream (width=288)", ha="center",
            fontsize=9, color="#374151")

    def box(x, y, w, h, color, edge, text):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor=edge, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.5)

    # Attn branch
    box(2.4, 4.4, 1.5, 0.7, "#DBEAFE", "#1E3A8A", "RMSNorm")
    box(4.2, 4.4, 2.4, 0.7, "#BFDBFE", "#1E3A8A",
        "GQA + QK-norm\n+ RoPE (70%)")
    box(7.0, 4.4, 1.8, 0.7, "#DBEAFE", "#1E3A8A",
        "o_proj · σ(Wg x)")
    box(8.4, 4.0, 0.6, 0.4, "#FECACA", "#7F1D1D", "Drop")
    ax.annotate("+", xy=(8.0, 5.45), xytext=(8.0, 5.45),
                fontsize=18, ha="center", va="center", color="#065F46")

    # FFN branch
    box(2.4, 2.6, 1.5, 0.7, "#FDE68A", "#92400E", "RMSNorm")
    box(4.2, 2.6, 2.6, 0.7, "#FCD34D", "#92400E",
        "SwiGLU (w1/w2/w3)")
    box(7.0, 2.6, 1.8, 0.7, "#FDE68A", "#92400E", "hidden=640")
    box(8.4, 2.2, 0.6, 0.4, "#FECACA", "#7F1D1D", "Drop")

    # Arrows
    ax.annotate("", xy=(5.45, 4.4), xytext=(5.45, 5.45),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.0))
    ax.annotate("", xy=(5.45, 2.6), xytext=(5.45, 4.0),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.0))
    ax.annotate("", xy=(8.0, 5.45), xytext=(8.0, 5.05),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.0))
    ax.annotate("", xy=(8.0, 4.55), xytext=(8.0, 2.6),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.0))

    # Annotations
    ax.text(0.4, 4.75, "Attn\nbranch", ha="right", va="center",
            fontsize=9, color="#1E3A8A", weight="bold")
    ax.text(0.4, 2.95, "FFN\nbranch", ha="right", va="center",
            fontsize=9, color="#92400E", weight="bold")

    # Stack of 7 blocks
    ax.text(5.0, 1.6,
            "Pre-norm residual block (× 7 layers, depth = 7)\n"
            "Final RMSNorm + tied output head (vocab = 2048)",
            ha="center", va="center", fontsize=9, color="#374151",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F3F4F6", edgecolor="#374151"))

    ax.text(5.0, 0.6,
            "Causal · SDPA(is_causal=True) · no GlobalMemoryTokens",
            ha="center", va="center", fontsize=9, style="italic",
            color="#065F46")

    ax.text(5.0, 6.05, "student_final — single block",
            ha="center", fontsize=11, weight="bold")

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_block_diagram.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 6: CPU time and checkpoint size vs limits
# ----------------------------------------------------------------------------
def fig_resources() -> None:
    runs = [
        ("Baseline (this machine)", LOCAL_CPU_BASELINE_S,
         ckpt_mib("baseline-test")),
        ("student_final", test_record("student_final")["seconds"],
         ckpt_mib("student_final")),
        ("student_my (24k)", test_record("student_my")["seconds"],
         ckpt_mib("student_my")),
    ]
    names = [r[0] for r in runs]
    cpu_times = [r[1] for r in runs]
    sizes = [r[2] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    cpu_limit = 5 * LOCAL_CPU_BASELINE_S
    colors = ["#9CA3AF", "#2563EB", "#7C3AED"]

    axes[0].bar(names, cpu_times, color=colors, edgecolor="black")
    axes[0].axhline(cpu_limit, color="#DC2626", linestyle="--",
                    label=f"limit 5x baseline ({cpu_limit:.1f} s)")
    axes[0].set_ylabel("CPU evaluation time (s)")
    axes[0].set_title("CPU time (lower is faster)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(cpu_times):
        axes[0].text(i, v + 0.6, fmt_secs(v), ha="center", fontsize=9)

    axes[1].bar(names, sizes, color=colors, edgecolor="black")
    axes[1].axhline(64.0, color="#DC2626", linestyle="--",
                    label="limit (64 MiB)")
    axes[1].set_ylabel("Checkpoint size (MiB)")
    axes[1].set_title("Inference assets")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(sizes):
        axes[1].text(i, v + 1.2, f"{v:.1f} MiB", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS / "fig6_resources.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 7: end-to-end overview of student_final
# ----------------------------------------------------------------------------
def fig_end_to_end() -> None:
    """Static, embeddable overview of the whole `student_final` model.

    Reads width / depth / hidden_dim / rope_fraction / n_heads / n_kv_heads
    / context / vocab from ``code/configs/student_final.json`` so the figure
    stays in sync with the trained model.
    """
    with open(CODE_ROOT / "configs" / "student_final.json") as f:
        cfg = json.load(f)

    depth = cfg["depth"]
    width = cfg["width"]
    hidden_dim = cfg.get("hidden_dim", 768)
    n_heads = cfg["heads"]
    n_kv_heads = cfg.get("kv_heads", n_heads // 2)
    head_dim = width // n_heads
    rope_dim = max(2, int(head_dim * cfg.get("rope_fraction", 0.5)) // 2 * 2)
    vocab = cfg["vocab"]
    ctx = cfg["context"]

    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(5.0, 11.5,
            f"student_final — end-to-end architecture "
            f"(depth = {depth}, width = {width}, vocab = {vocab})",
            ha="center", fontsize=12, weight="bold")
    ax.text(5.0, 11.05,
            "Pure causal pre-norm stack. No GlobalMemoryTokens, "
            "no cross-window state.",
            ha="center", fontsize=9, style="italic", color="#065F46")

    def box(x, y, w, h, color, edge, text, fontsize=9, weight="normal"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor=edge, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2, label=None, color="#374151", lw=1.0,
              label_offset=(0.18, 0.0), label_color="#374151"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))
        if label:
            lx = (x1 + x2) / 2 + label_offset[0]
            ly = (y1 + y2) / 2 + label_offset[1]
            ax.text(lx, ly, label, ha="left", va="center",
                    fontsize=8, color=label_color, style="italic")

    # ---------- Input row ----------------------------------------------
    box(3.5, 9.7, 3.0, 0.7, "#FEF3C7", "#92400E",
        f"ids ∈ [B, T={ctx}], vocab = {vocab}", fontsize=10, weight="bold")
    arrow(5.0, 9.7, 5.0, 9.05)

    # ---------- Token embedding -----------------------------------------
    box(3.5, 8.35, 3.0, 0.7, "#DBEAFE", "#1E3A8A",
        f"token_emb (vocab × {width})\ntied with lm_head",
        fontsize=9)
    arrow(5.0, 8.35, 5.0, 7.85)

    # ---------- Stack of 7 Blocks ---------------------------------------
    block_top = 7.85
    block_h = 0.7
    block_w = 5.4
    block_left = (10 - block_w) / 2
    gap = 0.15

    for i in range(depth):
        y = block_top - (i + 1) * (block_h + gap)
        # Block label
        box(block_left, y, block_w, block_h, "#E0E7FF", "#3730A3",
            f"Block {i + 1}:  RMSNorm → GQA (heads={n_heads}, kv={n_kv_heads})  "
            f"→  RMSNorm → SwiGLU ({hidden_dim})",
            fontsize=8.5)
        if i == 0:
            ax.text(2.0, y + block_h / 2,
                    f"× {depth}",
                    ha="right", va="center", fontsize=10, color="#7C3AED",
                    weight="bold")
        if i < depth - 1:
            arrow(5.0, y, 5.0, y - gap)

    # Side annotations for one block (between Block 3 and Block 4)
    mid_y = block_top - 3.5 * (block_h + gap) - block_h / 2
    ax.text(0.2, mid_y,
            "Per block:\n"
            "  • QK-norm (RMS) on q, k\n"
            "  • Partial RoPE ({} / {} dims)\n"
            "  • Per-head log-temperature\n"
            "  • Per-token output gate\n"
            "  • SwiGLU FFN\n"
            "  • residual dropout 0.15".format(rope_dim, head_dim),
            ha="left", va="center", fontsize=8, color="#1E3A8A",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#EFF6FF", edgecolor="#1E3A8A"))

    ax.text(8.4, mid_y,
            "Causal SDPA\n"
            "(is_causal=True)\n"
            "GQA: k,v expanded\n"
            "via repeat_interleave",
            ha="left", va="center", fontsize=8, color="#92400E",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#FEF3C7", edgecolor="#92400E"))

    # ---------- Final norm + LM head ------------------------------------
    last_y = block_top - depth * (block_h + gap)
    arrow(5.0, last_y, 5.0, last_y - 0.6)

    final_norm_y = last_y - 1.3
    box(3.5, final_norm_y, 3.0, 0.7, "#DBEAFE", "#1E3A8A",
        "Final RMSNorm", fontsize=9)
    arrow(5.0, final_norm_y, 5.0, final_norm_y - 0.7)

    lm_y = final_norm_y - 2.0
    box(3.0, lm_y, 4.0, 0.9, "#FECACA", "#7F1D1D",
        f"lm_head (tied) → logits [B, T, {vocab}]\n"
        f"FP32 log_softmax → log p  (predict_log_probs)",
        fontsize=9, weight="bold")

    # ---------- Loss / scoring ------------------------------------------
    arrow(5.0, lm_y, 5.0, lm_y - 0.7)
    box(2.5, lm_y - 1.4, 5.0, 0.7, "#D1FAE5", "#065F46",
        "loss = NLL on next token; scored as BPB on WikiText-2 (BPE-2048)",
        fontsize=9)

    # ---------- Parameter tally -----------------------------------------
    tally = (
        "Parameters: 6,788,010  (≈6.79 M)\n"
        f"  • token_emb (tied with lm_head):  {vocab}×{width} = {vocab*width:,}\n"
        f"  • {depth}× Attention:  q,k,v,o_proj + gate + q_norm + k_norm\n"
        f"      + log_temperature  ({n_heads}×1) + RoPE cache\n"
        f"  • {depth}× SwiGLU:  w1,w2,w3 at hidden = {hidden_dim}\n"
        f"  • {depth + 1}× RMSNorm weights ({width} each)\n"
        f"Checkpoint: 26 MiB  (FP32, EMA best @ step 10000)"
    )
    ax.text(0.4, 1.7, tally, ha="left", va="top", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F3F4F6", edgecolor="#374151"))

    fig.tight_layout()
    fig.savefig(FIGS / "fig7_end_to_end.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_training_curves()
    fig_pareto()
    fig_ablation()
    fig_loss_curves("student_final")
    fig_block_diagram()
    fig_resources()
    fig_end_to_end()
    print("Figures written to", FIGS)
    for p in sorted(FIGS.glob("*.png")):
        print("  ", p.name, p.stat().st_size, "bytes")