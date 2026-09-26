"""Generate all plots for the report.

Reads metrics.json from the runs directory and produces PNGs in report/figures/.
All data lives in this script so the figures are 100% reproducible from the
captured metrics.
"""
from __future__ import annotations
import json
import os
import sys
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


def test_bpb(name: str) -> float | None:
    p = RUNS / name / "test_cpu_fp32.json"
    if p.exists():
        return json.load(open(p))["bpb"]
    return None


def best_val_bpb(name: str) -> float:
    d = load(name)
    return best_validation(d)[1]


# ----------------------------------------------------------------------------
# Figure 1: validation BPB vs training step for all models
# ----------------------------------------------------------------------------
def fig_training_curves() -> None:
    runs = {
        "Baseline (1.2k steps)": ("baseline-test", "#9CA3AF", ":"),
        "student_my (24k steps)": ("student_my", "#2563EB", "-"),
        "no GlobalMemoryTokens": ("student_my_no_gm", "#EF4444", "-."),
        "no gate + temperature": ("student_my_no_gate_temp", "#F59E0B", "--"),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, (run, color, ls) in runs.items():
        d = load(run)
        vh = d["validation_history"]
        steps = [e["step"] for e in vh]
        bpb = [e["bpb"] for e in vh]
        ax.plot(steps, bpb, label=label, color=color, linestyle=ls, linewidth=1.8,
                marker="o", markersize=4, markevery=5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation BPB")
    ax.set_title("Validation BPB vs. Training Step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_training_curves.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: best validation BPB vs params scatter (Pareto front)
# ----------------------------------------------------------------------------
def fig_pareto() -> None:
    pts = []
    for name in ["baseline-test", "student_my_no_gm",
                 "student_my_no_gate_temp", "student_my_03",
                 "student_my_back", "student_my"]:
        d = load(name)
        bpb = test_bpb(name)
        if bpb is None:
            continue
        pts.append((d["parameters"], bpb, name))

    pts.sort()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for params, bpb, name in pts:
        label = {
            "baseline-test": "Baseline",
            "student_my_no_gm": "no GlobalMemory",
            "student_my_no_gate_temp": "no gate + temp",
            "student_my_03": "alt run #1",
            "student_my_back": "alt run #2",
            "student_my": "student_my (final)",
        }[name]
        color = "#2563EB" if name == "student_my" else "#9CA3AF"
        size = 110 if name == "student_my" else 70
        ax.scatter(params, bpb, s=size, c=color, edgecolors="black",
                   linewidths=0.8, zorder=3)
        ax.annotate(label, (params, bpb),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Test BPB")
    ax.set_title("Test BPB vs. Parameter Count")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_pareto.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: ablation bar chart
# ----------------------------------------------------------------------------
def fig_ablation() -> None:
    rows = []
    for name, label in [
        ("student_my", "Full (student_my)"),
        ("student_my_no_gm", "− GlobalMemoryTokens"),
        ("student_my_no_gate_temp", "− gate + temperature"),
    ]:
        d = load(name)
        bp = best_validation(d)
        tb = test_bpb(name) if test_bpb(name) is not None else bp[1]
        rows.append((label, d["parameters"], bp[1], tb))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    names = [r[0] for r in rows]
    params = [r[1] / 1e6 for r in rows]
    test_bpbs = [r[3] for r in rows]
    val_bpbs = [r[2] for r in rows]

    colors = ["#2563EB", "#EF4444", "#F59E0B"]
    axes[0].bar(names, params, color=colors, edgecolor="black")
    axes[0].set_ylabel("Parameters (M)")
    axes[0].set_title("Parameter Count")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(params):
        axes[0].text(i, v + 0.15, f"{v:.2f}M",
                     ha="center", fontsize=9)

    x = np.arange(len(names))
    w = 0.36
    axes[1].bar(x - w / 2, val_bpbs, w, color=colors, edgecolor="black",
                label="Best validation")
    axes[1].bar(x + w / 2, test_bpbs, w, color=colors, edgecolor="black",
                alpha=0.55, label="Test")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("BPB (lower is better)")
    axes[1].set_title("Ablation: BPB")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=9)
    for i, (v, t) in enumerate(zip(val_bpbs, test_bpbs)):
        axes[1].text(i - w / 2, v + 0.02, f"{v:.3f}",
                     ha="center", fontsize=8)
        axes[1].text(i + w / 2, t + 0.02, f"{t:.3f}",
                     ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGS / "fig3_ablation.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: training/validation loss for student_my only (smoothed)
# ----------------------------------------------------------------------------
def fig_loss_curves() -> None:
    d = load("student_my")
    history = d.get("history", [])
    vh = d["validation_history"]

    train_steps = [h["step"] for h in history if h.get("loss") is not None]
    train_loss = [h["loss"] for h in history if h.get("loss") is not None]
    val_steps = [v["step"] for v in vh]
    val_bpb = [v["bpb"] for v in vh]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Smoothed train loss
    window = 50
    if len(train_loss) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(train_loss, kernel, mode="valid")
        ax.plot(train_steps[window - 1:], smoothed,
                color="#60A5FA", linewidth=1.2, alpha=0.9,
                label="Train loss (smoothed, window=50)")

    ax.plot(val_steps, val_bpb, color="#2563EB", linewidth=2.0,
            marker="o", markersize=4, markevery=4,
            label="Validation BPB")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss / BPB")
    ax.set_title("student_my: Training Loss and Validation BPB")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_loss_curves.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: GlobalMemoryTokens architecture (diagram)
# ----------------------------------------------------------------------------
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Token input row
    for i in range(8):
        ax.add_patch(plt.Rectangle((0.6 + i * 1.0, 0.4), 0.8, 0.5,
                                   facecolor="#DBEAFE", edgecolor="#1E3A8A"))
    ax.text(0.2, 0.65, "Input\ntokens", ha="right", va="center", fontsize=9)
    ax.text(4.7, 0.15, "x ∈ [B, T=256], vocab=2048",
            ha="center", fontsize=8, style="italic")

    # Global memory tokens row
    for i in range(4):
        ax.add_patch(plt.Rectangle((0.6 + i * 1.0, 1.4), 0.8, 0.5,
                                   facecolor="#FDE68A", edgecolor="#92400E"))
    ax.text(0.2, 1.65, "Global\nmemory\ntokens",
            ha="right", va="center", fontsize=9)
    ax.text(4.7, 1.15, "n=12 learnable vectors (shared across examples)",
            ha="center", fontsize=8, style="italic")

    # Stack and cross attention
    ax.annotate("", xy=(5.0, 3.0), xytext=(5.0, 1.95),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    ax.text(5.2, 2.4, "concat", fontsize=8, color="#374151")

    ax.add_patch(plt.Rectangle((3.5, 3.0), 3.0, 0.7,
                               facecolor="#A7F3D0", edgecolor="#065F46"))
    ax.text(5.0, 3.35, "Cross-attention\n(memory ← sequence)",
            ha="center", va="center", fontsize=9)

    ax.annotate("", xy=(5.0, 4.4), xytext=(5.0, 3.75),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))

    ax.add_patch(plt.Rectangle((3.7, 4.4), 2.6, 0.6,
                               facecolor="#FCA5A5", edgecolor="#7F1D1D"))
    ax.text(5.0, 4.7, "Gated fuse → residual stream",
            ha="center", va="center", fontsize=9)

    # Block stack on the right
    for i in range(4):
        ax.add_patch(plt.Rectangle((7.4, 0.6 + i * 1.2), 2.2, 1.0,
                                   facecolor="#E0E7FF", edgecolor="#3730A3"))
        ax.text(8.5, 1.1 + i * 1.2,
                f"Block {i + 1}\nGQA+SwiGLU\n+T+Gate",
                ha="center", va="center", fontsize=8)

    # Down arrow from fused back to block stack
    ax.annotate("", xy=(7.4, 2.1), xytext=(6.3, 4.7),
                arrowprops=dict(arrowstyle="->", color="#374151",
                                lw=1.4, connectionstyle="arc3,rad=-0.2"))
    ax.text(7.0, 3.6, "fused ctx", fontsize=8, color="#374151")

    ax.text(5.0, 5.5,
            "GlobalMemoryTokens architecture (one block shown)",
            ha="center", fontsize=11, weight="bold")

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_architecture.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 6: CPU time + checkpoint size comparison
# ----------------------------------------------------------------------------
def fig_resources() -> None:
    runs = [
        ("Baseline", 5.92, 4.0),
        ("student_my", 21.90, 36.0),
    ]
    names = [r[0] for r in runs]
    cpu_times = [r[1] for r in runs]
    sizes = [r[2] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

    colors = ["#9CA3AF", "#2563EB"]
    axes[0].bar(names, cpu_times, color=colors, edgecolor="black")
    axes[0].axhline(5.92 * 5, color="#DC2626", linestyle="--",
                    label="limit (29.60 s)")
    axes[0].set_ylabel("CPU evaluation time (s)")
    axes[0].set_title("CPU time")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(cpu_times):
        axes[0].text(i, v + 0.6, f"{v:.2f}s", ha="center", fontsize=9)

    axes[1].bar(names, sizes, color=colors, edgecolor="black")
    axes[1].axhline(64.0, color="#DC2626", linestyle="--", label="limit (64 MiB)")
    axes[1].set_ylabel("Checkpoint size (MiB)")
    axes[1].set_title("Inference assets")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(sizes):
        axes[1].text(i, v + 1, f"{v:.0f} MiB", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS / "fig6_resources.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_training_curves()
    fig_pareto()
    fig_ablation()
    fig_loss_curves()
    fig_architecture()
    fig_resources()
    print("Figures written to", FIGS)
    for p in sorted(FIGS.glob("*.png")):
        print("  ", p.name, p.stat().st_size, "bytes")
