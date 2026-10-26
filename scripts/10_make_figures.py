"""Step 10: the paper's tables and figures, and reproduced results next to them.

Always (no data or GPU needed), from the numbers in src/ctxdiar/paper.py:
  results/paper/table2..table9 (.csv + .md), relative_reductions (.csv + .md)
  results/figures/fig5_multiscale_windows.png      the five scales of Table 2 on a 6 s excerpt
  results/figures/fig9_eval_der_comparison.png     Table 4, DISPLACE 2024 evaluation rows only
  results/figures/fig10_ablation.png               Table 5, dev and eval
  results/figures/fig11_conditions.png             Tables 6 and 7
  results/figures/fig12_parameter_budget.png       Table 8

When reproduced results exist in results/reproduced/ (written by steps 6-9):
  results/figures/fig_ablation_paper_vs_reproduced.png   the rows this code can produce
  results/reproduced/table5_reproduced.md
and with --weights_npz exp/<run>/eval/weights/<session>.npz:
  results/figures/fig_measured_scale_weights.png  measured w_s(t), the counterpart of the
                                                  schematic Figure 7

Reproduced numbers are only ever read from files a run wrote. Nothing is
filled in by hand.

Usage:  python scripts/10_make_figures.py [--weights_npz ...]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _common  # noqa: E402
from ctxdiar import paper  # noqa: E402
from ctxdiar.scales import SHIFTS, WINDOWS, subsegments  # noqa: E402

ROOT = _common.ROOT
PAPER, FIGS, REPRO = ROOT / "results/paper", ROOT / "results/figures", ROOT / "results/reproduced"

# Colours: one blue for the proposed system / evaluation, one orange for
# development, neutral greys for everything else, so the eye goes to the
# comparison the figure is about.
BLUE, ORANGE, GREY, DARK, MUTED = "#2a78d6", "#eb6834", "#a3a29d", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": DARK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 300, "savefig.bbox": "tight"})


def fmt(v):
    return "---" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def write_table(name, header, rows, title, note=""):
    with open(PAPER / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([["" if v is None else v for v in r] for r in rows])
    md = [f"# {title}", "", "Copied from the paper. Not a reproduced result.", "",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    if note:
        md += ["", note]
    (PAPER / f"{name}.md").write_text("\n".join(md) + "\n")


def paper_tables():
    PAPER.mkdir(parents=True, exist_ok=True)
    write_table("table2_hyperparameters", ["Parameter", "Value"], paper.TABLE2, "Table 2. Hyperparameter configuration")
    write_table("table3_corpus", ["Property", "Value"], paper.TABLE3, "Table 3. Properties of the DISPLACE 2024 corpus",
                "Measure these on your copy with scripts/01_prepare_data.py (results/reproduced/table3_corpus_measured.md).")
    write_table("table4_der_comparison", ["System", "Dev (%)", "Eval (%)", "Note in the paper"], paper.TABLE4,
                "Table 4. DER (%) on DISPLACE, zero collar",
                "Rows marked 'DISPLACE 2023 data' are not comparable with the 2024 rows. The proposed system's dev "
                "DER is training-set error (the decoder is trained on the development partition).")
    write_table("table5_ablation", ["Configuration", "Dev (%)", "Eval (%)", "Delta eval vs previous row (pp)"],
                paper.TABLE5, "Table 5. Cumulative component-wise ablation", paper.TABLE5_NOTES)
    write_table("table6_snr", ["SNR condition", "DER (%)", "Sessions"], paper.TABLE6, "Table 6. Evaluation DER by SNR")
    write_table("table7_overlap", ["Condition", "DER (%)", "Sessions"], paper.TABLE7,
                "Table 7. Evaluation DER by overlap density and utterance length")
    write_table("table8_profile", ["Module", "Parameters", "Updated in training", "Cost in T", "Notes"], paper.TABLE8,
                "Table 8. Computational profile")
    (PAPER / "table9_significance.md").write_text(
        "# Table 9. Paired session-level test, static vs dynamic weighting\n\n"
        f"**Status: {paper.TABLE9_STATUS}.**\n\n"
        "The paper describes the procedure (Algorithm 2) but reports no numbers. They must be computed from the\n"
        "per-session outputs of both systems with `scripts/07_significance.py`, and land in\n"
        "`results/reproduced/table9_significance.md`. They are never typed in by hand.\n")

    rows = []
    for label, (ref, quoted) in paper.REPORTED_RELATIVE.items():
        sys_der = paper.PROPOSED_EVAL
        rows.append([label, ref, sys_der, round(paper.relative_reduction(ref, sys_der), 2), quoted])
    with open(PAPER / "relative_reductions.csv", "w", newline="") as f:
        csv.writer(f).writerows([["comparison", "reference DER", "system DER", "recomputed (%)", "quoted in paper (%)"]] + rows)
    md = ["# Relative reductions, Eq. (12), recomputed", "",
          "| Comparison | Reference | System | Recomputed (%) | Quoted (%) |", "|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} |" for r in rows]
    t6, t7 = paper.session_weighted_mean(paper.TABLE6), paper.session_weighted_mean(paper.TABLE7)
    md += ["", f"Session-weighted means (paragraph under Table 7): Table 6 {t6:.2f} % (quoted 23.27), "
               f"Table 7 {t7:.2f} % (quoted 22.72), both above the duration-weighted 21.46 %."]
    (PAPER / "relative_reductions.md").write_text("\n".join(md) + "\n")


def fig5_windows():
    """Every window drawn as its own bar, alternating between two lanes per
    scale, so the 50 % overlap between neighbours is visible."""
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for s, (w, sh) in enumerate(zip(WINDOWS, SHIFTS)):
        for i, (a, b) in enumerate(subsegments(0.0, 6.0, w, sh)):
            y = 4 - s + (0.17 if i % 2 else -0.17)
            ax.plot([a + 0.02, b - 0.02], [y, y], color=BLUE if s == 4 else MUTED, lw=3, solid_capstyle="butt")
        ax.text(6.15, 4 - s, f"{w:.2f} s window, {sh:.3f} s shift", va="center", fontsize=8, color=DARK)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"scale {5 - i}" for i in range(5)])
    ax.set_xlim(0, 8.3)
    ax.set_xticks(range(7))
    ax.set_xlabel("time (s), 6 s of continuous speech")
    ax.set_title("Five-scale segmentation (Table 2); the finest scale (blue) defines the time index t", loc="left", fontsize=10)
    ax.spines["left"].set_visible(False)
    fig.savefig(FIGS / "fig5_multiscale_windows.png")
    plt.close(fig)


def fig9_comparison():
    rows = [r for r in paper.TABLE4 if "2023 data" not in r[0]]
    rows = sorted(rows, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(rows))
    for i, (name, _, ev, note) in enumerate(rows):
        prop = name == "Proposed system"
        ax.barh(i, ev, color=BLUE if prop else GREY, height=0.62, hatch="///" if "confirmed" in note else None,
                edgecolor="white", linewidth=0)
        ax.text(ev + 0.3, i, f"{ev:.2f}", va="center", fontsize=8, color=DARK)
    labels = [r[0].replace(", post-evaluation", " (post-eval.)") for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("evaluation DER (%), zero collar")
    ax.set_xlim(0, 40)
    ax.set_title("Table 4: DISPLACE 2024 evaluation set (2023-data rows omitted)", loc="left", fontsize=10)
    fig.text(0.01, -0.04, "Hatched: source still to be confirmed by the authors. Post-eval.: post-evaluation submission. "
             "Values copied from the paper.", fontsize=7, color=MUTED)
    fig.savefig(FIGS / "fig9_eval_der_comparison.png")
    plt.close(fig)


def fig10_ablation():
    rows = paper.TABLE5
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.38
    for i, (name, dev, ev, d) in enumerate(rows):
        base = i == 0
        train_err = i >= 3   # rows with the trained decoder: dev DER is training-set error
        ax.bar(i - w / 2, dev, w, color=ORANGE, alpha=0.55 if train_err else 1, hatch="///" if base else None,
               edgecolor="white", linewidth=0)
        ax.bar(i + w / 2, ev, w, color=BLUE, hatch="///" if base else None, edgecolor="white", linewidth=0)
        ax.text(i + w / 2, ev + 0.4, f"{ev:.2f}", ha="center", fontsize=7.5, color=DARK)
        if d is not None:
            ax.text(i + w / 2, ev + 2.0, f"{d:+.2f}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline*", "+ MarbleNet", "+ TitaNet-L", "+ MSDD static", "+ dynamic\n(full)"], fontsize=8)
    ax.set_ylabel("DER (%)")
    ax.set_ylim(0, 36)
    ax.bar(0, 0, color=ORANGE, label="dev (faded: training-set error)")
    ax.bar(0, 0, color=BLUE, label="eval")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Table 5: cumulative ablation; largest step is static -> dynamic (-4.30 pp)", loc="left", fontsize=10)
    ax.text(-0.5, -6.2, "* hatched: baseline figures reported on DISPLACE 2023 data; to be re-run on 2024 data.", fontsize=7, color=MUTED)
    fig.savefig(FIGS / "fig10_ablation.png")
    plt.close(fig)


def fig11_conditions():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)
    for ax, rows, title in [(axes[0], paper.TABLE6, "(a) SNR"), (axes[1], paper.TABLE7[:3], "(b) overlap density")]:
        for i, (name, der, n) in enumerate(rows):
            ax.bar(i, der, 0.6, color=BLUE)
            ax.text(i, der / 2, f"n={n}", ha="center", color="white", fontsize=8)
            ax.text(i, der + 0.6, f"{der:.2f}", ha="center", fontsize=8, color=DARK)
        ax.axhline(paper.PROPOSED_EVAL, ls="--", lw=1, color=MUTED)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0].replace(" overlap", "").replace(" (", "\n(") for r in rows], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("evaluation DER (%)")
    axes[0].text(2.45, paper.PROPOSED_EVAL + 0.6, "overall 21.46", fontsize=7, color=MUTED, ha="right")
    fig.savefig(FIGS / "fig11_conditions.png")
    plt.close(fig)


def fig12_params():
    p = paper.PARAMS_M
    parts = [("TitaNet-Large", p["titanet_large"]), ("scorer + decoder", p["scorer_plus_decoder"]), ("MarbleNet", p["marblenet"])]
    fig, ax = plt.subplots(figsize=(7.5, 1.6))
    left = 0
    for (name, v), c in zip(parts, [GREY, BLUE, MUTED]):
        ax.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", linewidth=2)
        left += v
    ax.text(p["titanet_large"] / 2, 0, "TitaNet-Large 23.0 M (frozen)", ha="center", va="center", color=DARK, fontsize=8)
    ax.annotate("scorer + decoder ~1.5 M (~6 %, trained)", xy=(23.75, 0.25), xytext=(16, 0.62), fontsize=8, color=DARK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.annotate("MarbleNet 0.31 M", xy=(24.66, -0.25), xytext=(19, -0.72), fontsize=8, color=DARK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xlim(0, 26)
    ax.set_ylim(-0.9, 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("parameters (millions), as stated in Table 8 (TitaNet paper reports 25.3 M)")
    fig.savefig(FIGS / "fig12_parameter_budget.png")
    plt.close(fig)


def reproduced():
    """Table 5 rows this code can produce, if their score files exist.
    Several seeds of a configuration are averaged (mean is shown, n in the label)."""
    groups = [("+ TitaNet-Large (clustering only)", "eval_clustering", 26.75),
              ("+ Multi-scale decoder, static weights", "eval_static", 25.76),
              ("+ Dynamic scale weighting (full system)", "eval_dynamic", 21.46)]
    rows = []
    for label, prefix, paper_val in groups:
        files = sorted(REPRO.glob(f"{prefix}*.json"))
        vals = [json.loads(f.read_text())["corpus"]["DER"] for f in files]
        rows.append((label, paper_val, float(np.mean(vals)) if vals else None, len(vals)))
    md = ["# Table 5, rows produced by this code", "",
          "| Configuration | Paper eval DER | Reproduced eval DER | Runs |", "|---|---|---|---|"]
    md += [f"| {l} | {p:.2f} | {'pending' if r is None else f'{r:.2f}'} | {n} |" for l, p, r, n in rows]
    md += ["", "Rows 1-2 of the paper's Table 5 need the official DISPLACE baseline system and are not part of this code."]
    (REPRO / "table5_reproduced.md").write_text("\n".join(md) + "\n")
    if not any(r[2] is not None for r in rows):
        print("no reproduced score files yet: results/reproduced/table5_reproduced.md lists all rows as pending")
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    w = 0.38
    labelled = False
    for i, (l, p, r, n) in enumerate(rows):
        ax.bar(i - w / 2, p, w, color=GREY, label="paper" if i == 0 else None)
        if r is not None:
            ax.bar(i + w / 2, r, w, color=BLUE, label=None if labelled else "reproduced")
            labelled = True
            ax.text(i + w / 2, r + 0.4, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i - w / 2, p + 0.4, f"{p:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["clustering only", "static weights", "dynamic weights"])
    ax.set_ylabel("evaluation DER (%)")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "fig_ablation_paper_vs_reproduced.png")
    plt.close(fig)


def measured_weights(npz_path, seconds=60.0):
    z = np.load(npz_path)
    w, b = z["weights"].astype(float), z["bounds"]
    t = b.mean(1)
    sel = t <= t[0] + seconds
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    colors = ["#1f4e79", BLUE, "#1baf7a", "#eda100", ORANGE]
    for s in range(w.shape[0]):
        ax.plot(t[sel], w[s, sel], lw=1.4, color=colors[s % 5], label=f"{WINDOWS[s]:.2f} s")
    ax.axhline(1 / w.shape[0], ls=":", color=MUTED, lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("w_s(t)")
    ax.legend(ncol=5, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title(f"Measured scale weights, {Path(npz_path).stem} (dotted: static 1/5)", loc="left", fontsize=10, pad=22)
    fig.savefig(FIGS / "fig_measured_scale_weights.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights_npz", default=None, help="a weights file written by step 5, for the measured-weights figure")
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    REPRO.mkdir(parents=True, exist_ok=True)
    paper_tables()
    fig5_windows()
    fig9_comparison()
    fig10_ablation()
    fig11_conditions()
    fig12_params()
    reproduced()
    if a.weights_npz:
        measured_weights(a.weights_npz)
    print(f"tables: {PAPER}\nfigures: {FIGS}")


if __name__ == "__main__":
    main()
