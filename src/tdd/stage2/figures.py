"""Stage 2 figures, drawn only from results/stage2/*.json.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.stage2.figures  -> results/stage2/figures/*.png

Colour = dataset identity, fixed everywhere: Anti-UAV blue, Svanström orange
(slots 1 and 2 of the validated default palette). Text stays in ink colours.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import ROOT, read_json

RES = ROOT / "results" / "stage2"
OUT = RES / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
AUV, SVAN = "Anti-UAV test", "Svanström dev"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def _pct(ax, axis="y"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def score_distribution(d):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, key, name, col in ((axes[0], "auv_test", AUV, BLUE), (axes[1], "svanstrom", SVAN, ORANGE)):
        h = d["tp_score_hist"][key]
        c = np.asarray(h["counts"], float)
        share = c / c.sum()
        ax.bar(range(len(c)), share, width=0.82, color=col)
        e = h["edges"]
        labels = [f"{a:g}–{min(b, 1):g}" for a, b in zip(e[:-1], e[1:])]
        ax.set_xticks(range(len(c)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_title(name)
        ax.set_xlabel("score of the best box with IoU ≥ 0.5")
        ax.grid(axis="x", visible=False)
        _pct(ax)
        thr_i = next(i for i, (a, b) in enumerate(zip(e[:-1], e[1:])) if a <= d["threshold_stage1"] < b)
        ax.axvline(thr_i - 0.5 + (d["threshold_stage1"] - e[thr_i]) / (e[thr_i + 1] - e[thr_i]), color=INK2, lw=1, ls="--")
        ax.annotate("threshold 0.954", (thr_i, ax.get_ylim()[1] if ax is axes[1] else share.max()), xytext=(-6, -2),
                    textcoords="offset points", ha="right", va="top", fontsize=8, color=INK2)
    axes[0].set_ylabel("share of drones found at IoU ≥ 0.5")
    fig.suptitle("Scores of correct boxes: on Svanström a large share sits near zero", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.04, "Stage 1 model. Near-zero scores are duplicate boxes: the high-score box is too tight for the loose label, a low-score one fits it.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, "score_distribution.png")


def size_recall(d):
    names = d["size_bins_reference_px"]
    a, s = d["recall_by_size"]["auv_test_plus_supplement"], d["recall_by_size"]["svanstrom"]
    sc = d["recall_by_size_svanstrom_corrected_size"]["table"]
    k = d["recall_by_size_svanstrom_corrected_size"]["k"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    x = np.arange(len(names))
    for ax, crit, title in ((axes[0], "recall_iou50", "IoU ≥ 0.5 with the label"), (axes[1], "recall_center", "Box centre inside the label")):
        for tab, col, lab, mk in ((a, BLUE, AUV, "o"), (sc, ORANGE, f"{SVAN} (label size ÷ {k:g})", "s")):
            y = [tab[n][crit] if tab[n]["n"] >= 20 else np.nan for n in names]
            ax.plot(x, y, color=col, marker=mk, label=lab, markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_xlabel("target size, √(w·h) in 640×512 pixels")
        _pct(ax)
    axes[0].set_ylabel("recall at threshold 0.954")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Recall by target size: the gap is the IoU test, except for the smallest targets", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.03, "Bins with fewer than 20 boxes are not drawn. Svanström sizes are divided by the box-size factor because its labels are drawn with a margin.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, "size_recall.png")


def far_recall(d):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, col, lab in (("auv_test", BLUE, f"{AUV} (drone-free frames)"), ("svanstrom", ORANGE, f"{SVAN} (bird, aeroplane, helicopter frames)")):
        c = sorted(d["far_recall_curve"][key], key=lambda r: r["thr"])
        far, rec, rec10 = [r["far"] for r in c], [r["recall"] for r in c], [r["recall_iou10"] for r in c]
        ax.plot(far, rec, color=col, label=f"{lab}, IoU ≥ 0.5")
        ax.plot(far, rec10, color=col, ls=(0, (1, 1.5)), label=f"{lab}, IoU ≥ 0.1")
        p = min(c, key=lambda r: abs(r["thr"] - d["threshold_stage1"]))
        ax.plot([p["far"]], [p["recall"]], "o", color=col, markeredgecolor=SURFACE, markeredgewidth=2, markersize=9)
        ax.annotate("threshold 0.954", (p["far"], p["recall"]), xytext=(8, -12), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("share of non-drone frames with an alarm")
    ax.set_ylabel("drone recall")
    _pct(ax)
    _pct(ax, "x")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Alarm rate against recall as the threshold moves (stage 1 model)")
    fig.tight_layout()
    _save(fig, "far_vs_recall.png")


def decomposition(d, name="decomposition.png", label=SVAN):
    p, ci = d["decomposition"]["point"], d["decomposition"]["ci95"]
    rows = [("Total gap (IoU ≥ 0.5, threshold 0.954)", "gap"), ("Box position test (IoU 0.5 → 0.1)", "share_localisation"),
            ("Score threshold (0.954 → 0.05)", "share_threshold"), ("Target size mix", "share_size"),
            ("Overlap between the three", "overlap"), ("Left after all three", "unexplained")]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(rows))[::-1]
    vals = [p[k] * 100 for _, k in rows]
    cols = [INK2] + [BLUE] * 3 + ["#9a9992"] * 2
    ax.barh(y, vals, height=0.58, color=cols)
    for yi, (lab, k), v in zip(y, rows, vals):
        lo, hi = ci[k][0] * 100, ci[k][1] * 100
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2)
        ax.annotate(f"{v:+.1f} pts  [{lo:+.1f}, {hi:+.1f}]", (max(hi, 0), yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(min(vals) - 12, max(vals) + 22)
    ax.set_xlabel("percentage points of recall")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Where the recall gap comes from: Anti-UAV test {p['recall_auv'] * 100:.1f}% → {label} {p['recall_svan'] * 100:.1f}%")
    fig.text(0.01, -0.05, "Each bar: how much the gap shrinks when only that factor is changed. Lines: 95% intervals, video-level bootstrap.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, name)


def before_after(final):
    """Svanström test: every frozen configuration against the stage 1 reference."""
    ref = final["reference_S1"]["detector"]
    def row(lab, dd):
        return (lab, dd["recall_iou50"], dd["ci95"]["recall_iou50"], dd["recall_center"], dd["ci95"]["recall_center"], dd["far"], dd["ci95"]["far"])
    rows = [row("Stage 1 as it was", ref), row("Stage 1, box size calibrated only", final["S1_box_calibrated"]["detector"])]
    labels = {"A": "A: no training", "B": "B: small-target training", "C4": "C4: + 4 target videos",
              "C12": "C12: + 12 target videos", "C37": "C37: + 37 target videos"}
    for name, lab in labels.items():
        m = final["models"].get(name)
        if not m or m["status"] != "ok":
            continue
        rows.append(row(lab, m["detector"]["T_base"]))
        rows.append(row("   + tracking", m["tracker"]["T_base"]))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 0.42 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (vi, ci_i, title, col) in zip(axes, ((1, 2, "Recall, IoU ≥ 0.5", ORANGE), (3, 4, "Recall, centre inside label", ORANGE),
                                                  (5, 6, "Alarm rate, non-drone frames", INK2))):
        for yi, r in zip(y, rows):
            ax.plot(r[ci_i], [yi, yi], color=col, lw=2, alpha=0.45)
            ax.plot([r[vi]], [yi], "o", color=col, markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=9)
            ax.annotate(f"{100 * r[vi]:.1f}%", (r[vi], yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        _pct(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(rows) - 0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows])
    fig.suptitle("Svanström test videos, before and after (alarm rate matched to stage 1 on dev videos)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.04, f"Dots: value on the {final['videos']} test videos. Bars: 95% intervals, video-level bootstrap. Thresholds, box-size factor, input settings and tracker were chosen on dev videos only.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, "before_after.png")


def main():
    d = read_json(RES / "diagnosis.json")
    score_distribution(d)
    size_recall(d)
    far_recall(d)
    decomposition(d)
    if (RES / "diagnosis_test.json").exists():
        decomposition(read_json(RES / "diagnosis_test.json"), "decomposition_test.png", "Svanström test")
    if (RES / "final__svanstrom.json").exists():
        before_after(read_json(RES / "final__svanstrom.json"))


if __name__ == "__main__":
    main()
