#!/usr/bin/env python3
"""원고 v2(확정안) 보충 S4 = 기존 Fig. 5e(문헌 방향 일치·순열 귀무), S5 = 기존 Fig. 6c(상위 5축 삭제). 재계산 없음.
출력 final_results/figures/manuscript_v2/Supp/{S4_literature_direction, S5_program_deletion}.{pdf,png}"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[4]; FR = ROOT / "final_results"; MR = ROOT / "results/mech"; OUTD = FR / "figures/manuscript_v2/Supp"; OUTD.mkdir(parents=True, exist_ok=True)
BLUE, INK, INK2, GREY, BASE = "#2a78d6", "#16181b", "#52514e", "#b9b8b3", "#9a9994"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
# S4
nn = pd.read_csv(FR / "05_interpretation/literature_direction/named_null_summary.csv")
fig, ax = plt.subplots(figsize=(3.0, 2.8))
for k, r in nn.iterrows():
    ax.errorbar(k, r.null_mean, yerr=1.96 * r.null_sd, fmt="_", ms=10, color=BASE, elinewidth=1.6, capsize=0)
    ax.plot(k, r.concordance, "o", ms=6, color=BLUE, mec="white", mew=0.6, zorder=3)
    ax.text(k, max(r.concordance, r.null_mean + 1.96 * r.null_sd) + 0.05, f"{int(r.k_agree)}/{int(r.n_tested)}\nP = {r.p_perm:.2f}", ha="center", va="bottom", fontsize=6.6)
ax.set_xticks(range(len(nn))); ax.set_xticklabels(["Breast", "Liver", "Pancreas"]); ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Concordance with literature-\nexpected sign")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Observed"), Line2D([], [], marker="_", ls="-", color=BASE, label="Permutation null, mean ± 1.96 SD")], fontsize=6.2, frameon=False, loc="upper left")
fig.savefig(OUTD / "S4_literature_direction.pdf", bbox_inches="tight"); fig.savefig(OUTD / "S4_literature_direction.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# S5
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6)); fig.subplots_adjust(wspace=0.3, top=0.78, bottom=0.18, left=0.08, right=0.98)
for ax, (o, on) in zip(axes, (("breast", "Breast"), ("liver", "Liver"), ("pancreas", "Pancreas"))):
    S = json.load(open(MR / f"exp1/{o}_SUMMARY.json"))["F2"]; rnd = np.load(MR / f"exp1/{o}_DRAWS.npz")["random5"]; obs = S["dC_top5"]; lo, hi = S["dC_top5_ci95"]
    hx = max(obs, hi, rnd.max()) * 1.08 + 0.005; lx = min(rnd.min(), lo, 0) - 0.005
    ax.hist(rnd, bins=np.linspace(lx, hx, 45), color=GREY, edgecolor="white", lw=0.3); ax.axvline(S["random5_q95"], color=INK2, lw=0.8, ls="--")
    yl = ax.get_ylim()[1]; ax.axvline(obs, color=BLUE, lw=1.6); ax.set_ylim(0, yl * 1.32); ax.plot([lo, hi], [yl * 1.16] * 2, color=BLUE, lw=1.4); ax.scatter(obs, yl * 1.16, s=20, color=BLUE, zorder=3)
    ax.set_xlim(lx, hx); ax.set_title(f"{on}\ntop-5 ΔC {obs:.3f} (P = {S['p']:.3f})\nrandom 95th pct {S['random5_q95']:.3f}", fontsize=6.8)
    ax.set_xlabel("ΔC-index after deleting 5 axes", fontsize=6.8); ax.tick_params(labelsize=6.2)
axes[0].set_ylabel("Random 5-axis sets")
fig.savefig(OUTD / "S5_program_deletion.pdf", bbox_inches="tight"); fig.savefig(OUTD / "S5_program_deletion.png", dpi=300, bbox_inches="tight"); print("S4 S5 ok")
