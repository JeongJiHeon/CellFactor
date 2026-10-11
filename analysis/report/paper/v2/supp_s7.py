#!/usr/bin/env python3
"""보충 S7 — ground-truth simulation(PLAN_SIMULATION). 재계산 없음.
원천 results/mech/simulation/{SCEN}_r{r}.json. 출력 final_results/figures/manuscript_v2/Supp/S7_simulation.{pdf,png}"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[4]; D = ROOT / "results/mech/simulation"; OUTD = ROOT / "final_results/figures/manuscript_v2/Supp"
INK, INK2, GREY = "#16181b", "#52514e", "#b9b8b3"
MC = {"CellFactor": "#2a78d6", "oracle_comp": "#d9822b", "oracle_state": "#3a9e6f", "oracle_both": "#52514e"}
ML = {"CellFactor": "CellFactor", "oracle_comp": "Oracle composition", "oracle_state": "Oracle state", "oracle_both": "Oracle both"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
SCENS = ["A", "B", "C0", "C5", "C9", "D"]; SL = {"A": "A", "B": "B", "C0": "C\nρ 0", "C5": "C\nρ 0.5", "C9": "C\nρ 0.9", "D": "D"}
METHODS = list(MC)
R = {s: [json.load(open(p)) for p in sorted(q for q in D.glob(f"{s}_r*.json") if not q.stem.endswith("_a0"))] for s in SCENS}; R = {s: v for s, v in R.items() if v}; SC = [s for s in SCENS if s in R]
TYPES3 = ["Cancer Epithelial", "T-cells", "CAFs"]; TL = {"Cancer Epithelial": "Malignant epithelial", "T-cells": "T cells", "CAFs": "CAFs"}
TERMS = ["Cancer Epithelial|CS_PROLIFERATION", "Cancer Epithelial|HM_Estrogen_Response_Early", "T-cells|T_CYTOTOXIC", "CAFs|HM_TGF_beta_Signaling", "Cancer Epithelial|CS_HYPOXIA"]
TERML = {"Cancer Epithelial|CS_PROLIFERATION": "Malig. proliferation (+)", "Cancer Epithelial|HM_Estrogen_Response_Early": "Malig. estrogen early (−)",
         "T-cells|T_CYTOTOXIC": "T cytotoxic (−)", "CAFs|HM_TGF_beta_Signaling": "CAF TGF-β (+)", "Cancer Epithelial|CS_HYPOXIA": "Malig. hypoxia (+, C)"}


def tag(ax, s, x=-0.2): ax.text(x, 1.12, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def strip(ax, xs, vals, color, w=0.09):
    rng = np.random.default_rng(0)
    for x, v in zip(xs, vals):
        v = np.asarray([u for u in v if u is not None], float)
        if not len(v): continue
        ax.scatter(x + rng.uniform(-w, w, len(v)), v, s=4, c=color, alpha=0.55, lw=0, zorder=2)
        ax.plot([x - 0.17, x + 0.17], [np.median(v)] * 2, color=color, lw=1.4, zorder=3)


fig = plt.figure(figsize=(7.2, 8.4))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.1, 1], hspace=0.75, wspace=0.38, left=0.09, right=0.98, top=0.93, bottom=0.07)

# a. P1 유형 기여 Spearman, 시나리오별, 4 방법
off = {m: (i - 1.5) * 0.2 for i, m in enumerate(METHODS)}
for j, t in enumerate(TYPES3):
    ax = fig.add_subplot(gs[0, j])
    for m in METHODS:
        strip(ax, [i + off[m] for i in range(len(SC))], [[r["methods"][m]["type_spearman"][t] for r in R[s]] for s in SC], MC[m], w=0.05)
    ax.axhline(0, color=GREY, lw=0.5); ax.axhline(0.5, color=GREY, lw=0.5, ls=":")
    ax.set_xticks(range(len(SC))); ax.set_xticklabels([SL[s] for s in SC], fontsize=6); ax.set_ylim(-0.6, 1.02)
    ax.set_title(f"{TL[t]}: type contribution", fontsize=7, pad=3); ax.tick_params(labelsize=6)
    if j == 0: ax.set_ylabel("Spearman (estimated vs true)", fontsize=6.5); tag(ax, "a")
fig.legend(handles=[Line2D([], [], marker="o", ls="-", color=MC[m], ms=3.5, lw=1.2, label=ML[m]) for m in METHODS],
           loc="upper center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.995))

# b. P2 프로그램 기여 Spearman, 시나리오 A (C 의 악성 저산소는 C ρ 0)
g1 = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.62)
ax = fig.add_subplot(g1[0])
for i, k in enumerate(TERMS):
    s = "C0" if k.endswith("CS_HYPOXIA") else "A"
    if s not in R: continue
    for m in METHODS:
        strip(ax, [i + off[m]], [[r["methods"][m]["terms"][k]["spearman"] for r in R[s]]], MC[m], w=0.05)
ax.axhline(0, color=GREY, lw=0.5); ax.axhline(0.5, color=GREY, lw=0.5, ls=":")
ax.set_xticks(range(len(TERMS))); BL = {"Cancer Epithelial|CS_PROLIFERATION": "Malig.\nprolif. (+)", "Cancer Epithelial|HM_Estrogen_Response_Early": "Malig.\nestrogen (−)", "T-cells|T_CYTOTOXIC": "T\ncytotoxic (−)",
      "CAFs|HM_TGF_beta_Signaling": "CAF\nTGF-β (+)", "Cancer Epithelial|CS_HYPOXIA": "Malig.\nhypoxia (+)"}
ax.set_xticklabels([BL[k] + ("\n[C, ρ 0]" if k.endswith("CS_HYPOXIA") else "\n[A]") for k in TERMS], fontsize=5.8); ax.set_xlim(-0.5, len(TERMS) - 0.5)
ax.set_ylabel("Spearman (estimated vs true\nprogram contribution)", fontsize=6.5); ax.set_ylim(-1.0, 1.02); ax.tick_params(labelsize=6); tag(ax, "b", -0.12)

# c. 부호 일치 (CellFactor), 시나리오 × 항
ax = fig.add_subplot(g1[1])
mat = np.full((len(TERMS), len(SC)), np.nan)
for j, s in enumerate(SC):
    for i, k in enumerate(TERMS):
        if k in R[s][0]["methods"]["CellFactor"]["terms"]: mat[i, j] = sum(r["methods"]["CellFactor"]["terms"][k]["sign_ok"] for r in R[s]) / len(R[s])
ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(len(TERMS)):
    for j, s in enumerate(SC):
        if np.isfinite(mat[i, j]):
            n = len(R[s]); ax.text(j, i, f"{int(round(mat[i, j] * n))}/{n}", ha="center", va="center", fontsize=5.6, color="white" if mat[i, j] > 0.6 else INK)
ax.set_xticks(range(len(SC))); ax.set_xticklabels([SL[s] for s in SC], fontsize=5.6); ax.set_yticks(range(len(TERMS))); ax.set_yticklabels([TERML[k] for k in TERMS], fontsize=5.6)
ax.spines[:].set_visible(False); ax.tick_params(length=0); ax.set_title("CellFactor sign agreement", fontsize=7, pad=3); tag(ax, "c", -0.75)

# d. P3 내피 저산소 귀속 비율 vs ρ
ax = fig.add_subplot(gs[2, 0]); CS = [s for s in ("C0", "C5", "C9") if s in R]; rho = {"C0": 0, "C5": 0.5, "C9": 0.9}
for m in METHODS:
    strip(ax, [i + off[m] for i in range(len(CS))], [[r["methods"][m]["P3_endo_share_contrib"] for r in R[s]] for s in CS], MC[m], w=0.05)
ax.set_xticks(range(len(CS))); ax.set_xticklabels([f"ρ {rho[s]}" for s in CS], fontsize=6)
ax.set_xlabel("Correlation of hypoxia latent\n(malignant vs endothelial)", fontsize=6.3); ax.set_ylabel("Share attributed to endothelial\nhypoxia (true share 0)", fontsize=6.3)
ax.set_ylim(0, 1); ax.tick_params(labelsize=6); tag(ax, "d")

# e. C-index
ax = fig.add_subplot(gs[2, 1]); CM = [("true", INK, "True risk η"), ("oracle_both", MC["oracle_both"], ML["oracle_both"]), ("CellFactor", MC["CellFactor"], "CellFactor"), ("bulk", GREY, "Bulk Cox")]
for i, (k, c, _) in enumerate(CM):
    get = (lambda r: r["c_true_eta"]) if k == "true" else (lambda r: r["bulk_cox"]["c_index"]) if k == "bulk" else (lambda r, k=k: r["methods"][k]["c_index"])
    strip(ax, [j + (i - 1.5) * 0.2 for j in range(len(SC))], [[get(r) for r in R[s]] for s in SC], c, w=0.04)
ax.set_xticks(range(len(SC))); ax.set_xticklabels([SL[s] for s in SC], fontsize=6); ax.set_ylabel("Test C-index", fontsize=6.5); ax.tick_params(labelsize=6); ax.set_ylim(0.5, 0.85)
ax.legend(handles=[Line2D([], [], color=c, lw=1.4, label=l) for _, c, l in CM], fontsize=5.6, frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8); tag(ax, "e")

# f. 상태 복원: M̂ vs M*, 시나리오 A
ax = fig.add_subplot(gs[2, 2]); s = "A" if "A" in R else SC[0]; keys = list(R[s][0]["state"])
KL = {"Cancer Epithelial|CS_PROLIFERATION": "Malig. prolif.", "Cancer Epithelial|HM_Estrogen_Response_Early": "Malig. estrogen", "Cancer Epithelial|CS_HYPOXIA": "Malig. hypoxia",
      "T-cells|T_CYTOTOXIC": "T cytotoxic", "CAFs|HM_TGF_beta_Signaling": "CAF TGF-β", "Endothelial|CS_HYPOXIA": "Endo. hypoxia"}
for i, k in enumerate(keys):
    strip(ax, [i - 0.15], [[r["state"][k]["M_hat"] for r in R[s]]], MC["CellFactor"], w=0.05)
    strip(ax, [i + 0.15], [[r["state"][k]["M_star"] for r in R[s]]], MC["oracle_state"], w=0.05)
ax.axhline(0, color=GREY, lw=0.5); ax.set_xticks(range(len(keys))); ax.set_xticklabels([KL[k] for k in keys], fontsize=5.6, rotation=40, ha="right")
ax.set_ylabel("Spearman with true state", fontsize=6.5); ax.set_ylim(-0.4, 1.02); ax.tick_params(labelsize=6)
ax.legend(handles=[Line2D([], [], color=MC["CellFactor"], lw=1.4, label="M̂ (from bulk)"), Line2D([], [], color=MC["oracle_state"], lw=1.4, label="M* (cell-level z′)")], fontsize=5.6, frameon=False, loc="lower left"); tag(ax, "f")

OUTD.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(OUTD / f"S7_simulation.{ext}", dpi=300)
print("saved", OUTD / "S7_simulation.png", {s: len(v) for s, v in R.items()})
