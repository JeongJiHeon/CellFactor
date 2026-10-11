#!/usr/bin/env python3
"""원고 v2(확정안) Figure 4 — Biological interpretability and orthogonal validation of prognostic programs.
  a 기존 Fig. 5a  세포유형 × 프로그램 위험 효과 상위 5(불리)·5(유리), 체인 6개 평균, 숫자 = 같은 부호 체인 수
  b 기존 Fig. 5b  체인 간 효과 코사인(명명 축)
  c 기존 Fig. 5d  PAM50 기대 축 AUROC(TCGA-BRCA, 바코드 연결 n=943)
  d 기존 Fig. 5c  독립 단일세포(GSE161529) 벌크→세포 상태 복원 — CellFactor 만(내부 변형 비교 제외)
  e 기존 Fig. 6b  TP53 변이 vs 야생형 악성 증식 기여(조성 가중 · 상태 단독†)
재계산 없음. 원천 final_results/05_interpretation/*, results/mech/exp2/TP53_*. 출력 final_results/figures/manuscript_v2/Fig4.{pdf,png}"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parents[4]; FR = ROOT / "final_results"; MR = ROOT / "results/mech"
OUTD = FR / "figures/manuscript_v2"; OUTD.mkdir(parents=True, exist_ok=True)
BLUE, RED, INK, INK2, MUTED, GRID, GREY, MRED = "#2a78d6", "#d64545", "#16181b", "#52514e", "#8a8985", "#e6e5e1", "#b9b8b3", "#c8323c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
interp = json.load(open(FR / "05_interpretation/program_weights/INTERP_ACAND3K.json"))
stab = pd.read_csv(FR / "05_interpretation/program_weights/chain_stability.csv")
sr = json.load(open(FR / "05_interpretation/state_recovery/breast_chains.json"))
# 4c = TCGA 바코드 연결(PAM50_BARCODE.json, 979 라벨 · Normal 제외 943); 이전 (OS.time, OS) 유일 키 연결(539)은 pam50_axis_auroc.csv 에 보존
_bc = json.load(open(FR / "05_interpretation/pam50/PAM50_BARCODE.json"))["barcode"]["A_cand3k"]
pam = pd.DataFrame([{"arm": "A_cand3k", "subtype": s, "n": _bc[s]["n"], "expected_axis": _bc[s]["axis"], "auroc": _bc[s].get("auroc"), "rank": _bc[s].get("rank")}
                    for s in ("BRCA_Basal", "BRCA_LumA", "BRCA_LumB", "BRCA_Her2")])
TP = pd.read_csv(MR / "exp2/TP53_PATIENTS.csv"); TS = json.load(open(MR / "exp2/TP53_STATE.json"))
CT = {"Cancer Epithelial": "Cancer epithelial", "Malignant cells": "Malignant", "Normal Epithelial": "Normal epithelial"}


def prog_label(p):
    src = " (HM)" if p.startswith("HM_") else (" (CS)" if p.startswith("CS_") else "")
    b = p.replace("HM_", "").replace("CS_", "").replace("PDAC_", "").replace("_", " ").title()
    for a, c in (("Estrogen Response", "Estrogen resp."), ("Er ", "ER "), ("Tgf ", "TGF-"), ("Kras", "KRAS"), ("Tam ", "TAM "), ("Caf ", "CAF "), ("Pi3K/Akt/Mtor", "PI3K/AKT/mTOR"), ("Metabolism", "metab."), ("Signaling", "sig."), ("Homeostasis", "homeost.")):
        b = b.replace(a, c)
    return b + src


fig = plt.figure(figsize=(7.2, 10.2))
G = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.95, 0.95], hspace=0.42, left=0.19, right=0.985, top=0.97, bottom=0.075)
LET = {}

# ---------- a ----------
ga = G[0].subgridspec(1, 3, wspace=1.45)
for i, (o, on) in enumerate((("breast", "Breast"), ("liver", "Liver"), ("pancreas", "Pancreas"))):
    ax = fig.add_subplot(ga[0, i]); p = interp["programs"][o]
    rows = sorted(p["top_adverse"][:5] + p["top_protective"][:5], key=lambda r: -r["effect"])
    y = np.arange(len(rows)); eff = np.array([r["effect"] for r in rows])
    ax.barh(y, eff, color=[RED if e > 0 else BLUE for e in eff], height=0.62)
    lim = np.abs(eff).max() * 1.45; ax.set_xlim(-lim, lim); ax.invert_yaxis()
    for k, r in enumerate(rows):
        n_ok = r["sign_agree_chains"]; col = INK2 if n_ok == r["n_chains"] else "#c0392b"
        ax.text(r["effect"] + (0.03 if r["effect"] > 0 else -0.03) * lim, k, f"{n_ok}/{r['n_chains']}", va="center", ha="left" if r["effect"] > 0 else "right", fontsize=6.4, color=col)
    ax.set_yticks(y); ax.set_yticklabels([""] * len(y)); ax.tick_params(axis="y", length=0)
    for k, r in enumerate(rows):
        ax.text(-lim * 1.04, k - 0.13, prog_label(r["program"]), ha="right", va="center", fontsize=6.4, color=INK, transform=ax.transData)
        ax.text(-lim * 1.04, k + 0.27, CT.get(r["cell_type"], r["cell_type"]), ha="right", va="center", fontsize=6.4, color=MUTED, transform=ax.transData)
    ax.axvline(0, color=INK2, lw=0.6); ax.axhline(4.5, color=GRID, lw=0.6, ls="--")
    ax.set_xlabel("Risk effect (W × SD)", fontsize=6.8); ax.tick_params(axis="x", labelsize=6.3)
    ax.set_title(on, fontsize=8, fontweight="bold", color=INK, pad=12)
    ax.text(0.02, 1.0, "protective", transform=ax.transAxes, fontsize=6.4, color=BLUE, va="bottom"); ax.text(0.98, 1.0, "adverse", transform=ax.transAxes, fontsize=6.4, color=RED, va="bottom", ha="right")
    if i == 0: LET["a"] = ax

# ---------- b · c · d ----------
gm = G[1].subgridspec(1, 3, width_ratios=[0.55, 0.95, 1.1], wspace=1.05)
ax = fig.add_subplot(gm[0, 0])
ax.bar(range(3), stab.effect_cos_named_mean, yerr=stab.effect_cos_named_sd, color=BLUE, width=0.6, error_kw=dict(ecolor=INK2, lw=0.7, capsize=2))
for k, v in enumerate(stab.effect_cos_named_mean): ax.text(k, v + stab.effect_cos_named_sd.iloc[k] + 0.03, f"{v:.2f}", ha="center", fontsize=6.4)
ax.set_xticks(range(3)); ax.set_xticklabels(["Br", "Li", "Pa"], fontsize=6.8, rotation=0); ax.set_ylim(0, 1.05)
ax.set_ylabel("Effect cosine between chains\n(named axes, mean ± SD)", fontsize=6.8); ax.set_title("Chain stability", fontsize=7.6, loc="left", color=INK); LET["b"] = ax

ax = fig.add_subplot(gm[0, 1]); q = pam[pam.arm == "A_cand3k"].reset_index(drop=True)
for k, r in q.iterrows():
    if pd.isna(r.auroc): ax.bar(k, 0.5, bottom=0.5, color="white", edgecolor=GRID, hatch="////", width=0.75); ax.text(k, 0.6, "no\naxis", ha="center", fontsize=6.4, color=MUTED)
    else:
        ax.bar(k, r.auroc - 0.5, bottom=0.5, color=BLUE, width=0.75)
        inside = r.auroc > 0.9   # 축 상한 1.0: 높은 막대는 막대 안쪽(흰 글씨)에 표기
        ax.text(k, r.auroc - 0.01 if inside else r.auroc + 0.01, f"{r.auroc:.2f}\n#{int(r['rank'])}", ha="center", fontsize=6.4, color="white" if inside else INK, va="top" if inside else "bottom")
ax.set_xticks(range(len(q))); ax.set_xticklabels([f"{s.replace('BRCA_', '')}\n{n}" for s, n in zip(q.subtype, q.n)], fontsize=6.4); ax.set_xlim(-0.6, len(q) - 0.4)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC of expected subtype axis\n(TCGA-BRCA)", fontsize=6.8); ax.set_title("PAM50 correspondence", fontsize=7.6, loc="left", color=INK); LET["c"] = ax

pt = sr["arms"]["A_cand3k"]["per_type_mean"]; types = sorted(pt, key=lambda t: -pt[t]["r"]["mean"])
ax = fig.add_subplot(gm[0, 2]); yy = np.arange(len(types))
for k in yy[1::2]: ax.axhspan(k - 0.5, k + 0.5, color="#f4f3f0", zorder=0)
ax.errorbar([pt[t]["r"]["mean"] for t in types], yy, xerr=[pt[t]["r"]["sd"] for t in types], fmt="o", ms=4.5, color=BLUE, mec="white", mew=0.5, ecolor=BLUE, elinewidth=0.9)
ax.set_yticks(yy); ax.set_yticklabels([f"{CT.get(t, t)} ({pt[t]['n_donors']})" for t in types], fontsize=6.4); ax.set_ylim(len(types) - 0.5, -0.5)
ax.set_xlim(0.1, 0.8); ax.set_xticks([0.2, 0.4, 0.6, 0.8]); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("State recovery r\n(median over axes; 6 chains)", fontsize=6.6)
ax.set_title("Independent scRNA-seq\nGSE161529 (donors)", fontsize=7.6, loc="left", color=INK); LET["d"] = ax

# ---------- e ----------
ge = G[2].subgridspec(1, 3, wspace=0.12)
for j, (o, on) in enumerate((("breast", "Breast"), ("liver", "Liver"), ("pancreas", "Pancreas"))):
    ax = fig.add_subplot(ge[0, j]); d = TP[(TP.organ == o) & TP.TP53_mut.notna()]
    for k, v in enumerate(("comp_weighted", "state_only")):
        sd = d[v].std()
        for gidx, (col, dx) in enumerate(((GREY, -0.2), (MRED, 0.2))):
            vals = (d[d.TP53_mut == gidx][v] - d[v].median()) / sd
            bp = ax.boxplot(vals, positions=[k + dx], widths=0.32, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.0),
                            whiskerprops=dict(color=INK2, lw=0.6), capprops=dict(color=INK2, lw=0.6))
            bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_edgecolor(INK2); bp["boxes"][0].set_linewidth(0.6)
        ax.text(k, 3.75, f"AUC {TS[o]['all'][v]['auc']:.2f}\nheld-out {TS[o]['heldout'][v]['auc']:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Composition-\nweighted", "State only†"], fontsize=6.8); ax.set_ylim(-2.6, 4.9); ax.set_xlim(-0.55, 1.55)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0); ax.set_title(f"{on} · TP53 {TS[o]['all']['n_mut']} mut / {TS[o]['all']['n_wt']} wt", fontsize=7.4, color=INK)
    if j == 0: ax.set_ylabel("Malignant proliferation contribution\n(SD from median)", fontsize=6.8); LET["e"] = ax
    else: ax.set_yticklabels([])
fig.legend(handles=[Patch(facecolor=GREY, edgecolor=INK2, label="TP53 wild type"), Patch(facecolor=MRED, edgecolor=INK2, label="TP53 mutant")], frameon=False, fontsize=6.6, loc="lower center", ncol=2, bbox_to_anchor=(0.57, 0.0))

fig.canvas.draw(); R_ = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
for k, a in LET.items():
    t = a._left_title if a._left_title.get_text() else a.title
    y1 = inv.transform(t.get_window_extent(R_).get_points())[1, 1]
    x = {"a": 0.008, "e": 0.008, "b": 0.008}.get(k, a.get_position().x0 - 0.085)
    fig.text(x, y1, k, fontsize=10, fontweight="bold", va="top")
fig.savefig(OUTD / "Fig4.pdf", dpi=400); fig.savefig(OUTD / "Fig4.png", dpi=300); print("Fig4 v3 ok")
