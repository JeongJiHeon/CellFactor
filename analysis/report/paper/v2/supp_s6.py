#!/usr/bin/env python3
"""보충 S6 — 유사벌크와 실제 벌크의 도메인 간극, conditional CORAL 켬/끔(PLAN_DOMAIN_GAP + 사후 진단 domain_gap_diag.py). 재계산 없음.
원천 results/mech/domain_gap/{organ}.json · {organ}_fig.npz. 출력 final_results/figures/manuscript_v2/Supp/S6_domain_gap.{pdf,png}"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[4]; D = ROOT / "results/mech/domain_gap"; OUTD = ROOT / "final_results/figures/manuscript_v2/Supp"
BLUE, INK, INK2, GREY, BASE, ORANGE = "#2a78d6", "#16181b", "#52514e", "#b9b8b3", "#9a9994", "#d9822b"
CP, CR = "#9a9994", "#2a78d6"            # pseudo-bulk, real bulk
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ORGS = (("breast", "Breast"), ("liver", "Liver"), ("pancreas", "Pancreas"))
R = {o: json.load(open(D / f"{o}.json")) for o, _ in ORGS}; Z = {o: np.load(D / f"{o}_fig.npz") for o, _ in ORGS}
DG = {o: json.load(open(D / f"{o}_diag.json")) for o, _ in ORGS}; DZ = {o: np.load(D / f"{o}_diag_fig.npz") for o, _ in ORGS}   # 사후 진단

fig = plt.figure(figsize=(7.2, 9.6))
gs = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 1.05, 1.05], hspace=0.95, wspace=0.75, left=0.08, right=0.98, top=0.95, bottom=0.05)
gs4 = gs[3, :].subgridspec(1, 4, wspace=0.45)


def scat(ax, pc, y, title, ve=None):
    ax.scatter(pc[y == 1, 0], pc[y == 1, 1], s=2.5, c=CR, alpha=0.45, lw=0, rasterized=True)
    ax.scatter(pc[y == 0, 0], pc[y == 0, 1], s=2.5, c=CP, alpha=0.6, lw=0, rasterized=True)
    ax.set_title(title, fontsize=5.9, pad=3); ax.set_xticks([]); ax.set_yticks([])
    if ve is not None: ax.set_xlabel(f"PC1 ({ve[0] * 100:.0f}%)", fontsize=6, labelpad=1); ax.set_ylabel(f"PC2 ({ve[1] * 100:.0f}%)", fontsize=6, labelpad=1)


def tag(ax, s): ax.text(-0.32, 1.18, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


# a. 입력 PCA (공통 단위 / 모델 입력). 모델 입력 LR AUROC 는 fold 안 도메인별 재표준화 값(전체 자료 표준화 누출 제거, 사후 진단 Q3)
for i, (o, on) in enumerate(ORGS):
    for j, (v, vn) in enumerate((("raw_log2cpm", "log2 CPM"), ("model_input", "model input"))):
        ax = fig.add_subplot(gs[0, 2 * i + j]); a = R[o]["A"]["views"][v]
        lr = a["auroc"]["logistic"] if v == "raw_log2cpm" else DG[o]["Q3_model_input_logistic"]["auc_foldwise_restandardised"]
        scat(ax, Z[o][f"A_{v}_pc"], Z[o]["A_y"], f"{on}: {vn}\nAUROC LR {lr:.2f} · RF {a['auroc']['random_forest']:.2f}", Z[o][f"A_{v}_ve"])
        if i == 0 and j == 0: tag(ax, "a")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=CP, ms=4, label="Pseudo-bulk (training donors)"), Line2D([], [], marker="o", ls="", color=CR, ms=4, label="TCGA bulk")],
           loc="upper center", ncol=2, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 0.995))

# b. 유전자 평균 발현 · 검출률 · SD (공통 단위)
for i, (o, on) in enumerate(ORGS):
    ax = fig.add_subplot(gs[1, 2 * i]); z = Z[o]; a = R[o]["A"]["views"]["raw_log2cpm"]
    ax.scatter(z["A_gene_mean_P"], z["A_gene_mean_R"], s=1.5, c=INK2, alpha=0.35, lw=0, rasterized=True)
    m = max(z["A_gene_mean_P"].max(), z["A_gene_mean_R"].max()); ax.plot([0, m], [0, m], color=GREY, lw=0.6, ls="--")
    ax.set_xlabel("Pseudo-bulk mean", fontsize=6); ax.set_ylabel("TCGA mean", fontsize=6); ax.tick_params(labelsize=5.6)
    ax.set_title(f"{on}: gene means\nr = {a['gene_mean_pearson']:.2f}", fontsize=6.6, pad=3)
    if i == 0: tag(ax, "b")
    ax = fig.add_subplot(gs[1, 2 * i + 1])
    bp = ax.boxplot([z["A_det_P"], z["A_det_R"]], widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=0.8))
    for p, c in zip(bp["boxes"], (CP, CR)): p.set_facecolor(c); p.set_alpha(0.6); p.set_linewidth(0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Pseudo", "TCGA"], fontsize=6); ax.set_ylabel("Gene detection rate", fontsize=6); ax.tick_params(labelsize=5.6)
    ax.set_title(f"zero fraction\n{a['zero_fraction_P']:.3f} vs {a['zero_fraction_R']:.3f}", fontsize=6.4, pad=3)

# c. H 표현(공동 조성 잔차): 유사벌크로만 맞춘 PCA 좌표(유사벌크 PC SD 단위)에 TCGA held-out 사영, 체인 c1, 같은 축 범위
for i, (o, on) in enumerate(ORGS):
    lim = max(np.abs(np.r_[DZ[o][f"{a}_P"], DZ[o][f"{a}_R"]]).max() for a in ("CORAL_off", "CORAL_on")) * 1.05
    for j, (arm, an) in enumerate((("CORAL_off", "CORAL off"), ("CORAL_on", "CORAL on"))):
        ax = fig.add_subplot(gs[2, 2 * i + j]); Pp, Rp = DZ[o][f"{arm}_P"], DZ[o][f"{arm}_R"]
        ax.scatter(Rp[:, 0], Rp[:, 1], s=3, c=CR, alpha=0.6, lw=0, rasterized=True); ax.scatter(Pp[:, 0], Pp[:, 1], s=2.5, c=CP, alpha=0.6, lw=0, rasterized=True)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC1 (pseudo-bulk SD)", fontsize=5.6, labelpad=1); ax.set_ylabel("PC2", fontsize=5.6, labelpad=1)
        ax.set_title(f"{on}: {an}\nCORAL objective {R[o]['B'][arm]['1']['P_vs_Runseen']['coral_objective']:.3f}", fontsize=5.9, pad=3)
        if i == 0 and j == 0: tag(ax, "c")

# d. 정렬 지표 체인 짝 (사후 진단: 목적값의 평균 항은 구성상 0)
def pair(ax, get, title, log=False):
    for i, (o, on) in enumerate(ORGS):
        off = [get(o, "CORAL_off", c) for c in range(1, 7)]; onv = [get(o, "CORAL_on", c) for c in range(1, 7)]
        for a_, b_ in zip(off, onv): ax.plot([i - 0.17, i + 0.17], [a_, b_], color=GREY, lw=0.5)
        ax.scatter([i - 0.17] * 6, off, s=9, color=BASE, zorder=3); ax.scatter([i + 0.17] * 6, onv, s=9, color=BLUE, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in ORGS], fontsize=5.8); ax.set_title(title, fontsize=6.2, pad=3); ax.tick_params(labelsize=5.6)
    if log: ax.set_yscale("log")
dg = lambda k: (lambda o, a, c: DG[o]["arms"][a][str(c)][k])
D4 = (("coral_objective", "CORAL objective\n(covariance term; mean term = 0)", True), ("pooled_raw_mean_diff", "Mean difference,\nfeature units", False),
      ("pooled_sd_mean", "Feature scale\n(pooled SD)", False), ("pooled_std_mean_diff", "Mean difference,\nSD units", False))
for j, (k, kn, lg) in enumerate(D4):
    ax = fig.add_subplot(gs4[j]); pair(ax, dg(k), kn, lg)
    if j == 0:
        tag(ax, "d"); bb = ax.get_position()
        fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BASE, ms=4, label="CORAL off"), Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="CORAL on")], fontsize=6, frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(0.98, bb.y1 + 0.035))

# e. 외부 C(중심) · 상태 오차 · 분류 AUROC
ax = fig.add_subplot(gs[4, 0:2]); tag(ax, "e")
for i, (o, on) in enumerate(ORGS):
    b6 = R[o]["B6"]
    for dx, arm, col in ((-0.12, "CORAL_off", BASE), (0.12, "CORAL_on", BLUE)):
        p = b6["point"][arm]["external_mean"]; lo, hi = b6["external_mean"]["ci95_" + ("on" if arm == "CORAL_on" else "off")]
        ax.plot([i + dx] * 2, [lo, hi], color=col, lw=1.2); ax.scatter(i + dx, p, s=16, color=col, zorder=3)
ax.axhline(0.5, color=GREY, lw=0.6, ls="--"); ax.set_ylim(0.48, 0.70); ax.set_xticks(range(3)); ax.tick_params(labelsize=5.8)
ax.set_xticklabels([f"{n}\nΔ {R[o]['B6']['external_mean']['diff_on_minus_off']:+.3f}\n({R[o]['B6']['external_mean']['ci95'][0]:+.3f}, {R[o]['B6']['external_mean']['ci95'][1]:+.3f})" for o, n in ORGS], fontsize=5.6)
ax.set_title("External C-index (ensemble c1–c6)\nmean, 95% CI; Δ = on − off (paired)", fontsize=6.4, pad=3)
ax = fig.add_subplot(gs[4, 2:4])
for i, (o, on) in enumerate(ORGS):
    off = np.array([R[o]["B"]["CORAL_off"][str(c)]["state"]["mae_ratio_mean"] for c in range(1, 7)]); onv = np.array([R[o]["B"]["CORAL_on"][str(c)]["state"]["mae_ratio_mean"] for c in range(1, 7)])
    rel = onv / off - 1; ax.scatter([i] * 6, rel, s=10, color=BLUE, zorder=3, facecolors="white" if o == "breast" else BLUE)
ax.axhline(0, color=GREY, lw=0.6, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in ORGS], fontsize=5.8); ax.tick_params(labelsize=5.6)
ax.set_title("Held-out pseudo-bulk state error,\nrelative change on vs off (per chain)", fontsize=6.4, pad=3)
ax = fig.add_subplot(gs[4, 4:6])
KS = (("auc_pooled", "LR, composition-adjusted", "o"), ("auc_per_domain_foldwise_lr", "LR, mean removed", "s"), ("auc_per_domain_foldwise_rf", "RF, mean removed", "^"))
for i, (o, on) in enumerate(ORGS):
    for m, (k, kn, mk) in enumerate(KS):
        for dx, arm, col in ((-0.08, "CORAL_off", BASE), (0.08, "CORAL_on", BLUE)):
            v = np.median([DG[o]["arms"][arm][str(c)][k] for c in range(1, 7)]); ax.scatter(i + (m - 1) * 0.26 + dx, v, s=12, color=col, marker=mk, zorder=3)
ax.axhline(0.5, color=GREY, lw=0.6, ls="--"); ax.set_ylim(0.35, 1.05); ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in ORGS], fontsize=5.8); ax.tick_params(labelsize=5.6)
ax.set_title("Domain classifier AUROC\n(median of chains)", fontsize=6.4, pad=3)
ax.legend(handles=[Line2D([], [], marker=mk, ls="", color=INK2, ms=3.5, label=kn) for _, kn, mk in KS], fontsize=5.4, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.66))
fig.savefig(OUTD / "S6_domain_gap.pdf", bbox_inches="tight"); fig.savefig(OUTD / "S6_domain_gap.png", dpi=300, bbox_inches="tight"); print("S6 ok")
