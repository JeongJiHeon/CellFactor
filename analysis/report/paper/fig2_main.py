#!/usr/bin/env python3
"""Figure 2 — CellFactor 고위험 세포와 예후 검증(재계산 없음, final_results 저장 산출만 읽음).
  a  단일세포 UMAP(세포 유형) · b 같은 좌표의 고위험 상위 10% 세포 · c 고위험 세포의 유형 구성(고위험 전체 대비 %)
  d  C-index 막대(기본 DSTYLE=bar): CellFactor 가운데 줄, 위 = 기준선 4종, 아래 = 단순 기준선 5종(† 이전 파이프라인); 연한 = held-out, 진한 = 외부 평균, 점선 = CellFactor 외부 평균. DSTYLE=radar 는 비교용 대안
  e  위험 상위 20% vs 나머지 80% KM, 행 = 외부 OS 4코호트 · 열 = 5모델, 칸마다 로그순위 P
원천 final_results/{01_umap/*/cells.csv.gz, cell_type_counts.csv, 02_cindex/cindex_summary_6cells.csv, bootstrap_paired_B2000.csv,
      04_prognosis/patients/*.csv (CellFactor), 04_prognosis/KM_5MODELS_ARRAYS.npz (기준선, dump_km_5models.py)}
출력 final_results/figures/Fig2.{pdf,png}"""
from pathlib import Path
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[3]; FR = ROOT / "final_results"
BLUE, INK, INK2, MUTED, GRID, GREY = "#2a78d6", "#16181b", "#52514e", "#8a8985", "#e6e5e1", "#8a8984"
RED, BGC = "#d1495b", "#cfcdc8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ORG = [("breast", "Breast"), ("liver", "Liver"), ("pancreas", "Pancreas")]
PAL = ["#3b6fb6", "#e07b39", "#3a9a5b", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#5fa2a8", "#b8a03a"]
# 세포 계열 고정 색(장기 공통): 악성·종양 상피 = 빨강, 정상 상피 = 연분홍, 혈관 계열 = 보라, 섬유아 = 주황, T·면역 = 파랑, B = 연파랑, 골수 = 초록
CTCOL = {"Cancer Epithelial": "#c8323c", "Malignant cells": "#c8323c", "Epithelial": "#c8323c", "Normal Epithelial": "#efb0ae",
         "Endothelial": "#8172b3", "TECs": "#8172b3", "PVL": "#b9a7d6", "CAFs": "#e08a3c", "Fibroblast": "#e08a3c",
         "T-cells": "#3b6fb6", "T cells": "#3b6fb6", "Immune": "#3b6fb6", "B-cells": "#86b0dc", "B cells": "#86b0dc", "Plasmablasts": "#5fa2a8",
         "Myeloid": "#3a9a5b", "TAMs": "#3a9a5b", "Endocrine": "#b8a03a", "Schwann": "#937860"}
MODELS = ["CellFactor", "SIDISH", "scSurv", "Scissor", "DEGAS"]; MK = {"CellFactor": "o", "SIDISH": "D", "scSurv": "s", "Scissor": "^", "DEGAS": "v"}

# 가로 배치: 왼쪽 = a–c(위) · d(아래), 오른쪽 = e KM 격자
fig = plt.figure(figsize=(12.8, 7.4))
G = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.92], wspace=0.13, top=0.955, bottom=0.06, left=0.11, right=0.99)
GL = G[0].subgridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.25)
gtop = GL[0].subgridspec(3, 3, width_ratios=[1, 1, 0.9], hspace=0.18, wspace=0.42)
LET = {}

# ---------- a–c ----------
for i, (o, on) in enumerate(ORG):
    C = pd.read_csv(FR / f"01_umap/{o}/cells.csv.gz", usecols=["cell_type", "high_risk_top10", "umap_1", "umap_2"])
    T = pd.read_csv(FR / f"01_umap/{o}/cell_type_counts.csv").sort_values("percent_of_high_risk_cells", ascending=False)
    types = list(T.cell_type); col = {t: CTCOL[t] for t in types}
    rng = np.random.default_rng(0); C = C.iloc[rng.permutation(len(C))]
    ax = fig.add_subplot(gtop[i, 0]); ax.scatter(C.umap_1, C.umap_2, s=0.15, c=C.cell_type.map(col), linewidths=0, rasterized=True)
    # 라벨: 유형 중앙값에 놓고, 렌더 크기로 겹침을 재서 세로로 밀어냄(표시용)
    TX = [ax.text(float(np.median(C.umap_1[C.cell_type == t])), float(np.median(C.umap_2[C.cell_type == t])), t, fontsize=5.8, ha="center", va="center",
                  color=INK, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75)) for t in types if (C.cell_type == t).sum() > 0.01 * len(C)]
    ax.set_xlim(C.umap_1.min(), C.umap_1.max()); ax.set_ylim(C.umap_2.min(), C.umap_2.max())
    rend = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for _ in range(80):
        moved = False; bb = [t_.get_window_extent(rend).expanded(1.05, 1.15) for t_ in TX]
        for u in range(len(TX)):
            for w in range(u + 1, len(TX)):
                if not bb[u].overlaps(bb[w]): continue
                ov = min(bb[u].y1, bb[w].y1) - max(bb[u].y0, bb[w].y0) + 1.0
                up, dn = (u, w) if bb[u].y0 + bb[u].y1 >= bb[w].y0 + bb[w].y1 else (w, u)
                for k_, sg in ((up, 1), (dn, -1)):
                    x_, y_ = TX[k_].get_position(); dpx = ax.transData.transform((x_, y_)) + np.array([0, sg * ov / 2])
                    TX[k_].set_position(tuple(inv.transform(dpx)))
                moved = True; bb = [t_.get_window_extent(rend).expanded(1.05, 1.15) for t_ in TX]
        if not moved: break
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(on, fontsize=10.1, fontweight="bold", color=INK)
    if i == 0: ax.set_title("Cell types", fontsize=9.5, color=INK); LET["a"] = ax
    ax2 = fig.add_subplot(gtop[i, 1]); hi = C.high_risk_top10.astype(bool)
    ax2.scatter(C.umap_1[~hi], C.umap_2[~hi], s=0.15, c=BGC, linewidths=0, rasterized=True)
    ax2.scatter(C.umap_1[hi], C.umap_2[hi], s=0.25, c=RED, linewidths=0, rasterized=True)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(0.02, -0.06, f"high-risk {int(hi.sum()):,} / {len(C):,} cells", transform=ax2.transAxes, fontsize=6.8, color=INK2)
    if i == 0: ax2.set_title("Top 10% CellFactor risk", fontsize=9.5, color=INK); LET["b"] = ax2
    ax3 = fig.add_subplot(gtop[i, 2]); TT = T[T.percent_of_high_risk_cells > 0].copy(); y = np.arange(len(TT))[::-1]
    ax3.barh(y, TT.percent_of_high_risk_cells, color=[col[t] for t in TT.cell_type], height=0.7)
    for yy, v in zip(y, TT.percent_of_high_risk_cells): ax3.text(v + 1.5, yy, f"{v:.0f}%", va="center", fontsize=6.5, color=INK2)
    ax3.set_yticks(y); ax3.set_yticklabels(TT.cell_type, fontsize=6.8); ax3.set_xlim(0, 100); ax3.tick_params(axis="y", length=0)
    ax3.set_xlabel("% of high-risk cells" if i == 2 else "", fontsize=7.8); ax3.tick_params(axis="x", labelsize=6.8, labelbottom=(i == 2))
    if i == 0: ax3.set_title("High-risk composition", fontsize=9.5, color=INK); LET["c"] = ax3
    for a_ in (ax, ax2):
        for sp in a_.spines.values(): sp.set_visible(False)

# ---------- d ----------
# 기준선 4종 + 단순 기준선(02_cindex/simple_baselines). † = 이전(E2″) 파이프라인 팔: 같은 2A·코호트, H·W 는 이전 모델
SIMPLE = [("Bulk Cox", "Bulk Cox"), ("Composition-only Cox", "Composition Cox"), ("Bulk program-score Cox", "Program score Cox †"),
          ("Composition + bulk program score (additive Cox)", "Comp. + score Cox †"), ("State only (composition fixed, W fixed)", "State only †")]
SBC = "#6b6a66"
gbot = [GL[1], G[1]]
gd = gbot[0].subgridspec(1, 3, wspace=0.16)
B = pd.read_csv(FR / "02_cindex/bootstrap_paired_B2000.csv")
SB = pd.read_csv(FR / "02_cindex/simple_baselines/TABLE_simple_baselines.csv"); SBB = pd.read_csv(FR / "02_cindex/simple_baselines/PAIRED_BOOT.csv")
ROWS = [(m, m) for m in MODELS[1:]] + SIMPLE; YS = list(np.r_[np.arange(len(SIMPLE)), np.arange(len(MODELS) - 1) + len(SIMPLE) + 1.0][::-1])


def delta(o, m, scope):
    """CellFactor − 기준선: (차이, CI 하한, CI 상한) — CI 없으면 None"""
    if m in MODELS:
        r = B[(B.organ == o) & (B.baseline == m) & (B.validation == scope)].iloc[0]; return r.diff_allchain, r.ci_lo_allchain, r.ci_hi_allchain
    r = SBB[(SBB.organ == o) & (SBB.comparison == f"CellFactor − {m}") & (SBB.scope == ("mean of external endpoints" if scope == "external" else "held-out OS"))]
    if len(r): return r['diff'].iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0]
    q = SB[SB.organ == o]; row = q[q.endpoint == "Mean of external endpoints"] if scope == "external" else q[q.set == "Held-out"]
    return float(row.CellFactor.iloc[0] - row[m].iloc[0]), None, None


import os
DSTYLE = os.environ.get("DSTYLE", "bar")   # bar | radar
SC = pd.read_csv(FR / "02_cindex/cindex_summary_6cells.csv")


def cval(o, m, scope):
    """절대 C-index: 기준선 4종·CellFactor = cindex_summary_6cells, 단순 기준선 = simple_baselines 표"""
    if m in MODELS: return float(SC[(SC.organ == o) & (SC.validation == scope)][m].iloc[0])
    q = SB[SB.organ == o]; return float((q[q.endpoint == "Mean of external endpoints"] if scope == "external" else q[q.set == "Held-out"])[m].iloc[0])


LIGHT = {BLUE: "#a9c8ee", GREY: "#d9d8d4", SBC: "#bdbcb7"}
if DSTYLE == "bar":
    # CellFactor 를 가운데: 위 = 발표 기준선 4종, 아래 = 단순 기준선 5종
    ROWS_B = [(m, m) for m in MODELS[1:]] + [("CellFactor", "CellFactor")] + SIMPLE
    ns = len(SIMPLE); YCF = ns + 0.6; YB = list(np.r_[np.arange(len(MODELS) - 1) + YCF + 1.6, YCF, np.arange(ns)][::-1][::-1])
    YB = [YCF + 1.6 + (len(MODELS) - 2 - i) for i in range(len(MODELS) - 1)] + [YCF] + [ns - 1 - i for i in range(ns)]
    for j, (o, on) in enumerate(ORG):
        ax = fig.add_subplot(gd[0, j])
        for yy, (m, lab) in zip(YB, ROWS_B):
            c = BLUE if m == "CellFactor" else (GREY if m in MODELS else SBC)
            h, e = cval(o, m, "held-out"), cval(o, m, "external")
            ax.barh(yy + 0.19, h - 0.4, left=0.4, height=0.36, color=LIGHT[c], lw=0)
            ax.barh(yy - 0.19, e - 0.4, left=0.4, height=0.36, color=c, lw=0)
            ax.text(e + 0.004, yy - 0.19, f"{e:.3f}", va="center", fontsize=5.9, color=INK if m == "CellFactor" else INK2, fontweight="bold" if m == "CellFactor" else "normal",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"), zorder=4)
            ax.text(h + 0.004, yy + 0.19, f"{h:.3f}", va="center", fontsize=5.4, color=MUTED, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"), zorder=4)
        ax.axhspan(YCF - 0.55, YCF + 0.55, color=BLUE, alpha=0.07, lw=0)
        ax.axvline(cval(o, "CellFactor", "external"), color=BLUE, lw=0.7, ls=":", zorder=0)
        ax.axhline(YCF + 0.8, color=GRID, lw=0.8); ax.axhline(YCF - 0.8, color=GRID, lw=0.8); ax.axvline(0.5, color=MUTED, lw=0.6, ls="--")
        ax.set_xlim(0.4, 0.8); ax.set_xticks([0.4, 0.5, 0.6, 0.7]); ax.set_ylim(-0.6, max(YB) + 1.3)
        ax.set_yticks(YB); ax.set_yticklabels([l for _, l in ROWS_B] if j == 0 else [], fontsize=8.1); ax.tick_params(axis="y", length=0)
        for tl in ax.get_yticklabels():
            if tl.get_text() == "CellFactor": tl.set_color(BLUE); tl.set_fontweight("bold")
            elif tl.get_text() in [l for _, l in SIMPLE]: tl.set_color(SBC)
        ax.set_title(on, fontsize=9.5, color=INK); ax.set_xlabel("C-index", fontsize=8.1); ax.grid(axis="x", color=GRID, lw=0.4); ax.set_axisbelow(True)
        if j == 0: LET["d"] = ax
        if j == 2:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=LIGHT[GREY], label="Held-out"), Patch(color=GREY, label="External mean")], frameon=True, fontsize=6.8, loc="center", bbox_to_anchor=(0.6, max(YB) + 0.95), bbox_transform=ax.transData, ncol=2, borderpad=0.25,
                      facecolor="white", edgecolor="none", framealpha=1)
else:
    SPOKES = [(o, sc) for o, _ in ORG for sc in ("held-out", "external")]
    SLAB = [f"{on}\n{'held-out' if sc == 'held-out' else 'external'}" for o, on in ORG for sc in ("held-out", "external")]
    ANG = np.linspace(0, 2 * np.pi, len(SPOKES), endpoint=False); ANG_C = np.r_[ANG, ANG[:1]]
    CAT = ["#e07b39", "#3a9a5b", "#8172b3", "#c44e52", "#937860"]
    gr = gbot[0].subgridspec(1, 5, width_ratios=[0.02, 1, 0.45, 1, 0.12], wspace=0)
    for k, (title, mods) in enumerate((("Published baselines", [(m, m) for m in MODELS]), ("Simple baselines", [("CellFactor", "CellFactor")] + SIMPLE))):
        ax = fig.add_subplot(gr[0, 1 + 2 * k], projection="polar"); ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        others = [x for x in mods if x[0] != "CellFactor"]
        for (m, lab) in others + [("CellFactor", "CellFactor")]:
            v = [cval(o, m, sc) for o, sc in SPOKES]; v = np.r_[v, v[:1]]
            c = BLUE if m == "CellFactor" else CAT[[x[0] for x in others].index(m)]
            ax.plot(ANG_C, v, color=c, lw=1.6 if m == "CellFactor" else 0.9, label=lab, zorder=3 if m == "CellFactor" else 2)
            ax.scatter(ANG, v[:-1], s=6 if m != "CellFactor" else 10, color=c, zorder=3)
            if m == "CellFactor": ax.fill(ANG_C, v, color=BLUE, alpha=0.08)
        ax.set_ylim(0.4, 0.76); ax.set_yticks([0.5, 0.6, 0.7]); ax.set_yticklabels(["0.5", "0.6", "0.7"], fontsize=6.2, color=MUTED)
        ax.set_xticks(ANG); ax.set_xticklabels(SLAB, fontsize=7.3, color=INK); ax.tick_params(axis="x", pad=1)
        ax.grid(color=GRID, lw=0.5); ax.spines["polar"].set_color(GRID)
        ax.set_title(title, fontsize=9.5, color=INK, pad=16)
        ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, handlelength=1.2, columnspacing=0.8)
        if k == 0: LET["d"] = ax

# ---------- e ----------
def km(t, e):
    o = np.argsort(t); t, e = t[o], e[o]; n = len(t); S_ = 1.; xs = [0.]; ys = [1.]; cx, cy = [], []
    for i_, (ti, ei) in enumerate(zip(t, e)):
        if ei > 0: S_ *= 1 - 1 / (n - i_); xs.append(ti); ys.append(S_)
        else: cx.append(ti); cy.append(S_)
    xs.append(t[-1]); ys.append(S_); return np.array(xs), np.array(ys), np.array(cx), np.array(cy)


def logrank(t, e, g):
    ts = np.unique(t[e > 0]); O1 = E1 = V = 0.
    for x in ts:
        at = t >= x; n = at.sum(); n1 = (at & (g == 1)).sum(); d = ((t == x) & (e > 0)).sum(); d1 = ((t == x) & (e > 0) & (g == 1)).sum()
        if n < 2: continue
        O1 += d1; E1 += d * n1 / n; V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    from math import erfc, sqrt
    return float(erfc(abs((O1 - E1) / np.sqrt(max(V, 1e-12))) / sqrt(2)))


COH = [("Breast_SCAN-B_OS", "Breast · SCAN-B OS (external)", "Breast · SCAN-B OS"), ("Breast_METABRIC_OS", "Breast · METABRIC OS (external)", "Breast · METABRIC OS"),
       ("Liver_GSE14520_OS", "Liver · GSE14520 OS", "Liver · GSE14520 OS"), ("Pancreas_GSE62452_OS", "Pancreas · GSE62452 OS", "Pancreas · GSE62452 OS")]
Z = np.load(FR / "04_prognosis/KM_5MODELS_ARRAYS.npz"); P = {}
ge = gbot[1].subgridspec(len(COH), len(MODELS), wspace=0.12, hspace=0.30)
for i, (f, key, lab) in enumerate(COH):
    D = pd.read_csv(FR / f"04_prognosis/patients/{f}_patients.csv")
    xmax = np.percentile(D.time_years.values, 98)
    for j, m in enumerate(MODELS):
        if m == "CellFactor": t, e, hi = D.time_years.values, D.event.values, D.high_risk_top20.values.astype(int)
        else:
            r, t, e = Z[f"{key}|{m}|risk"], Z[f"{key}|{m}|time"] / 365.25, Z[f"{key}|{m}|event"]; hi = (r >= np.quantile(r, 0.8)).astype(int)   # npz 시간 = 일
        P[(lab, m)] = p = logrank(t, e, hi)
        ax = fig.add_subplot(ge[i, j])
        for g_, c_, lw, nm in ((0, GREY, 0.9, "Remaining 80%"), (1, RED, 1.1, "Top 20% risk")):
            xs, ys, cx, cy = km(t[hi == g_], e[hi == g_]); ax.step(xs, ys, where="post", color=c_, lw=lw, label=nm)
            ax.plot(cx, cy, "|", color=c_, ms=1.8, mew=0.4)
        ax.text(0.97, 0.95, f"P = {p:.1e}".replace("e-0", "e-") if p < 0.01 else f"P = {p:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.8, color=INK, fontweight="bold" if p < 0.05 else "normal")
        ax.set_ylim(0, 1.17); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xlim(0, xmax); ax.tick_params(labelsize=6.5, length=2)
        if i == 0: ax.set_title(m, fontsize=8.9, color=BLUE if m == "CellFactor" else INK, fontweight="bold" if m == "CellFactor" else "normal")
        if j == 0: ax.set_ylabel(lab.replace(" · ", "\n"), fontsize=7.6, color=INK)
        else: ax.set_yticklabels([])
        if i == len(COH) - 1: ax.set_xlabel("Years", fontsize=7.6)
        if i == 0 and j == 0: LET["e"] = ax
        if i == 0 and j == len(MODELS) - 1: ax.legend(frameon=False, fontsize=6.5, loc="lower left", handlelength=1.0, borderaxespad=0.1)

# ---------- 패널 글자(a·d·e 같은 왼쪽 x) ----------
XL = 0.01
for k, ax in LET.items():
    p_ = ax.get_position(); x = XL if k in ("a", "d") else (p_.x0 - 0.04 if k == "e" else p_.x0 - 0.02)
    fig.text(x, p_.y1 + (0.012 if k != "e" else 0.018), k, fontsize=12.2, fontweight="bold", va="bottom")
NAME = os.environ.get("FIGNAME", "Fig2"); fig.savefig(FR / f"figures/{NAME}.pdf", dpi=400); fig.savefig(FR / f"figures/{NAME}.png", dpi=300)
for (l, m), v in sorted(P.items()): print(f"{l:22s} {m:10s} {v:.3g}")
