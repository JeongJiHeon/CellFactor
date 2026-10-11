#!/usr/bin/env python3
"""원고 v2 Figure 2 — CellFactor identifies prognostic cellular states and preserves within-cell-type transcriptional heterogeneity.
  a–c  기존 Fig. 2a–c (UMAP 세포유형 · 고위험 상위 10% · 고위험 구성)      — fig2_main.py 의 a–c 블록 그대로 실행
  d    기존 Fig. 3a (축 부분집합별 군집 판독 AUROC)                          — fig_program_axis_full.py 의 a 블록
  e    기존 Fig. 3c (CellFactor vs 기준선, 같은 bag 판독기 공정 비교)          — fig_program_axis_full.py 의 c 블록
  f    KM 요약 표(CellFactor 상위 20% vs 나머지 80%, 로그순위 P)              — fig2_main.py 와 같은 함수·자료
재계산 없음. 출력 final_results/figures/manuscript_v2/Fig2.{pdf,png}"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]; P = ROOT / "analysis/report/paper"
OUTD = ROOT / "final_results/figures/manuscript_v2"; OUTD.mkdir(parents=True, exist_ok=True)
g = {"__name__": "v2", "__file__": str(P / "fig2_main.py")}
s2 = open(P / "fig2_main.py").read()
exec(compile(s2[:s2.index("# 가로 배치")], "fig2(head)", "exec"), g)
exec(compile(s2[s2.index("def km("):s2.index("COH = [")], "fig2(km)", "exec"), g)
plt, np, pd = g["plt"], g["np"], g["pd"]
plt.rcParams.update({"font.size": 7.0})


def scale(code, f=1.0, floor=6.0, cap=None):
    """잘라 온 코드의 fontsize 를 v2 판형(7.2 in)에 맞춤: 최소 floor pt."""
    def r(m):
        v = float(m.group(2)) * f; v = max(floor, v) if cap is None else min(cap, max(floor, v))
        return f"{m.group(1)}={v:.1f}"
    return re.sub(r"(fontsize|labelsize)=(\d+(?:\.\d+)?)", r, code)


fig = plt.figure(figsize=(7.2, 9.6))
G = fig.add_gridspec(4, 1, height_ratios=[1.55, 0.78, 0.78, 0.46], hspace=0.72, left=0.1, right=0.985, top=0.975, bottom=0.03)
g.update(fig=fig, gtop=G[0].subgridspec(3, 3, width_ratios=[1, 1, 0.9], hspace=0.18, wspace=0.55), LET={})
blk = s2[s2.index("# ---------- a–c ----------"):s2.index("# ---------- d ----------")]
blk = blk.replace('fontsize=10.1, fontweight="bold"', 'fontsize=8.0, fontweight="bold"').replace("fontsize=9.5", "fontsize=7.6")
exec(compile(scale(blk, 0.82, 5.6), "fig2(a-c)", "exec"), g)
LET = g["LET"]

# ---------- d·e: 프로그램 축 그림의 자료 준비 + a·c 블록 ----------
sp = open(P / "fig_program_axis_full.py").read()
h = {"__name__": "v2", "__file__": str(P / "fig_program_axis_full.py")}
exec(compile(sp[:sp.index("# ---------- 그림 ----------")], "prog(prep)", "exec"), h)
h["plt"].rcParams.update({"font.size": 7.0})
h.update(fig=fig, G=G, jit=lambda n, w=0.05: np.linspace(-w, w, n), tag=lambda *a, **k: None)
SHORT = [("liver", "int", "Liver T cells\ninternal (GSE151530)"), ("pancreas", "int", "Pancreas fibroblasts\ninternal (GSE202051)"), ("breast", "int", "Breast T cells\ninternal (GSE176078)"), ("breast", "ext", "Breast T cells\nexternal (GSE161529)")]
h["GROUPS"] = SHORT
bd = sp[sp.index("# a — E2 시드별 AUROC"):sp.index("# b — E2 판정 지표")]
bd = bd.replace("fig.add_subplot(gs[0, :4])", "fig.add_subplot(G[1])").replace('ax.text(0.0, 1.17, "Axis subsets", transform=ax.transAxes, fontsize=7, color=INK)', "")
exec(compile(scale(bd, 1.25, 6.4), "prog(a)", "exec"), h); axd = h["ax"]
axd.set_title("Within-cell-type subpopulation recovery by axis subset", fontsize=7.6, color=g["INK"], loc="left", pad=20)
be = sp[sp.index("# c — 기준선 모델 비교"):sp.index("# d — E1 ρ 분포")]
be = be.replace("fig.add_subplot(gs[1, :4])", "fig.add_subplot(G[2])").replace('ax.text(0.0, 1.36, "CellFactor vs baselines (same bag readout)", transform=ax.transAxes, fontsize=7, color=INK)', "")
exec(compile(scale(be, 1.25, 6.0), "prog(c)", "exec"), h); axe = h["ax"]
axe.set_title("CellFactor vs comparator representations (same bag readout)", fontsize=7.6, color=g["INK"], loc="left", pad=34)

# ---------- f: KM 요약 표 ----------
COH = [("Breast_SCAN-B_OS", "Breast", "SCAN-B"), ("Breast_METABRIC_OS", "Breast", "METABRIC"), ("Liver_GSE14520_OS", "Liver", "GSE14520"), ("Pancreas_GSE62452_OS", "Pancreas", "GSE62452")]
rows = []
for f, org, coh in COH:
    D = pd.read_csv(g["FR"] / f"04_prognosis/patients/{f}_patients.csv")
    p = g["logrank"](D.time_years.values, D.event.values, D.high_risk_top20.values.astype(int)); rows.append((org, coh, len(D), int(D.event.sum()), p))
ax = fig.add_subplot(G[3]); ax.axis("off")
def fmt_p(p):
    if p >= 0.01: return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e"); return f"{m} × 10$^{{{int(e)}}}$"
cols = ["Cancer", "Cohort (OS)", "n (events)", "Log-rank P"]; xs = [0.02, 0.24, 0.46, 0.70]
for x, c in zip(xs, cols): ax.text(x, 0.92, c, fontsize=7, fontweight="bold", color=g["INK"], transform=ax.transAxes, va="top")
ax.plot([0.0, 1.0], [0.78, 0.78], color=g["INK2"], lw=0.6, transform=ax.transAxes)
for k, (org, coh, n, ev, p) in enumerate(rows):
    y = 0.64 - k * 0.165
    for x, v in zip(xs, [org, coh, f"{n:,} ({ev:,})", fmt_p(p)]):
        ax.text(x, y, v, fontsize=7, color=g["INK"], transform=ax.transAxes, va="top", fontweight="bold" if (v == fmt_p(p) and p < 0.05) else "normal")
ax.text(0.0, -0.06, "CellFactor top 20% vs remaining 80% of risk; two-sided log-rank test. Full Kaplan–Meier curves for all models: Supplementary Fig. S1.",
        fontsize=6.2, color=g["INK2"], transform=ax.transAxes, va="top")
ax.set_title("Survival stratification in external cohorts", fontsize=7.6, color=g["INK"], loc="left", pad=2)

LET.update(d=axd, e=axe, f=ax)
fig.canvas.draw(); R_ = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
for k, a in LET.items():
    t = a._left_title if a._left_title.get_text() else a.title
    y1 = inv.transform(t.get_window_extent(R_).get_points())[1, 1]; p_ = a.get_position()
    x = 0.012 if k in ("a", "d", "e", "f") else p_.x0 - 0.035
    fig.text(x, y1, k, fontsize=10, fontweight="bold", va="top")
fig.savefig(OUTD / "Fig2.pdf", dpi=400); fig.savefig(OUTD / "Fig2.png", dpi=300)
print("Fig2 v2 ok", [(r[1], r[4]) for r in rows])
