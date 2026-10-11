#!/usr/bin/env python3
"""원고 v2 보충 그림 — 기존 패널 코드를 그대로 실행해 배치만 바꾼다(재계산 없음).
  S1_KM_all            기존 Fig. 2e: 5모델 × 외부 4코호트 KM(상위 20% vs 나머지 80%)
  S2_representation    기존 Fig. 3b(명명 vs 자유 축 보존) · 3d(판독 방향 ρ) · 3e–g(군집 예시)
  S3_objective_ablation 기존 Fig. 3h–i(H 학습 손실 절제: AUROC · 판독 방향)
출력 final_results/figures/manuscript_v2/Supp/*.{pdf,png}"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]; P = ROOT / "analysis/report/paper"
OUTD = ROOT / "final_results/figures/manuscript_v2/Supp"; OUTD.mkdir(parents=True, exist_ok=True)


def scale(code, f=1.0, floor=6.0):
    return re.sub(r"(fontsize|labelsize)=(\d+(?:\.\d+)?)", lambda m: f"{m.group(1)}={max(floor, float(m.group(2)) * f):.1f}", code)


def letters(fig, LET, xs=None):
    fig.canvas.draw(); R_ = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for k, a in LET.items():
        t = a._left_title if a._left_title.get_text() else a.title
        bb = t.get_window_extent(R_) if t.get_text() else a.get_window_extent(R_)
        y1 = inv.transform(bb.get_points())[1, 1]; x = (xs or {}).get(k, a.get_position().x0 - 0.07)
        fig.text(x, y1, k, fontsize=10, fontweight="bold", va="top")


def save(fig, name):
    fig.savefig(OUTD / f"{name}.pdf", dpi=400); fig.savefig(OUTD / f"{name}.png", dpi=300); print(name, "ok")


# ═════ S_KM ═════
s2 = open(P / "fig2_main.py").read(); g = {"__name__": "v2", "__file__": str(P / "fig2_main.py")}
exec(compile(s2[:s2.index("# 가로 배치")], "fig2(head)", "exec"), g)
plt, np, pd = g["plt"], g["np"], g["pd"]
fig = plt.figure(figsize=(7.2, 7.6)); G = fig.add_gridspec(1, 1, left=0.11, right=0.985, top=0.95, bottom=0.07)
g.update(fig=fig, gbot=[None, G[0]], LET={})
blk = s2[s2.index("def km("):s2.index("# ---------- 패널 글자")]
exec(compile(scale(blk, 0.85, 5.8), "fig2(e)", "exec"), g)
save(fig, "S1_KM_all")

# ═════ 프로그램 축 그림(기존 Fig. 3) ═════
sp = open(P / "fig_program_axis_full.py").read(); h = {"__name__": "v2", "__file__": str(P / "fig_program_axis_full.py")}
exec(compile(sp[:sp.index("# ---------- 그림 ----------")], "prog(prep)", "exec"), h)
h["plt"].rcParams.update({"font.size": 7.0}); jit = lambda n, w=0.05: np.linspace(-w, w, n)
part = lambda a, b: sp[sp.index(a):sp.index(b)]

# S_representation: b · d · e–g
fig = plt.figure(figsize=(7.2, 6.6)); gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.95], hspace=0.85, wspace=1.45, left=0.17, right=0.97, top=0.86, bottom=0.08)
h.update(fig=fig, gs=gs, jit=jit, tag=lambda ax, letter, ttl, x=-0.1, y=1.12: ax.set_title(ttl, loc="left", fontsize=7, color=h["INK"], pad=5) if ttl else None)
LET = {}
cb = part("# b — E2 판정 지표", "# c — 기준선 모델 비교").replace("fig.add_subplot(gs[0, 4:])", "fig.add_subplot(gs[0, :3])")
cb = cb.replace('ax.text(1.42, len(BK) - 0.55, "rand-4 >\\nfree", ha="center", va="bottom", fontsize=5.0, color=INK2)', 'ax.text(1.42, len(BK) - 0.55, "random 4 named\\n> free (seeds)", ha="center", va="bottom", fontsize=5.0, color=INK2)')
cb = cb.replace('ax.text(0.5, 1.17, "Named vs free axes", transform=ax.transAxes, fontsize=7, color=INK, ha="center")', 'ax.set_title("Named vs free axes", fontsize=7.4, color=INK, loc="left", pad=16)')
exec(compile(scale(cb, 1.2, 6.0), "prog(b)", "exec"), h); LET["a"] = h["ax"]; h["ax"].set_xlim(-0.35, 1.75)
cd = part("# d — E1 ρ 분포", "# e–g — E1 예시").replace("fig.add_subplot(gs[1, 4:])", "fig.add_subplot(gs[0, 3:])")
cd = cd.replace('ax.text(0.5, 1.27, "Readout direction", transform=ax.transAxes, fontsize=7, color=INK, ha="center")', 'ax.set_title("Readout direction vs program change", fontsize=7.4, color=INK, loc="left", pad=34)')
exec(compile(scale(cd, 1.2, 6.0), "prog(d)", "exec"), h); LET["b"] = h["ax"]
ce = part("# e–g — E1 예시", "# h–i — E4").replace("fig.add_subplot(gs[2, 2 * i:2 * i + 2])", "fig.add_subplot(gs[1, 2 * i:2 * i + 2])")
ce = ce.replace('ORGL = {"liver": "Liver T", "pancreas": "Pancreas fibroblast", "breast": "Breast T"}', 'ORGL = {"liver": "Liver T", "pancreas": "Pancreas fib.", "breast": "Breast T"}')
ce = ce.replace('tag(ax, "efg"[i], ', 'ax.set_title(').replace('\\nρ = {r.rho:.2f}", x=-0.2)', '\\nρ = {r.rho:.2f} (best-case cluster)", loc="left", fontsize=6.2, color=INK)')
exec(compile(scale(ce, 1.2, 6.0), "prog(e-g)", "exec"), h)
for k, ax_ in zip("cde", fig.axes[-3:]): LET[k] = ax_
letters(fig, LET, {"a": 0.01, "c": 0.01, "d": LET["d"].get_position().x0 - 0.06, "e": LET["e"].get_position().x0 - 0.06})
save(fig, "S2_representation")

# S_objective_ablation: h · i
fig = plt.figure(figsize=(7.2, 3.3)); gs = fig.add_gridspec(1, 6, wspace=1.45, left=0.11, right=0.97, top=0.78, bottom=0.24); h.update(fig=fig, gs=gs)
ch = part("# h–i — E4", "fig.savefig(FIG").replace("fig.add_subplot(gs[3, :4])", "fig.add_subplot(gs[0, :4])").replace("fig.add_subplot(gs[3, 4:])", "fig.add_subplot(gs[0, 4:])")
ch = ch.replace('ax.text(0.0, 1.2, "H loss ablation", transform=ax.transAxes, fontsize=7, color=INK)', 'ax.set_title("Training-objective ablation (state estimator H)", fontsize=7.4, color=INK, loc="left", pad=18)')
ch = ch.replace('ax.text(0.5, 1.2, "Readout direction", transform=ax.transAxes, fontsize=7, color=INK, ha="center")', 'ax.set_title("Readout direction", fontsize=7.4, color=INK, loc="left", pad=18)')
ch = ch.replace('tag(ax, "h", "", x=-0.1, y=1.2)', "").replace('tag(ax, "i", "", x=-0.42, y=1.2)', "")
exec(compile(scale(ch, 1.2, 6.0), "prog(h-i)", "exec"), h)
letters(fig, {"a": fig.axes[0], "b": fig.axes[1]}, {"a": 0.01})
save(fig, "S3_objective_ablation")

# (기존 Fig. 5a·b·d 는 본문 Fig. 4 로, 5e 는 supp_s4_s5.py 의 S4 로 옮김)
