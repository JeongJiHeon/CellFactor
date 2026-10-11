#!/usr/bin/env python3
"""프로그램 축 검정 — 사후 민감도 분석: 간세포 전사체 군집 제외(2026-10-06, 결과를 본 뒤 정함 → 사전 판정을 바꾸지 않음).
제외 규칙(세 장기 공통 고정): 군집의 상위 3개 마커(E1 서명, Cohen's d 상위)에 대표 간세포 유전자 HEP 가 하나라도 있으면 제외.
  → 간 41군집 중 17개 제외, 췌장·유방 0개(따라서 췌장·유방 값은 사전 분석과 같다).
재계산 없음: 저장된 군집별 AUROC 를 남은 군집으로 다시 평균. 정의는 summarize_program_axis.py 와 같다
  E2 시드값 = 남은 군집 평균 AUROC, 무작위 명명 4축 = 반복별 군집 평균의 50회 중앙값, 보존율 = (AUC−0.5)/(AUC_all−0.5)
  E1 ρ 중앙값 · 교차 군집 대비 차이의 부호 뒤집기 순열 5000회 p
  E3 시드 짝 차이(A_cand3k − 팔), 시드 bootstrap 2000회 95% CI, 비열등 여유 −0.02
출력 final_results/06_cell_state/program_axis/SENS_HEPEXCL.{json,md}, SENS_HEPEXCL_PER_SEED.csv, figures/S_program_axis_hepexcl.{pdf,png}"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
PA = ROOT / "results_json/report_acand3k/cellaxis/program_axis"; OUT = ROOT / "final_results/06_cell_state/program_axis"; FIG = ROOT / "final_results/figures"
HEP = {"ALB", "APOA1", "APOA2", "APOC3", "APOH", "AHSG", "SERPINA1", "CYP3A4", "TTR", "FGB", "FGA", "HP", "APOB", "TF", "RBP4", "CYP2E1"}
J = lambda p: json.load(open(p)); SEEDS = range(10); RNG = np.random.default_rng(0)
SIDES = {"liver": ["int"], "pancreas": ["int"], "breast": ["int", "ext"]}
e1 = pd.read_csv(OUT / "E1_PER_CLUSTER.csv", dtype={"cluster": str})
e1["excluded"] = e1.markers_top5.str.split(",").str[:3].apply(lambda t: any(g in HEP for g in t))
EXCL = {(r.organ, r.seed, r.cluster) for r in e1.itertuples() if r.excluded}
boot = lambda x: [float(v) for v in np.percentile([np.asarray(x)[RNG.integers(0, len(x), len(x))].mean() for _ in range(2000)], [2.5, 97.5])]


def cmean(p, o, s, side, excl):
    cl = J(p)["clusters"]; v = [d[f"CP-PCH_{side}"][0] for c, d in cl.items() if not (excl and (o, s, c) in EXCL)]
    return float(np.mean(v))


res, rows = {}, []
for o in ("liver", "pancreas", "breast"):
    for side in SIDES[o]:
        tag = o if side == "int" else f"{o}_ext"; res[tag] = {}
        for excl in (False, True):
            per = []
            for s in SEEDS:
                f = cmean(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__RAWGENE_s{s}.json", o, s, side, excl)
                n = cmean(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__MASK_named_s{s}.json", o, s, side, excl)
                fr = cmean(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__MASK_free_s{s}.json", o, s, side, excl)
                rr = [cmean(p, o, s, side, excl) for p in sorted(glob.glob(str(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__MASK_named_rand4_r*_s{s}.json")))]
                assert len(rr) == 50
                arms = {a: cmean(PA / f"SSTAR_LEIDEN_{o}_{a}_s{s}.json", o, s, side, excl) for a in ("B0", "B1_cand3k", "F") if (PA / f"SSTAR_LEIDEN_{o}_{a}_s{s}.json").exists()}
                r4 = float(np.median(rr))
                per.append({"organ": o, "eval_set": side, "clusters": "excl_hepatocyte" if excl else "all", "seed": s, "full": f, "named": n, "free": fr, "named_rand4_median": r4,
                            "retain_named": (n - .5) / (f - .5), "retain_free": (fr - .5) / (f - .5), "rand4_gt_free": r4 > fr, **arms})
            rows += per; P = pd.DataFrame(per)
            e2 = {"retain_named_mean": float(P.retain_named.mean()), "retain_free_mean": float(P.retain_free.mean()), "n_seeds_rand4_gt_free": int(P.rand4_gt_free.sum())}
            e2["pass"] = bool(e2["retain_named_mean"] >= 0.8 and e2["n_seeds_rand4_gt_free"] >= 8)
            e3 = {}
            for a in ("B0", "B1_cand3k", "F"):
                if a in P and P[a].notna().all():
                    d = (P.full - P[a]).values; ci = boot(d)
                    e3[a] = {"A_cand3k": float(P.full.mean()), "arm": float(P[a].mean()), "ci95_diff": ci, "noninferior": bool(ci[0] > -0.02)}
            out = {"E2": e2, "E3": e3}
            if side == "int":   # E1 (내부 학습 도너 서명 기준)
                q = e1[(e1.organ == o) & (~e1.excluded if excl else True)]
                cross = (q.rho - q.rho_crosscluster_mean).dropna().values
                p = float(np.mean([RNG.choice([-1, 1], len(cross)) @ np.abs(cross) >= cross.sum() for _ in range(5000)]))
                out["E1"] = {"n_clusters": int(len(q)), "rho_median": float(q.rho.median()), "p_vs_crosscluster": p, "pass": bool(q.rho.median() > 0 and p < 0.05)}
            res[tag]["excl_hepatocyte" if excl else "all"] = out
res["rule"] = {"hepatocyte_genes": sorted(HEP), "criterion": "exclude cluster if any of its top-3 markers is in hepatocyte_genes",
               "n_excluded": {o: int(e1[(e1.organ == o) & e1.excluded].shape[0]) for o in ("liver", "pancreas", "breast")},
               "n_clusters": {o: int((e1.organ == o).sum()) for o in ("liver", "pancreas", "breast")}, "status": "post hoc sensitivity analysis; pre-registered verdicts unchanged"}
# 항등: '전체 군집' 행은 사전 분석 SUMMARY 와 같아야 함
SM = J(PA / "SUMMARY.json")
for tag in ("liver", "pancreas", "breast", "breast_ext"):
    assert res[tag]["all"]["E2"]["n_seeds_rand4_gt_free"] == SM["E2"][tag]["n_seeds_rand4_gt_free"], tag
    assert abs(res[tag]["all"]["E2"]["retain_named_mean"] - SM["E2"][tag]["retain_named_mean"]) < 1e-9, tag
pd.DataFrame(rows).to_csv(OUT / "SENS_HEPEXCL_PER_SEED.csv", index=False)
json.dump(res, open(OUT / "SENS_HEPEXCL.json", "w"), indent=1)
L = res["liver"]

# ---------- 요약 문서 ----------
def line(k):
    a, b = L["all"], L["excl_hepatocyte"]
    return a, b
a, b = L["all"], L["excl_hepatocyte"]
md = f"""# 사후 민감도 분석 — 간세포 전사체 군집 제외

**사후 분석**입니다. 간 결과를 본 뒤 정했으므로 사전 판정(간 E2 미통과)을 바꾸지 않습니다. 원인 진단과 민감도로만 씁니다.

- 규칙(세 장기 공통): 군집 상위 3개 마커에 대표 간세포 유전자({", ".join(sorted(HEP))})가 하나라도 있으면 제외.
- 제외 군집: 간 {res['rule']['n_excluded']['liver']}/{res['rule']['n_clusters']['liver']}, 췌장 0/{res['rule']['n_clusters']['pancreas']}, 유방 0/{res['rule']['n_clusters']['breast']}. 췌장·유방 값은 사전 분석과 같습니다.
- 제외된 간 군집은 ALB·APOC3·APOA1 군집(7시드)과 CYP3A4·GLUL·AHSG 군집(10시드)입니다.

## 간 T세포

| 지표 | 전체 군집 (사전 분석) | 간세포 군집 제외 |
|---|---|---|
| 군집 수 (10시드 합) | {a['E1']['n_clusters']} | {b['E1']['n_clusters']} |
| E2 명명 축 보존율 | {a['E2']['retain_named_mean']:.2f} | {b['E2']['retain_named_mean']:.2f} |
| E2 자유 4축 보존율 | {a['E2']['retain_free_mean']:.2f} | {b['E2']['retain_free_mean']:.2f} |
| E2 무작위 명명 4축 > 자유 4축 | {a['E2']['n_seeds_rand4_gt_free']}/10 | {b['E2']['n_seeds_rand4_gt_free']}/10 |
| E2 판정 | {'통과' if a['E2']['pass'] else '미통과'} | {'통과' if b['E2']['pass'] else '미통과'} |
| E1 ρ 중앙값 | {a['E1']['rho_median']:.2f} | {b['E1']['rho_median']:.2f} |
| E1 교차 군집 대비 p | {a['E1']['p_vs_crosscluster']:.4f} | {b['E1']['p_vs_crosscluster']:.4f} |
""" + "".join(f"| E3 AUROC CellFactor / {k} | {a['E3'][k]['A_cand3k']:.3f} / {a['E3'][k]['arm']:.3f} ({'비열등' if a['E3'][k]['noninferior'] else '비열등 아님'}) | "
              f"{b['E3'][k]['A_cand3k']:.3f} / {b['E3'][k]['arm']:.3f} ({'비열등' if b['E3'][k]['noninferior'] else '비열등 아님'}) |\n" for k in ("B0", "B1_cand3k"))
open(OUT / "SENS_HEPEXCL.md", "w").write(md)

# ---------- 그림 ----------
BLUE, INK, INK2, MUTED, GRID = "#2a78d6", "#16181b", "#52514e", "#8a8985", "#e6e5e1"; NAMED, FREE, RAND = "#1d4f91", "#e08a2c", "#9a9994"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
R = pd.DataFrame(rows); LV = R[(R.organ == "liver")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.55})
ax = axs[0]
for j, cl in enumerate(("all", "excl_hepatocyte")):
    q = LV[LV.clusters == cl].sort_values("seed")
    for k, col, m, dx in (("free", FREE, "D", -0.12), ("named_rand4_median", RAND, "^", 0.12)):
        ax.scatter(j + dx + np.linspace(-0.04, 0.04, 10), q[k], s=8, marker=m, color=col, alpha=0.75, edgecolor="none", zorder=3)
        ax.hlines(q[k].mean(), j + dx - 0.07, j + dx + 0.07, color=col, lw=1.6, zorder=4)
    for s in range(10):
        r = q[q.seed == s].iloc[0]; ax.plot([j - 0.12, j + 0.12], [r.free, r.named_rand4_median], color=GRID if r.named_rand4_median > r.free else "#d9a5a0", lw=0.6, zorder=2)
    e = res["liver"][cl]["E2"]
    ax.text(j, 0.775, f"rand-4 > free\n{e['n_seeds_rand4_gt_free']}/10 · {'pass' if e['pass'] else 'fail'}", ha="center", fontsize=5.6,
            color=INK if e["pass"] else "#b23b2e", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels([f"All clusters\nn = {a['E1']['n_clusters']}\npre-registered", f"Hepatocyte clusters\nexcluded, n = {b['E1']['n_clusters']}\npost hoc"])
ax.set_ylim(0.5, 0.82); ax.set_xlim(-0.5, 1.5); ax.set_ylabel("Seed AUROC (liver T cells)")
ax.legend(handles=[Line2D([], [], marker="D", color=FREE, ls="none", markersize=3.5, label="Free only (4)"),
                   Line2D([], [], marker="^", color=RAND, ls="none", markersize=3.5, label="Random named 4 (median of 50)")],
          frameon=False, fontsize=5.4, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.text(-0.22, 1.1, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
ax = axs[1]
q = e1[e1.organ == "liver"]
for j, cl in enumerate(("all", "excl_hepatocyte")):
    qq = q if cl == "all" else q[~q.excluded]
    ax.scatter(j + np.linspace(-0.12, 0.12, len(qq)), qq.rho, s=5, color=[("#c96f66" if x else BLUE) for x in qq.excluded], alpha=0.7, edgecolor="none")
    ax.hlines(qq.rho.median(), j - 0.2, j + 0.2, color=INK, lw=1.4); ax.text(j, 0.98, f"ρ̃ {qq.rho.median():.2f}", ha="center", fontsize=5.8)
ax.axhline(0, color=MUTED, lw=0.6, ls="--"); ax.set_xticks([0, 1]); ax.set_xticklabels(["All", "Excluded\nhepatocyte"]); ax.set_ylim(-0.3, 1.08)
ax.set_ylabel("E1 Spearman ρ per cluster"); pass
ax.text(-0.3, 1.1, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
ax = axs[2]
for j, cl in enumerate(("all", "excl_hepatocyte")):
    E = res["liver"][cl]["E3"]; vals = [E["B0"]["A_cand3k"]] + [E[k]["arm"] for k in ("B0", "B1_cand3k")]
    for i, (v, c) in enumerate(zip(vals, [BLUE, "#3f3e3b", "#8a8984"])):
        ax.bar(j + (i - 1) * 0.24, v - 0.5, bottom=0.5, width=0.22, color=c)
        ax.text(j + (i - 1) * 0.24, v + 0.005, f"{v:.2f}", ha="center", fontsize=4.8, va="bottom")
ax.set_xticks([0, 1]); ax.set_xticklabels(["All", "Excluded\nhepatocyte"]); ax.set_ylim(0.5, 0.9); ax.set_ylabel("E3 AUROC")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in [BLUE, "#3f3e3b", "#8a8984"]], labels=["CellFactor", "B0", "B1"],
          frameon=False, fontsize=5.2, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=0.8, columnspacing=0.5)
ax.text(-0.3, 1.1, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
fig.savefig(FIG / "S_program_axis_hepexcl.pdf", bbox_inches="tight"); fig.savefig(FIG / "S_program_axis_hepexcl.png", dpi=300, bbox_inches="tight")
print(md)
