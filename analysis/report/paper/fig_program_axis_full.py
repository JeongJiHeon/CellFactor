#!/usr/bin/env python3
"""세포 상태 판독의 프로그램 축 검정(PLAN_PROGRAM_AXIS E1·E2·E3) — 전체 실행(10시드 · 무작위 50회) 그림. 재계산 없음(저장 산출만 읽음).
  a  E2 시드별 AUROC: 전체 축 · 명명 축만 · 자유 4축만 · 무작위 명명 4축(50회 중앙값) — 간 T · 췌장 섬유아세포 · 유방 T 내부/외부
  b  E2 판정 지표: 명명 축 보존율 · 무작위 명명 4축 > 자유 4축 시드 수
  c  E3 같은 레시피 잠재공간 비교(A_cand3k · B0 · B1_cand3k · F) 시드별 AUROC + 원시 유전자 참고선
  d  E1 군집별 Spearman ρ(판독 기여 W×SD(z′) vs 모델 독립 프로그램 Δ) 분포와 교차 군집 귀무 평균
  e–g E1 장기별 예시(ρ 최대 군집): 기여 vs Δ 산점도
원천 results_json/report_acand3k/cellaxis/program_axis/{SUMMARY.json, SSTAR_LEIDEN_*.json, W_*.pt, signature_*.json}
출력 final_results/06_cell_state/program_axis/{E2_PER_SEED.csv, E3_PER_SEED.csv, E1_PER_CLUSTER.csv, E1_EXAMPLES_CONTRIB_VS_DELTA.csv, SUMMARY.md}
     final_results/figures/S_program_axis.{pdf,png}"""
import os, json, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
PA = ROOT / "results_json/report_acand3k/cellaxis/program_axis"
OUT = ROOT / "final_results/06_cell_state/program_axis"; FIG = ROOT / "final_results/figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, INK, INK2, MUTED, GRID = "#2a78d6", "#16181b", "#52514e", "#8a8985", "#e6e5e1"
NAMED, FREE, RAND = "#1d4f91", "#e08a2c", "#9a9994"
ARMC = {"A_cand3k": BLUE, "B0": "#3f3e3b", "B1_cand3k": "#8a8984", "F": "#c4c2bc"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
J = lambda p: json.load(open(p))
SM = J(PA / "SUMMARY.json"); SEEDS = range(10)
GROUPS = [("liver", "int", "Liver T cells\ninternal (GSE151530)"), ("pancreas", "int", "Pancreas fibroblasts\ninternal (GSE202051)"), ("breast", "int", "Breast T cells\ninternal (GSE176078)"),
          ("breast", "ext", "Breast T cells\nexternal (GSE161529)")]
key = lambda side: "CP-PCH_int" if side == "int" else "CP-PCH_ext"
mean_of = lambda p, side: J(p)[key(side) + "_mean"][0]

# ---------- E2 ----------
e2 = []
for o, side, lab in GROUPS:
    for r in SM["E2"][o if side == "int" else f"{o}_ext"]["per_seed"]:
        e2.append({"organ": o, "eval_set": side, **r})
e2 = pd.DataFrame(e2); e2.to_csv(OUT / "E2_PER_SEED.csv", index=False)

# ---------- E3 ----------
e3 = []
for o, side, lab in GROUPS:
    for s in SEEDS:
        fa = PA / f"SSTAR_LEIDEN_{o}_A_cand3k__RAWGENE_s{s}.json"; d = J(fa)
        row = {"organ": o, "eval_set": side, "seed": s, "A_cand3k": d[key(side) + "_mean"][0], "rawgene_reference": d[f"rawgene_{side}_mean"][0]}
        for a in ("B0", "B1_cand3k", "F"):
            p = PA / f"SSTAR_LEIDEN_{o}_{a}_s{s}.json"; row[a] = mean_of(p, side) if p.exists() else np.nan
        e3.append(row)
e3 = pd.DataFrame(e3); e3.to_csv(OUT / "E3_PER_SEED.csv", index=False)

# ---------- E1 (군집별 ρ · 교차 군집 평균, SUMMARY 와 같은 정의) ----------
e1, pts = [], {}
for o in ("liver", "pancreas", "breast"):
    for s in SEEDS:
        W = torch.load(PA / f"W_{o}_A_cand3k_s{s}.pt", map_location="cpu", weights_only=False); sig = J(PA / f"signature_{o}_s{s}.json")
        axes = sorted(set.intersection(*[{n for n in nm if not n.startswith(("free", "dim"))} for nm in W["axis_names"]]) & set(sig["programs"]))
        contrib = {}
        for c, v in W["clusters"].items():
            pc = [np.array([(np.mean([f["W_focus"] for f in fits], 0) * np.asarray(W["z_sd_train_focus"][ch]))[W["axis_names"][ch].index(a)] for a in axes])
                  for ch, fits in enumerate(v["fits"])]
            contrib[c] = np.mean(pc, 0)
        cl = [c for c in contrib if c in sig["clusters"]]
        for c in cl:
            dv = np.array([sig["clusters"][c]["delta"][a] for a in axes]); wv = contrib[c]
            rho = spearmanr(wv, dv).correlation
            oth = [spearmanr(wv, np.array([sig["clusters"][c2]["delta"][a] for a in axes])).correlation for c2 in cl if c2 != c]
            e1.append({"organ": o, "seed": s, "cluster": c, "rho": rho, "rho_crosscluster_mean": np.mean(oth) if oth else np.nan,
                       "markers_top5": ",".join(sig["clusters"][c]["markers"][:5]), "top3_readout": ",".join(axes[i] for i in np.argsort(-wv)[:3]),
                       "top5_delta": ",".join(axes[i] for i in np.argsort(-dv)[:5])})
            pts[(o, s, c)] = (axes, wv, dv)
e1 = pd.DataFrame(e1); e1.to_csv(OUT / "E1_PER_CLUSTER.csv", index=False)
for o in ("liver", "pancreas", "breast"):   # 항등: SUMMARY 의 ρ 중앙값과 같아야 함
    assert abs(e1[e1.organ == o].rho.median() - SM["E1"][o]["rho_median"]) < 1e-9, o
# ---------- 그림용: 간은 간세포 전사체 군집 제외판(사후 민감도, program_axis_hepexcl.py) — CSV 는 위에서 사전 분석(전체 군집)으로 저장됨 ----------
SENS = J(OUT / "SENS_HEPEXCL.json"); SP = pd.read_csv(OUT / "SENS_HEPEXCL_PER_SEED.csv")
HEP = set(SENS["rule"]["hepatocyte_genes"])
e1["excluded"] = (e1.organ == "liver") & e1.markers_top5.str.split(",").str[:3].apply(lambda t: any(g in HEP for g in t))
assert int(e1.excluded.sum()) == SENS["rule"]["n_excluded"]["liver"]
# 간 군집 범위(2026-10-10 결정): 주 그림 = 사전 지정 전체 군집(all). env LIVER_CLUSTERS=excl 이면 예전 사후 제외판.
LIVER_CLUSTERS = os.environ.get("LIVER_CLUSTERS", "all"); assert LIVER_CLUSTERS in ("all", "excl")
LX = SP[(SP.organ == "liver") & (SP.clusters == "excl_hepatocyte")].set_index("seed")
EXL = {(r.seed, r.cluster) for r in e1[e1.excluded].itertuples()} if LIVER_CLUSTERS == "excl" else set()
for i in e3.index[e3.organ == "liver"]:
    s = int(e3.at[i, "seed"])
    if LIVER_CLUSTERS == "excl":
        e3.at[i, "A_cand3k"] = LX.at[s, "full"]
        for a in ("B0", "B1_cand3k", "F"): e3.at[i, a] = LX.at[s, a]
    cl = J(PA / f"SSTAR_LEIDEN_liver_A_cand3k__RAWGENE_s{s}.json")["clusters"]
    e3.at[i, "rawgene_reference"] = float(np.mean([v["rawgene_int"] for c, v in cl.items() if (s, c) not in EXL]))
if LIVER_CLUSTERS == "excl":
    for k in ("full", "named", "free", "named_rand4_median"):
        e2.loc[e2.organ == "liver", k] = e2.loc[e2.organ == "liver", "seed"].map(LX[k]).values
    e1 = e1[~e1.excluded]
LSTAR = "*" if LIVER_CLUSTERS == "excl" else ""
LK4 = "liver_exclhep" if LIVER_CLUSTERS == "excl" else "liver"
BLN = ["scSurv", "SIDISH", "Scissor", "DEGAS"]; blm = []
for o in ("liver", "pancreas"):
    for s in SEEDS:
        cl = J(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__RAWGENE_s{s}.json")["clusters"]
        keep = [c for c in cl if (s, c) not in EXL] if o == "liver" else list(cl)
        for m in ["CP-PCH"] + BLN + ["rawgene"]:
            k = "rawgene_int" if m == "rawgene" else f"{m}_int"
            blm.append({"organ": o, "eval_set": "int", "seed": s, "model": "CellFactor" if m == "CP-PCH" else m,
                        "auroc": float(np.mean([(cl[c][k][0] if isinstance(cl[c][k], list) else cl[c][k]) for c in keep]))})
PB = pd.read_csv(ROOT / "final_results/06_cell_state/LEIDEN_PER_SEED_breast.csv")
for side in ("int", "ext"):
    for r in PB.itertuples():
        for m in ["CellFactor"] + BLN:
            blm.append({"organ": "breast", "eval_set": side, "seed": int(r.seed), "model": m, "auroc": float(getattr(r, f"{m}_{side}"))})
        blm.append({"organ": "breast", "eval_set": side, "seed": int(r.seed), "model": "rawgene",
                    "auroc": float(J(PA / f"SSTAR_LEIDEN_breast_A_cand3k__RAWGENE_s{int(r.seed)}.json")[f"rawgene_{side}_mean"][0])})
BLM = pd.DataFrame(blm)
for o, side in (("pancreas", "int"), ("breast", "int"), ("breast", "ext")):   # 항등: CellFactor = E2 전체 축 값
    a_ = BLM[(BLM.organ == o) & (BLM.eval_set == side) & (BLM.model == "CellFactor")].sort_values("seed").auroc.values
    b_ = e2[(e2.organ == o) & (e2.eval_set == side)].sort_values("seed").full.values; assert np.allclose(a_, b_, atol=1e-9), (o, side)
E1N = {o: (int((e1.organ == o).sum()), float(e1[e1.organ == o].rho.median())) for o in ("liver", "pancreas", "breast")}
EX = {o: e1[e1.organ == o].sort_values("rho", ascending=False).iloc[0] for o in ("liver", "pancreas", "breast")}
ex_rows = []
for o, r in EX.items():
    axes, wv, dv = pts[(o, r.seed, r.cluster)]
    ex_rows += [{"organ": o, "seed": r.seed, "cluster": r.cluster, "axis": a, "readout_contrib": x, "delta_sd": y} for a, x, y in zip(axes, wv, dv)]
pd.DataFrame(ex_rows).to_csv(OUT / "E1_EXAMPLES_CONTRIB_VS_DELTA.csv", index=False)
shutil.copy(PA / "SUMMARY.md", OUT / "SUMMARY.md")

# ---------- 그림 ----------
fig = plt.figure(figsize=(7.2, 9.9))
gs = fig.add_gridspec(4, 6, height_ratios=[1.05, 1.0, 0.95, 0.95], hspace=1.0, wspace=1.45)


def tag(ax, letter, ttl, x=-0.1, y=1.12):
    ax.set_title(ttl, loc="left", fontsize=7, color=INK, pad=5)
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


jit = lambda n, w=0.05: np.linspace(-w, w, n)

# a — E2 시드별 AUROC
ax = fig.add_subplot(gs[0, :4])
STY = [("full", BLUE, "o", "All axes"), ("named", NAMED, "s", "Named only"), ("free", FREE, "D", "Free only (4)"),
       ("named_rand4_median", RAND, "^", "Random named 4")]
off = np.linspace(-0.3, 0.3, len(STY))
for g, (o, side, lab) in enumerate(GROUPS):
    q = e2[(e2.organ == o) & (e2.eval_set == side)]
    for (k, col, m, _), dx in zip(STY, off):
        ax.scatter(g + dx + jit(len(q)), q[k], s=7, marker=m, color=col, alpha=0.55, edgecolor="none", zorder=2)
        ax.hlines(q[k].mean(), g + dx - 0.07, g + dx + 0.07, color=col, lw=1.6, zorder=3)
for i in range(len(GROUPS) - 1): ax.axvline(i + 0.5, color=GRID, lw=0.6)
ax.axhline(0.5, color=MUTED, lw=0.6, ls="--", zorder=1)
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([g[2] for g in GROUPS]); ax.set_xlim(-0.5, len(GROUPS) - 0.5)
ax.set_ylim(0.4, 0.9); ax.set_ylabel("Cluster AUROC (held-out cells)")
ax.legend(handles=[Line2D([], [], marker=m, color=c, ls="none", markersize=4, label=l) for _, c, m, l in STY], frameon=False, fontsize=5.6,
          ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), handletextpad=0.2, columnspacing=0.9)
tag(ax, "a", "", x=-0.1, y=1.17)
ax.text(0.0, 1.17, "Axis subsets", transform=ax.transAxes, fontsize=7, color=INK)

# b — E2 판정 지표
ax = fig.add_subplot(gs[0, 4:])
lab_b = ["Liver T cells\ninternal, all clusters\n(GSE151530)", "Liver T cells, excl.\nhepatocyte-marker\nclusters (post hoc)", "Pancreas fibroblasts\ninternal\n(GSE202051)", "Breast T cells\ninternal\n(GSE176078)", "Breast T cells\nexternal\n(GSE161529)"]
BK = [SENS["liver"]["all"]["E2"], SENS["liver"]["excl_hepatocyte"]["E2"], SM["E2"]["pancreas"], SM["E2"]["breast"], SM["E2"]["breast_ext"]]
ret_n = [b["retain_named_mean"] for b in BK]; ret_f = [b["retain_free_mean"] for b in BK]
wins = [b["n_seeds_rand4_gt_free"] for b in BK]; passes = [b["pass"] for b in BK]
y = np.arange(len(BK))[::-1]
ax.barh(y + 0.17, ret_n, height=0.32, color=NAMED, zorder=2, label="Named only")
ax.barh(y - 0.17, ret_f, height=0.32, color=FREE, zorder=2, label="Free only (4)")
ax.axvline(0.8, color=MUTED, lw=0.6, ls="--"); ax.axvline(0, color=INK2, lw=0.5); ax.axvline(1, color=BLUE, lw=0.6)
for yy, w, p in zip(y, wins, passes):
    ax.text(1.42, yy, f"{w}/10", va="center", ha="center", fontsize=5.6, color=INK, fontweight="bold")
ax.text(1.42, len(BK) - 0.55, "rand-4 >\nfree", ha="center", va="bottom", fontsize=5.0, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(lab_b); ax.set_xlim(-0.35, 1.62); ax.set_xticks([0, 0.4, 0.8, 1.2])
ax.set_xlabel("Retention = (AUC − 0.5)/(AUC$_{all}$ − 0.5)")
ax.legend(frameon=False, fontsize=5.4, loc="lower left", bbox_to_anchor=(-0.12, 1.0), ncol=2, handlelength=0.8, columnspacing=0.4, handletextpad=0.3)
tag(ax, "b", "", x=-0.45, y=1.17); ax.text(0.5, 1.17, "Named vs free axes", transform=ax.transAxes, fontsize=7, color=INK, ha="center")

# c — 기준선 모델 비교(같은 평가 세포·군집; 기준선 = 판독기 재학습 없는 세포 위험 점수)
ax = fig.add_subplot(gs[1, :4])
# E5 공정 비교(program_axis_e5.py): 같은 bag 판독기 — SIDISH·DEGAS 은닉층, scSurv·Scissor 는 1차원 점수 상한. 간은 간세포 군집 제외판
E5 = pd.read_csv(OUT / "E5_PER_SEED.csv"); _e5o = {("liver", "int"): ("liver", "excl_hepatocyte" if LIVER_CLUSTERS == "excl" else "all"), ("pancreas", "int"): ("pancreas", "all"),
                                                   ("breast", "int"): ("breast_int", "all"), ("breast", "ext"): ("breast_ext", "all")}
_m5 = {"CellFactor": "readout|CellFactor", "SIDISH": "readout|SIDISH (hidden)", "DEGAS": "readout|DEGAS (hidden)", "scSurv": "readout|scSurv (1-D bound)", "Scissor": "readout|Scissor (1-D bound)"}
rows5 = []
for (o, side), (oo, cc) in _e5o.items():
    q5 = E5[(E5.organ == oo) & (E5.clusters == cc)]
    for m, col_ in _m5.items():
        for i5 in q5.index: rows5.append({"organ": o, "eval_set": side, "seed": int(q5.at[i5, "seed"]), "model": m, "auroc": float(q5.at[i5, col_])})
    rows5 += [r for r in BLM[(BLM.organ == o) & (BLM.eval_set == side) & (BLM.model == "rawgene")].to_dict("records")]
BLM = pd.DataFrame(rows5)
MOD = [("CellFactor", BLUE), ("SIDISH", "#3f3e3b"), ("DEGAS", "#6f6e6a"), ("scSurv", "#9e9d98"), ("Scissor", "#c9c7c1")]
off = np.linspace(-0.32, 0.32, len(MOD))
for g, (o, side, lab) in enumerate(GROUPS):
    q = BLM[(BLM.organ == o) & (BLM.eval_set == side)]
    for (m, col), dx in zip(MOD, off):
        v = q[q.model == m].auroc
        ax.bar(g + dx, v.mean() - 0.3, bottom=0.3, width=0.15, color=col, zorder=2)
        ax.scatter(g + dx + jit(len(v), 0.03), v, s=3, color=INK, alpha=0.55, edgecolor="none", zorder=3)
        ax.text(g + dx, v.max() + 0.008, f"{v.mean():.2f}", ha="center", fontsize=4.4, color=INK, va="bottom")
    rg = q[q.model == "rawgene"].auroc.mean(); ax.hlines(rg, g - 0.42, g + 0.42, color=INK, lw=0.9, ls=(0, (2, 1.2)), zorder=4)
for i in range(len(GROUPS) - 1): ax.axvline(i + 0.5, color=GRID, lw=0.6)
ax.axhline(0.5, color=MUTED, lw=0.6, ls="--", zorder=1)
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([g[2] for g in GROUPS]); ax.set_xlim(-0.5, len(GROUPS) - 0.5); ax.set_ylim(0.3, 0.95)
ax.set_ylabel("Cluster AUROC (held-out cells)")
ax.legend(handles=[Patch(color=c, label={"SIDISH": "SIDISH (hidden)", "DEGAS": "DEGAS (hidden)", "scSurv": "scSurv (1-D)", "Scissor": "Scissor (1-D)"}.get(m, m)) for m, c in MOD] + [Line2D([], [], color=INK, ls=(0, (2, 1.2)), lw=0.9, label="Raw-gene (ref.)")],
          frameon=False, fontsize=5.2, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), handlelength=1.0, columnspacing=0.6)
tag(ax, "c", "", x=-0.1, y=1.4)
ax.text(0.0, 1.36, "CellFactor vs baselines (same bag readout)", transform=ax.transAxes, fontsize=7, color=INK)

# d — E1 ρ 분포
ax = fig.add_subplot(gs[1, 4:])
for g, o in enumerate(("liver", "pancreas", "breast")):
    q = e1[e1.organ == o]
    ax.scatter(g - 0.13 + jit(len(q), 0.08), q.rho, s=4, color=BLUE, alpha=0.5, edgecolor="none", zorder=2)
    ax.scatter(g + 0.13 + jit(len(q), 0.08), q.rho_crosscluster_mean, s=4, color=RAND, alpha=0.5, edgecolor="none", zorder=2)
    ax.hlines(q.rho.median(), g - 0.25, g - 0.01, color=BLUE, lw=1.6, zorder=3); ax.hlines(q.rho_crosscluster_mean.median(), g + 0.01, g + 0.25, color=INK2, lw=1.6, zorder=3)
    ax.text(g, 1.02, f"ρ̃ {E1N[o][1]:.2f}\n{E1N[o][0]} cl.", ha="center", va="bottom", fontsize=5.2, color=INK)
ax.axhline(0, color=MUTED, lw=0.6, ls="--")
ax.set_xticks(range(3)); ax.set_xticklabels([f"Liver{LSTAR}", "Pancreas", "Breast"]); ax.set_ylim(-0.75, 1.25); ax.set_yticks([-0.5, 0, 0.5, 1])
ax.set_ylabel("Spearman ρ\n(readout vs program Δ)")
ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="none", markersize=3, label="own cluster"),
                   Line2D([], [], marker="o", color=RAND, ls="none", markersize=3, label="other clusters (null)")],
          frameon=False, fontsize=5.2, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, handletextpad=0.1, columnspacing=0.6)
tag(ax, "d", "", x=-0.55, y=1.27); ax.text(0.5, 1.27, "Readout direction", transform=ax.transAxes, fontsize=7, color=INK, ha="center")

# e–g — E1 예시
ORGL = {"liver": "Liver T", "pancreas": "Pancreas fibroblast", "breast": "Breast T"}
for i, o in enumerate(("liver", "pancreas", "breast")):
    ax = fig.add_subplot(gs[2, 2 * i:2 * i + 2]); r = EX[o]; axes, wv, dv = pts[(o, r.seed, r.cluster)]
    ax.axhline(0, color=GRID, lw=0.6, zorder=1); ax.axvline(0, color=GRID, lw=0.6, zorder=1)
    ax.scatter(dv, wv, s=8, color=BLUE, alpha=0.75, edgecolor="none", zorder=2)
    top = np.argsort(-wv)[:3]
    ax.scatter(dv[top], wv[top], s=15, facecolor="none", edgecolor=INK, linewidth=0.6, zorder=3)
    ax.text(0.97, 0.03, "top readout axes\n" + "\n".join(axes[k].replace("HM_", "").replace("_", " ")[:24] for k in top), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=4.8, color=INK2, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, linewidth=0.5, alpha=0.9), zorder=4)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.6 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xlabel("Cluster program Δ (SD)")
    if i == 0: ax.set_ylabel("Readout contribution\n(W × SD z′)")
    tag(ax, "efg"[i], f"{ORGL[o]} · {'/'.join(r.markers_top5.split(',')[:2])}\nρ = {r.rho:.2f}", x=-0.2)
# h–i — E4 H 학습 손실 절제(z′ 고정, H 만 다름)
E4 = J(OUT / "E4_HLOSS.json"); ABL = [("NO_PAIR", "no pair loss", "#c96f66"), ("NO_CORAL", "no CORAL", "#d9a441"), ("NONE", "neither", "#8a8984")]
G4 = [(LK4, f"Liver T cells{LSTAR}\ninternal\n(GSE151530)"), ("pancreas", "Pancreas\nfibroblasts, internal\n(GSE202051)"), ("breast", "Breast T cells\ninternal\n(GSE176078)"), ("breast_ext", "Breast T cells\nexternal\n(GSE161529)")]
ax = fig.add_subplot(gs[3, :4])
for g, (k, lab) in enumerate(G4):
    for j, (a, al, col) in enumerate(ABL):
        d = E4[k][f"full_minus_{a}"]; x = g + (j - 1) * 0.22
        ax.errorbar(x, 100 * d["mean"], yerr=[[100 * (d["mean"] - d["ci95"][0])], [100 * (d["ci95"][1] - d["mean"])]], fmt="o", color=col, ms=3.5, elinewidth=0.9, capsize=0, zorder=3)
for i in range(len(G4) - 1): ax.axvline(i + 0.5, color=GRID, lw=0.6)
ax.axhline(0, color=MUTED, lw=0.7, ls="--")
ax.set_xticks(range(len(G4))); ax.set_xticklabels([g[1] for g in G4]); ax.set_xlim(-0.5, len(G4) - 0.5)
ax.set_ylabel("AUROC, CellFactor − ablated H\n(×100, seed-paired, 95% CI)")
ax.legend(handles=[Line2D([], [], marker="o", color=c, ls="none", markersize=3.5, label=l) for _, l, c in ABL], frameon=False, fontsize=5.4, ncol=3,
          loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2)
ax.text(0.0, 1.2, "H loss ablation", transform=ax.transAxes, fontsize=7, color=INK)
tag(ax, "h", "", x=-0.1, y=1.2)
ax = fig.add_subplot(gs[3, 4:])
for g, (k, lab) in enumerate([(LK4, f"Liver{LSTAR}"), ("pancreas", "Pancreas"), ("breast", "Breast")]):
    m = E4[k]["E1_rho_median"]
    for j, (a, al, col) in enumerate([("full", "CellFactor", BLUE)] + ABL):
        ax.bar(g + (j - 1.5) * 0.19, m[a], width=0.17, color=col, zorder=2)
ax.axhline(0, color=INK2, lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels([f"Liver{LSTAR}", "Pancreas", "Breast"]); ax.set_ylim(-0.1, 0.75)
ax.set_ylabel("Median ρ")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in [BLUE] + [x[2] for x in ABL]], labels=["CellFactor"] + [x[1] for x in ABL], frameon=False, fontsize=5.0, loc="upper left", handlelength=0.8)
ax.text(0.5, 1.2, "Readout direction", transform=ax.transAxes, fontsize=7, color=INK, ha="center")
tag(ax, "i", "", x=-0.42, y=1.2)
fig.savefig(FIG / "S_program_axis.pdf", bbox_inches="tight"); fig.savefig(FIG / "S_program_axis.png", dpi=300, bbox_inches="tight")
print(e2.groupby(["organ", "eval_set"])[["full", "named", "free", "named_rand4_median"]].mean().round(3))
print(e3.groupby(["organ", "eval_set"])[["A_cand3k", "B0", "B1_cand3k", "F", "rawgene_reference"]].mean().round(3))
print({o: (round(r.rho, 3), r.seed, r.cluster, r.top3_readout) for o, r in EX.items()})
