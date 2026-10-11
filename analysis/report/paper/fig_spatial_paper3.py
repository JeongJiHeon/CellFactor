#!/usr/bin/env python3
"""공간전사체 그림 — 논문 Fig. 5 결과(P0 입력, CellFactor = A_cand3k c1–c6 절편 내 z 평균)에서 HER2ST 를 뺀 판(TNBC96 · Wu 간 · Zhou 췌장).
재계산 없음(저장 산출만 읽음): results/report_acand3k/{spatial2/SPATIAL2_ACAND3K.json, spatial2/section_auroc_long.csv, spots_acand3k/},
results/paper_acand3k/supp_spatial/SPATIAL_PANEL.json(5분위 Δ, 적격 절편·환자 수).
  a  대표 절편(CellFactor 종양 절편 AUROC 가 환자 평균에 가장 가까운 절편; 표시 가능 조건: 라벨 스팟 ≥150, 종양 비율 15–85%) — 최선 사례 아님.
     병리 + 절편 내 위험 백분위(6-NN 평활은 표시용)
  b  병리 × 위험 5분위 Δ = P(label | 5분위) − P(label), pp, 절편별 계산 후 적격 절편 평균(평활 없음)
  c  환자 단위 AUROC(절편 AUROC → 환자 평균 → 환자 평균), 환자 bootstrap 95% CI: CellFactor vs scSurv · SIDISH · Scissor · DEGAS
  d  짝 차이(같은 환자 재표집) — 기대 방향으로 맞춤: (+) 라벨은 CellFactor − 기준선, (−) 라벨은 기준선 − CellFactor. 양수 = CellFactor 가 기대 방향으로 더 강함
출력 final_results/figures/Fig_spatial.{pdf,png} · final_results/03_spatial/figure_spatial/NUMBERS_P0.json"""
import json, glob, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[3]; RA = ROOT / "results/report_acand3k"; SPD = RA / "spots_acand3k"
OUTD = ROOT / "final_results/03_spatial/figure_spatial"; OUTD.mkdir(parents=True, exist_ok=True); FIG = ROOT / "final_results/figures"
BLUE, INK, INK2, MUTED, GRID, GMD = "#2a78d6", "#16181b", "#52514e", "#8a8985", "#e6e5e1", "#8a8984"
LABC = {"Tumor": "#6a3d9a", "Ref": "#1b9e77", "Other": "#d5d2ca", "NA": "#f3f2ef"}
RISK = LinearSegmentedColormap.from_list("risk", ["#2166ac", "#f7f7f7", "#e66101"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
J = lambda p: json.load(open(p))
S2 = J(RA / "spatial2/SPATIAL2_ACAND3K.json")["labels"]; SP = J(ROOT / "results/paper_acand3k/supp_spatial/SPATIAL_PANEL.json")
T = pd.read_csv(RA / "spatial2/section_auroc_long.csv")
ACIN = ["pancreatic acinar cell", "type B pancreatic cell", "pancreatic epsilon cell"]


def tnbc(sec):
    z = np.load(SPD / "tnbc96_spots_v2" / sec, allow_pickle=True); keep = np.asarray(z["purity"], float) > 0; lab = np.asarray(z["label"], dtype=object).astype(str)[keep]
    l = np.where(np.isin(lab, ["Tumor", "in situ"]), "Tumor", np.where(np.isin(lab, ["High TIL stroma", "Lymphocyte", "Lymphoid nodule"]), "Ref",
                 np.where(np.isin(lab, ["nan", "None", ""]), "NA", "Other")))
    return np.asarray(z["xy"], float)[keep], l, np.asarray(z["CP-PCH"], float)[keep]


def liver(sec):
    z = np.load(SPD / "liver_wu_spots_e2pp" / sec, allow_pickle=True); lab = np.asarray(z["he"], dtype=object).astype(str)
    l = np.where(lab == "Tumor", "Tumor", np.where(lab == "Adj. liver", "Ref", np.where(lab == "nan", "NA", "Other")))
    return np.asarray(z["xy"], dtype=object).astype(float), l, np.asarray(z["CP-PCH"], float)


def panc(sec):
    z = np.load(SPD / "panc_zhou_spots_e2pp" / sec, allow_pickle=True); it = np.asarray(z["it"], bool); ct = np.asarray(z["ct"], dtype=object).astype(str)
    l = np.where(it, "Tumor", np.where(np.isin(ct, ACIN), "Ref", "Other"))
    return np.asarray(z["xy"], float), l, np.asarray(z["CP-PCH"], float)


DS = [("TNBC96", "TNBC96|Tumor", tnbc, "TNBC (legacy ST array)", "lymphocyte · TIL stroma", "tnbc", True),
      ("Wu", "Wu|Tumor", liver, "Liver HCC (CytAssist Visium)", "adjacent liver", "liver", True),
      ("Zhou", "Zhou|tumor (is_tumor)", panc, "Pancreas PDAC (Visium)", "acinar/islet", "panc", True)]


def smooth(xy, v, k=6):
    nb = cKDTree(xy).query(xy, k=k + 1)[1]; return v[nb].mean(1)


def pct(v): return v.argsort().argsort() / max(len(v) - 1, 1) * 100


def representative(ds, key, getter):
    lab = key.split("|")[1]; sub = T[(T.ds == ds) & (T.label == lab) & (T.model == "A_cand3k")]; pm = S2[key]["models"]["A_cand3k"]["patient_mean"]; cand = []
    for r in sub.itertuples():
        xy, l, v = getter(r.section); m = l != "NA"; fr = float((l[m] == "Tumor").mean()) if m.any() else 0
        if m.sum() >= 150 and 0.15 <= fr <= 0.85: cand.append((abs(r.auc - pm), r.section, float(r.auc)))
    cand.sort(); return cand[0][1], cand[0][2], pm


import os
from scipy.stats import rankdata as _rk
PICK = os.environ.get("PICK", "best")   # best = 종양·기준 라벨 둘 다 적격인 절편 중 (종양 상위 20% Δ + 기준 하위 20% Δ) 최대 · rep = 환자 평균에 가장 가까운 종양 AUROC


def best_case(ds, getter):
    files = {"TNBC96": "tnbc96_spots_v2/*_all5.npz", "Wu": "liver_wu_spots_e2pp/*.npz", "Zhou": "panc_zhou_spots_e2pp/*.npz"}[ds]; cand = []
    for f in sorted(Path(p).name for p in glob.glob(str(SPD / files))):
        xy, l, v = getter(f); m = l != "NA"; n = int(m.sum())
        if n < 150: continue
        l_, v_ = l[m], v[m]; q = np.minimum(((_rk(v_) - 1) / n * 5).astype(int), 4); sc = 0; ok = True
        for c, band in (("Tumor", 4), ("Ref", 0)):
            y = l_ == c
            if y.sum() < 20 or y.mean() < 0.05: ok = False; break
            sc += 100 * (y[q == band].mean() - y.mean())
        if ok: cand.append((sc, f))
    cand.sort(reverse=True); k = RANK.get(ds, 1) - 1; return cand[k][1], cand[k][0]


RANK = {"Wu": 2}   # 사용자 선택(10/7): 간은 최선 사례 2위 절편(Wu73 post, 다른 환자). 유방·췌장은 1위. env RANK_<ds> 로 바꿈
for _d in ("TNBC96", "Wu", "Zhou"):
    if os.environ.get(f"RANK_{_d}"): RANK[_d] = int(os.environ[f"RANK_{_d}"])


fig = plt.figure(figsize=(7.2, 8.0))
fig.set_size_inches(7.2, 8.9)
gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.0, 0.45, 0.62, 1.45], hspace=0.5, wspace=0.25, top=0.915)
num = {}
for i, (ds, key, getter, title, refname, bk, inv) in enumerate(DS):
    if PICK == "best":
        sec, sc = best_case(ds, getter); xy, l, v = getter(sec); m_ = l != "NA"
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(l[m_] == "Tumor", v[m_])); pm = S2[key]["models"]["A_cand3k"]["patient_mean"]
        num[f"example_{ds}"] = {"section": sec, "rule": "best case: max tumour top-20% Δ + reference bottom-20% Δ (both labels eligible)", "score_pp": sc, "section_tumour_auroc": auc}
    else:
        sec, auc, pm = representative(ds, key, getter); xy, l, v = getter(sec); num[f"example_{ds}"] = {"section": sec, "section_auroc": auc, "patient_mean_auroc": pm}
    if ds == "Wu":   # 간: 전체 절편 대부분이 미주석 → 주석 스팟만 표시(정량은 위에서 이미 주석 스팟만 사용)
        mk_ = l != "NA"; xy, l, v = xy[mk_], l[mk_], v[mk_]
    s = max(1.2, min(16.0, 3.2e3 / len(xy))) * {"TNBC96": 2.3}.get(ds, 1.0)   # TNBC: 점 사이 흰 바탕이 비쳐 색이 연해 보이지 않도록 스팟 간격에 맞춤
    for row in (0, 1):
        ax = fig.add_subplot(gs[row, i]); ax.set_aspect("equal"); ax.axis("off")
        if row == 0:
            for c in ("NA", "Other", "Ref", "Tumor"):
                m = l == c; ax.scatter(xy[m, 0], xy[m, 1], s=s, c=LABC[c], linewidths=0)
            ax.set_title(f"{title}\n{sec.replace('_all5.npz', '').replace('.npz', '').rstrip('-')} · {len(xy):,} {'labelled ' if ds == 'Wu' else ''}spots", fontsize=6.6)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=s, c=pct(smooth(xy, v)), cmap=RISK, vmin=0, vmax=100, linewidths=0)
            if PICK == "best":
                m_ = l != "NA"; qq = np.minimum(((_rk(v[m_]) - 1) / m_.sum() * 5).astype(int), 4); lt = l[m_]
                dt = 100 * ((lt == "Tumor")[qq == 4].mean() - (lt == "Tumor").mean()); dr = 100 * ((lt == "Ref")[qq == 0].mean() - (lt == "Ref").mean())
                ax.set_title(f"top 20% risk: tumor +{dt:.0f} pp\nbottom 20% risk: non-malignant +{dr:.0f} pp", fontsize=5.6, color=INK2)
            else:
                ax.set_title(f"section tumor AUROC {auc:.2f}\n(patient-level mean {pm:.2f})", fontsize=5.9, color=INK2)
        if ds == "Wu":
            mm = l != "NA"; px = 0.08 * max(np.ptp(xy[mm, 0]), np.ptp(xy[mm, 1]))
            ax.set_xlim(xy[mm, 0].min() - px, xy[mm, 0].max() + px); ax.set_ylim(xy[mm, 1].min() - px, xy[mm, 1].max() + px)
        if inv: ax.invert_yaxis()
        if i == 0: ax.text(-0.06, 0.5, "Pathology" if row == 0 else "CellFactor risk\n(percentile in section)", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.6)
fig.legend(handles=[Patch(color=LABC["Tumor"], label="Tumor"), Patch(color=LABC["Ref"], label="Non-malignant"),
                    Patch(color=LABC["Other"], label="Other labelled"), Patch(color=LABC["NA"], label="Unlabelled")],
           loc="upper left", bbox_to_anchor=(0.08, 0.985), ncol=4, frameon=False, fontsize=5.8, handlelength=1.0)
cax = fig.add_axes([0.76, 0.968, 0.15, 0.007]); cb = matplotlib.colorbar.ColorbarBase(cax, cmap=RISK, norm=Normalize(0, 100), orientation="horizontal")
cb.set_ticks([0, 50, 100]); cb.ax.tick_params(labelsize=5.4, length=1.5); cax.set_title("risk percentile in section", fontsize=5.6, pad=1.5, color=INK2)

# 라벨 정의 표 — 장기별로 각 병리 범주에 들어간 원 주석
ax = fig.add_subplot(gs[2, :]); ax.axis("off")
COLS = ["TNBC\n(pathologist annotation)", "Liver HCC\n(H&E annotation)", "Pancreas PDAC\n(cell-type annotation)"]
ROWS = [("Tumor", "Tumor", ["Tumor, in situ", "Tumor", "Tumor spots"]),
        ("Non-malignant", "Ref", ["Lymphocyte, high-TIL stroma, lymphoid nodule", "Adjacent normal liver", "Acinar and islet cells"]),
        ("Other", "Other", ["Other annotated (stroma, fat, normal duct, …)", "Stroma", "All other spots"])]
x0, cw, rh = 0.2, (1 - 0.2) / 3, 0.2
for j, c in enumerate(COLS): ax.text(x0 + (j + 0.5) * cw, 1.0, c, ha="center", va="center", fontsize=5.9, color=INK, fontweight="bold", transform=ax.transAxes)
for i_, (rn, ck, cells) in enumerate(ROWS):
    y = 0.6 - i_ * 0.27
    ax.add_patch(Rectangle((0.0, y - 0.07), 0.012, 0.14, color=LABC[ck], transform=ax.transAxes, clip_on=False))
    ax.text(0.02, y, rn, ha="left", va="center", fontsize=6.0, color=INK, transform=ax.transAxes)
    for j, t in enumerate(cells): ax.text(x0 + (j + 0.5) * cw, y, t, ha="center", va="center", fontsize=5.7, color=INK2, transform=ax.transAxes)
    ax.plot([0, 1], [y + 0.135, y + 0.135], color=GRID, lw=0.5, transform=ax.transAxes, clip_on=False)

# b — 병리(Tumour · Reference · Other) × 위험 5분위 Δ: 절편별(평활 없음), 라벨 ≥20 스팟·≥5% 인 적격 절편 평균. c 와 같은 기준 라벨
from scipy.stats import rankdata
FILES = {"TNBC96": sorted(Path(p).name for p in glob.glob(str(SPD / "tnbc96_spots_v2" / "*_all5.npz"))),
         "Wu": sorted(Path(p).name for p in glob.glob(str(SPD / "liver_wu_spots_e2pp" / "*.npz"))),
         "Zhou": sorted(Path(p).name for p in glob.glob(str(SPD / "panc_zhou_spots_e2pp" / "*.npz")))}
PID = {"TNBC96": lambda f: re.findall(r"TNBC\d+", f)[0], "Wu": lambda f: re.findall(r"\d+", f)[0], "Zhou": lambda f: re.findall(r"HT\d+P\d+", f)[0]}
vmax = 25
for i, (ds, key, getter, title, refname, bk, inv) in enumerate(DS):
    acc = {c: [] for c in ("Tumor", "Ref", "Other")}; pats = {c: set() for c in acc}
    for f in FILES[ds]:
        xy, l, v = getter(f); m = l != "NA"
        if m.sum() < 20: continue
        l, v = l[m], v[m]; q = np.minimum(((rankdata(v) - 1) / len(v) * 5).astype(int), 4)
        for c in acc:
            y = l == c; base = y.mean()
            if y.sum() < 20 or base < 0.05: continue
            acc[c].append([100 * (y[q == k].mean() - base) for k in range(5)]); pats[c].add(PID[ds](f))
    B = np.vstack([np.mean(acc[c], 0) if acc[c] else np.full(5, np.nan) for c in acc]); ns = [len(acc[c]) for c in acc]; nps = [len(pats[c]) for c in acc]
    ax = fig.add_subplot(gs[3, i]); ax.imshow(B, cmap=RISK, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect="auto")
    for a in range(3):
        for b in range(5):
            if np.isfinite(B[a, b]): ax.text(b, a, "0" if round(B[a, b]) == 0 else f"{B[a, b]:+.0f}", ha="center", va="center", fontsize=5.4, color=INK)
        if ns[a] == 1: ax.add_patch(Rectangle((-0.5, a - 0.5), 5, 1, fill=False, hatch="////", ec=INK2, lw=0, alpha=0.55))
    ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"], fontsize=5.4)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Tumor", "Non-malignant", "Other"] if i == 0 else [], fontsize=5.8); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    nl = lambda k: f"{ns[k]}" + (f"/{nps[k]}" if nps[k] != ns[k] else "")
    ax.set_title(f"{ {'TNBC96': 'TNBC', 'Wu': 'Liver', 'Zhou': 'Pancreas'}[ds]} · n sections{'/patients' if nps != ns else ''}: T {nl(0)} · N {nl(1)} · O {nl(2)}", fontsize=5.6, color=INK2)
    if i == 0: AX_B = ax
    num[f"bands_{ds}"] = {"delta_pp": np.round(B, 2).tolist(), "n_sections": ns, "n_patients": nps}

# c
LABS = [("TNBC96|Tumor", "TNBC96 · tumor (+)", 1), ("TNBC96|Lymphocyte", "TNBC96 · lymphocyte (−)", -1), ("TNBC96|High TIL stroma", "TNBC96 · high-TIL stroma (−)", -1),
        ("Wu|Tumor", "Liver · tumor (+)", 1), ("Wu|Adj. liver", "Liver · adjacent liver (−)", -1),
        ("Zhou|tumor (is_tumor)", "Pancreas · tumor (+)", 1), ("Zhou|acinar/islet (celltype)", "Pancreas · acinar/islet (−)", -1)]
MOD = [("A_cand3k", "CellFactor", "o"), ("scSurv", "scSurv", "s"), ("SIDISH", "SIDISH", "D"), ("Scissor", "Scissor", "^"), ("DEGAS", "DEGAS", "v")]
gsb = gs[4, :].subgridspec(1, 3, width_ratios=[1.35, 0.5, 1.0], wspace=0.05)
ax = fig.add_subplot(gsb[0, 0]); y0 = np.arange(len(LABS))[::-1]; off = np.linspace(0.27, -0.27, len(MOD))
for (k, nm, sg), y in zip(LABS, y0):
    for (mk, mn, mrk), dy in zip(MOD, off):
        m = S2[k]["models"][mk]; col = BLUE if mk == "A_cand3k" else GMD
        ax.plot([m["lo"], m["hi"]], [y + dy, y + dy], color=col, lw=1.0 if mk == "A_cand3k" else 0.7)
        ax.plot([m["patient_mean"]], [y + dy], marker=mrk, ms=3.6 if mk == "A_cand3k" else 2.8, color=col, mec="white", mew=0.3, ls="none", zorder=3)
    num[k] = {mn: S2[k]["models"][mk] for mk, mn, _ in MOD}; num[k]["n_patients"] = S2[k]["n_patients"]
for yy in y0[1::2]: ax.axhspan(yy - 0.5, yy + 0.5, color="#f6f6f4", zorder=0)
ax.axvline(0.5, color=INK2, lw=0.7, ls=(0, (3, 2))); ax.set_xlim(0.0, 1.0); ax.set_ylim(-0.6, len(LABS) - 0.4)
ax.set_yticks(y0); ax.set_yticklabels([f"{nm}\n(n = {S2[k]['n_patients']} patients)" for k, nm, _ in LABS], fontsize=5.4); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Patient-level AUROC", fontsize=6.0)
ax.legend(handles=[plt.Line2D([], [], marker=mrk, color=BLUE if mk == "A_cand3k" else GMD, ls="none", ms=3.5, label=mn) for mk, mn, mrk in MOD],
          frameon=False, fontsize=5.4, ncol=5, loc="lower center", bbox_to_anchor=(0.4, 1.0), handletextpad=0.2, columnspacing=0.7)
AX_C = ax

# d
ax = fig.add_subplot(gsb[0, 2]); BL = ["scSurv", "SIDISH", "Scissor", "DEGAS"]
D = np.full((len(LABS), 4), np.nan); SIG = np.zeros((len(LABS), 4), int)
for a, (k, nm, sg) in enumerate(LABS):
    for b, m in enumerate(BL):
        p = S2[k]["paired"][m]; d, lo, hi = sg * p["delta"], *sorted((sg * p["lo"], sg * p["hi"]))
        D[a, b] = d; SIG[a, b] = 1 if lo > 0 else (-1 if hi < 0 else 0)
lim = np.nanmax(np.abs(D))
ax.imshow(D, cmap=LinearSegmentedColormap.from_list("cmp", ["#8c510a", "#f7f7f7", "#2a78d6"]), norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
for a in range(len(LABS)):
    for b in range(4):
        ax.text(b, a, f"{D[a, b]:+.2f}", ha="center", va="center", fontsize=5.4, color=INK, fontweight="bold" if SIG[a, b] else "normal")
        if SIG[a, b]: ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, edgecolor=INK if SIG[a, b] > 0 else "#8c510a", lw=0.9))
ax.set_xticks(range(4)); ax.set_xticklabels(BL, fontsize=5.8); ax.xaxis.tick_top(); ax.tick_params(length=0)
ax.set_yticks(range(len(LABS))); ax.set_yticklabels([nm.split(" (")[0] for _, nm, _ in LABS], fontsize=5.4)
for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor(GRID)
ax.set_xlabel("ΔAUROC, CellFactor vs baseline (expected direction)", fontsize=6.0)
AX_D = ax
num["paired_expected_direction"] = {k: {m: {"d": float(D[a, b]), "sig": int(SIG[a, b])} for b, m in enumerate(BL)} for a, (k, _, _) in enumerate(LABS)}

# 패널 글자: a·b·c 는 같은 왼쪽 x, d 는 c 와 같은 높이
XL = 0.015
fig.text(XL, 0.985, "a", fontsize=9, fontweight="bold", va="top")
pb = AX_B.get_position(); fig.text(XL, pb.y1 + 0.03, "b", fontsize=9, fontweight="bold", va="bottom")
pc = AX_C.get_position(); yc = pc.y1 + 0.022; fig.text(XL, yc, "c", fontsize=9, fontweight="bold", va="bottom")
pd_ = AX_D.get_position(); fig.text(pd_.x0 - 0.13, yc, "d", fontsize=9, fontweight="bold", va="bottom")
fig.savefig(FIG / "Fig_spatial.pdf", bbox_inches="tight"); fig.savefig(FIG / "Fig_spatial.png", dpi=300, bbox_inches="tight")
json.dump(num, open(OUTD / "NUMBERS_P0.json", "w"), indent=1, default=float)
for k, nm, sg in LABS: print(nm, {mn: round(S2[k]["models"][mk]["patient_mean"], 3) for mk, mn, _ in MOD})
print(np.round(D, 3)); print(SIG)
