#!/usr/bin/env python3
"""PLAN_MECH §2(변경 0fbccae 반영) — TCGA 독립 표현형과 π̂·상태·기여의 연관. 재학습 없음.
원천: results/hvg_design/run1a/{organ}/A_cand3k/c1–c6/pred_bulk.pt(TCGA 전체 π̂·M̂·W·α),
     00.data/tcga_pheno/thorsson_mmc2.xlsx(TIL 영역 비율·면역 아형·백혈구 비율), cbio_mutations_TP53_CTNNB1_KRAS.json.
판정 가설(장기별 Holm): H2 T세포 π̂ ~ TIL 영역 비율 ρ>0 · H4 TP53 변이군 악성 증식 기여 큼 · H5(간) CTNNB1 변이군 T세포 π̂ 작음 ·
  H6a 면역 아형 C2 vs C4∪C6 T세포 π̂ 큼 · H6b 같은 비교에서 악성 IFN_I 상태 큼. 모두 단측.
기술: H3 T세포 기여 ~ TIL·백혈구(양측) · 학습 표적 점검 π̂_면역 합 ~ 백혈구 비율(held-out 만).
출력 results/mech/exp2/{PHENO_ASSOC.csv,.json,.md} (+ results_json 미러)"""
import os
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[3]; PH = Path(os.path.join(os.environ.get("CELLFACTOR_DATA", "data"), "tcga_pheno"))
OUT = ROOT / "results/mech/exp2"; OUT.mkdir(parents=True, exist_ok=True); OUTJ = ROOT / "results_json/mech/exp2"; OUTJ.mkdir(parents=True, exist_ok=True)
SPEC = {"breast": dict(study="brca", T="T-cells", mal="Cancer Epithelial", immune=["T-cells", "B-cells", "Plasmablasts", "Myeloid"]),
        "liver": dict(study="lihc", T="T cells", mal="Malignant cells", immune=["T cells", "B cells", "TAMs"]),
        "pancreas": dict(study="paad", T="Immune", mal="Epithelial", immune=["Immune"])}
PROLIF = ["CS_PROLIFERATION", "HM_E2F_Targets", "HM_G2_M_Checkpoint"]

TH = pd.read_excel(PH / "thorsson_mmc2.xlsx").set_index("TCGA Participant Barcode")
MU = json.load(open(PH / "cbio_mutations_TP53_CTNNB1_KRAS.json"))
# 사후 기준선(2026-10-10, 리뷰 대응·기술용, Holm 밖): (a) bulk 증식 점수 · (b) CIBERSORT LM22 T세포(+NK, CCE 교사와 같은 대응) 상대비 × 메틸화 백혈구 비율
import sys as _sys; _sys.path.insert(0, str(ROOT)); import train as _T
TD = Path(os.path.join(os.environ.get("CELLFACTOR_DATA", "data"), "tcga")); CANCER = {"breast": "BRCA", "liver": "LIHC", "pancreas": "PAAD"}
_T_COLS = ["T.cells.CD8", "T.cells.CD4.naive", "T.cells.CD4.memory.resting", "T.cells.CD4.memory.activated", "T.cells.follicular.helper",
           "T.cells.regulatory..Tregs.", "T.cells.gamma.delta", "NK.cells.resting", "NK.cells.activated"]
def cib_T_abs(org, pats):
    cib = pd.read_csv(TD / "TCGA_cibersort_relative.tsv", sep="\t"); cib = cib[cib.CancerType == CANCER[org]].copy()
    cib["pid"] = cib.SampleID.str.replace(".", "-", regex=False).str[:12]; cib = cib[pd.to_numeric(cib["P.value"], errors="coerce") <= 0.05]
    cib = cib.sort_values(["pid", "P.value"]).drop_duplicates("pid").set_index("pid"); tf = cib[_T_COLS].sum(1)
    lf = pd.read_csv(TD / "TCGA_leukocyte_fraction.tsv", sep="\t", header=None, names=["c", "s", "v"]); lf["pid"] = lf.s.str[:12]; lf = lf.groupby("pid").v.mean()
    return (tf.reindex(pats) * lf.reindex(pats)).values.astype(float)
def bulk_prolif(org):
    cfg = _T.ORGANS[org]; b = _T.load(cfg["registry"]); g = {str(x): i for i, x in enumerate(b["gene_names"])}; d = json.load(open(ROOT / cfg["programs"])); x = b["bulk_x"].double().numpy()
    sc = [x[:, [g[t] for t in d[k] if t in g]].mean(1) for k in PROLIF]; return np.mean([(v - v.mean()) / v.std() for v in sc], 0)


def ids(org):
    if org == "breast":   # 2026-10-10: CCE 학습에 쓴 정확 TCGA ID(1,089행 전부; 기존 레지스트리 ID 717개와 전부 일치). 이전 = 레지스트리 생존 일치 복구 ID(372 결측)
        return [str(x) for x in torch.load(ROOT / "results/stage2_redesign/run_20260828_p16k_stage2a_100k/BREAST_2A_SUPERVISION.pt", weights_only=False)["patient_ids"]]
    p = {"liver": ROOT / "results/liver/LIVER_STAGE1_DATA.pt", "pancreas": ROOT / "sandbox6/results/pancreas/PANC_STAGE1_DATA.pt"}[org]
    return [str(x) for x in torch.load(p, map_location="cpu", weights_only=False, mmap=True)["bulk_patient_ids"]]


def mutated(org, gene, pats):
    st = MU[SPEC[org]["study"]]; seq = {s[:12] for s in st["sequenced"]}; mut = {m["sample"][:12] for m in st["mutations"] if m["gene"] == gene}
    return np.array([(1.0 if p in mut else 0.0) if p in seq else np.nan for p in pats])


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


rows = []
for org, sp in SPEC.items():
    PB = [torch.load(ROOT / f"results/hvg_design/run1a/{org}/A_cand3k/c{c}/pred_bulk.pt", map_location="cpu", weights_only=False) for c in range(1, 7)]
    types, progs = list(PB[0]["cell_types"]), list(PB[0]["programs"]); n = len(PB[0]["time_all"])
    pid = ids(org); assert len(pid) == n, (org, len(pid), n)
    pats = [p[:12] for p in pid]; held = np.zeros(n, bool); held[PB[0]["heldout_idx"].numpy()] = True
    it, im = types.index(sp["T"]), types.index(sp["mal"]); kp = [progs.index(p) for p in PROLIF]; kifn = progs.index("CS_IFN_I")
    pi = np.mean([p["pi"].double().numpy() for p in PB], 0)
    contT, prolif, ifn = [], [], []
    for p in PB:
        P, M, W, a = p["pi"].double().numpy(), p["M"].double().numpy(), p["W"].double().numpy(), p["alpha"].double().numpy()
        r = (P * a[None]).sum(1) + (P[:, :, None] * W[None] * M).sum((1, 2)); s = r.std()
        contT.append((P[:, it] * (a[it] + (W[it][None] * M[:, it]).sum(1))) / s)
        prolif.append((P[:, im, None] * W[im, kp][None] * M[:, im, kp]).sum(1) / s)
        x = M[:, im, kifn]; ifn.append((x - x.mean()) / x.std())
    contT, prolif, ifn = np.mean(contT, 0), np.mean(prolif, 0), np.mean(ifn, 0)
    th = TH.reindex(pats); til = th["TIL Regional Fraction"].values.astype(float); lf = th["Leukocyte Fraction"].values.astype(float); isub = th["Immune Subtype"].values
    tp53 = mutated(org, "TP53", pats); ctnnb1 = mutated(org, "CTNNB1", pats)
    bpro = bulk_prolif(org); assert len(bpro) == n; cibT = cib_T_abs(org, pats)
    piT = pi[:, it]; piImm = pi[:, [types.index(t) for t in sp["immune"]]].sum(1)

    def sp_test(x, y, mask, alt):
        m = mask & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10: return dict(n=int(m.sum()), stat=np.nan, p=np.nan)
        rho, p2 = spearmanr(x[m], y[m]); p = p2 / 2 if (alt == "greater" and rho > 0) else (1 - p2 / 2 if alt == "greater" else p2)
        return dict(n=int(m.sum()), stat=float(rho), p=float(p))

    def mw_test(x, g, mask, alt):   # g: 1 vs 0
        m = mask & np.isfinite(x) & np.isfinite(g); a_, b_ = x[m & (g == 1)], x[m & (g == 0)]
        if min(len(a_), len(b_)) < 5: return dict(n=f"{len(a_)}/{len(b_)}", stat=np.nan, p=np.nan)
        u = mannwhitneyu(a_, b_, alternative=alt); return dict(n=f"{len(a_)}/{len(b_)}", stat=float(np.median(a_) - np.median(b_)), p=float(u.pvalue), auc=float(u.statistic / (len(a_) * len(b_))))

    c2 = np.array([1.0 if s == "C2" else (0.0 if s in ("C4", "C6") else np.nan) for s in isub])
    for scope, mask in (("all", np.ones(n, bool)), ("heldout", held)):
        tests = [("H2", "T-cell π̂ ~ TIL regional fraction (ρ>0)", sp_test(piT, til, mask, "greater"), True),
                 ("H4", "TP53 mut: malignant proliferation contribution higher", mw_test(prolif, tp53, mask, "greater"), True)]
        if org == "liver": tests.append(("H5", "CTNNB1 mut: T-cell π̂ lower", mw_test(piT, ctnnb1, mask, "less"), True))
        tests += [("H6a", "Immune subtype C2 vs C4/C6: T-cell π̂ higher", mw_test(piT, c2, mask, "greater"), True),
                  ("H6b", "Immune subtype C2 vs C4/C6: malignant IFN-I state higher", mw_test(ifn, c2, mask, "greater"), True),
                  ("H3a", "T-cell contribution ~ TIL regional fraction (descriptive)", sp_test(contT, til, mask, "two"), False),
                  ("H3b", "T-cell contribution ~ leukocyte fraction (descriptive)", sp_test(contT, lf, mask, "two"), False)]
        tests += [("BL-H4", "Baseline (post hoc): bulk proliferation score, TP53 mut higher", mw_test(bpro, tp53, mask, "greater"), False),
                  ("BL-H2", "Baseline (post hoc): CIBERSORT T × leukocyte fraction ~ TIL regional fraction", sp_test(cibT, til, mask, "greater"), False),
                  ("BL-H6a", "Baseline (post hoc): CIBERSORT T × leukocyte fraction, C2 vs C4/C6 higher", mw_test(cibT, c2, mask, "greater"), False)]
        if org == "liver": tests.append(("BL-H5", "Baseline (post hoc): CIBERSORT T × leukocyte fraction, CTNNB1 mut lower", mw_test(cibT, ctnnb1, mask, "less"), False))
        if scope == "heldout": tests.append(("CHK", "Immune π̂ sum ~ leukocyte fraction (2A training target; check only)", sp_test(piImm, lf, mask, "two"), False))
        conf = [i for i, t in enumerate(tests) if t[3] and np.isfinite(t[2]["p"])]
        adj = holm(np.array([tests[i][2]["p"] for i in conf])) if conf else []
        for j, (h, desc, res, judged) in enumerate(tests):
            pa = float(adj[conf.index(j)]) if j in conf else np.nan
            rows.append({"organ": org, "scope": scope, "hyp": h, "description": desc, **res, "p_holm": pa, "judged": judged,
                         "verdict": ("supported" if pa < 0.05 else "not supported") if judged and np.isfinite(pa) else ("n/a" if judged else ("baseline (post hoc)" if h.startswith("BL") else "descriptive"))})
    print(org, "n", n, "TIL", int(np.isfinite(til).sum()), "TP53 seq", int(np.isfinite(tp53).sum()), "C2/C4C6", int(np.isfinite(c2).sum()), "CIBERSORT T", int(np.isfinite(cibT).sum()), flush=True)

R = pd.DataFrame(rows)
for d in (OUT, OUTJ): R.to_csv(d / "PHENO_ASSOC.csv", index=False, float_format="%.4g"); R.to_json(d / "PHENO_ASSOC.json", orient="records", indent=1)
md = ["# TCGA 독립 표현형 연관 (PLAN_MECH §2, 변경 0fbccae)", "", "단측 검정, 장기·범위별 Holm. 효과 = Spearman ρ 또는 중앙값 차(변이군 − 야생형). 'all' = train+held-out(train 은 W 적합에 쓰임), 'heldout' = held-out 만.", "",
      "| 장기 | 범위 | 가설 | 내용 | n | 효과 | p | Holm p | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    f = lambda v: "—" if not isinstance(v, (int, float)) or not np.isfinite(v) else f"{v:.3g}"
    md.append(f"| {r['organ']} | {r['scope']} | {r['hyp']} | {r['description']} | {r['n']} | {f(r['stat'])} | {f(r['p'])} | {f(r['p_holm'])} | {r['verdict']} |")
for d in (OUT, OUTJ): open(d / "PHENO_ASSOC.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
