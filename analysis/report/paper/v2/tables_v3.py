#!/usr/bin/env python3
"""원고 v2(확정안) Table 1 · Table 2. 재계산 없음.
Table 1 — Prognostic performance of CellFactor and comparator methods: 행 = 방법(발표된 방법 / 단순화 모델), 열 = 장기별 외부 평균 · held-out C.
  CI·짝 판정: final_results/02_cindex/bootstrap_ci_B2000/{organ}_BOOT.json (B=2000). † 모델: final_results/02_cindex/simple_baselines/TABLE_simple_baselines.csv(이전 파이프라인, 점추정).
Table 2 — Clinical-adjusted prognostic association: results/mech/exp3/COX_ADJ.json
출력 final_results/figures/manuscript_v2/{Table1,Table2}.{md,csv}"""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[4]; D = ROOT / "final_results/02_cindex"; OUTD = ROOT / "final_results/figures/manuscript_v2"
ORG = [("breast", "Breast"), ("liver", "Liver"), ("pancreas", "Pancreas")]
BOOT = {o: json.load(open(D / f"bootstrap_ci_B2000/{o}_BOOT.json")) for o, _ in ORG}
SB = pd.read_csv(D / "simple_baselines/TABLE_simple_baselines.csv")
PUB = ["CellFactor", "SIDISH", "scSurv", "Scissor", "DEGAS"]
SIMPLE = [("Bulk Cox", "Bulk Cox", None), ("Composition-only Cox", "Composition Cox", None),
          ("Program-score Cox", "Program-score Cox", None), ("Composition + program-score Cox", "Composition + program-score Cox", None)]


def cell(o, m, k):
    if m in BOOT[o]["models"]:
        v = BOOT[o]["models"][m][k]; s = f"{v['point']:.3f} ({v['ci95'][0]:.3f}–{v['ci95'][1]:.3f})"
        pc = BOOT[o]["paired_vs_CellFactor"].get(m, {}).get(k)
        if pc: s += "*" if pc["ci95"][0] > 0 else ("‡" if pc["ci95"][1] < 0 else "")
        return s, v["point"]
    q = SB[SB.organ == o]; row = q[q.endpoint == "Mean of external endpoints"] if k == "external_mean" else q[q.set == "Held-out"]
    x = float(row[m].iloc[0]); return f"{x:.3f}", x


rows = []
for grp, items in (("CellFactor and published single-cell–bulk methods", [(m, m, None) for m in PUB]), ("Simplified bulk models", SIMPLE)):
    for key, lab, src in items:
        r = {"Group": grp, "Method": lab}
        for o, on in ORG:
            for k, kl in (("external_mean", "external mean"), ("heldout", "held-out")):
                r[f"{on} {kl}"], r[f"_{o}_{k}"] = cell(o, key, k)
        rows.append(r)
T = pd.DataFrame(rows)
cols = [f"{on} {kl}" for kl in ("external mean", "held-out") for _, on in ORG]
best = {c: T[f"_{o}_{k}"].max() for c, (o, k) in zip(cols, [(o, k) for k in ("external_mean", "heldout") for o, _ in ORG])}
T[["Group", "Method"] + cols].to_csv(OUTD / "Table1.csv", index=False)
md = ["**Table 1. Prognostic performance of CellFactor and comparator methods.** Harrell's C-index (95% CI, 2,000 patient bootstrap resamples within cohorts).", "",
      "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
for grp in dict.fromkeys(T.Group):
    md.append(f"| ***{grp}*** |" + " |" * len(cols))
    for _, r in T[T.Group == grp].iterrows():
        cells = []
        for c, (o, k) in zip(cols, [(o, k) for k in ("external_mean", "heldout") for o, _ in ORG]):
            v = r[c]; cells.append(f"**{v}**" if abs(r[f'_{o}_{k}'] - best[c]) < 1e-12 else v)
        md.append(f"| {'**CellFactor**' if r.Method == 'CellFactor' else r.Method} | " + " | ".join(cells) + " |")
md += ["", "External mean = mean C over primary external endpoints (breast: SCAN-B OS, RFi, DRFi, BCFi and METABRIC OS, RFS; liver: GSE14520 and GSE76427 OS and RFS; "
       "pancreas: GSE62452, GSE57495 and GSE85916 OS). Held-out = TCGA overall survival of held-out patients (events: breast 30, liver 26, pancreas 19). "
       "Bold, highest in column. * Paired bootstrap 95% CI of CellFactor − method above 0; ‡ below 0 (method higher). "
       "CellFactor = ensemble of chains c1–c6. Bulk Cox = ridge Cox on all bulk genes; Composition Cox = Cox on estimated cell-type fractions. "
       "Simplified models were re-implemented from CellFactor components (same composition estimates and patients, without the CellFactor state estimator; fitted on TCGA training patients): "
       "Program-score Cox = ridge Cox on bulk program scores without deconvolution; Composition + program-score Cox = additive ridge Cox on fractions and bulk program scores (no cell-type × state interaction); Ridge penalty chosen by 5-fold cross-validation on training patients (breast program-score models selected the smallest penalty in the grid). Per cohort–endpoint values: Supplementary Table S1."]
open(OUTD / "Table1.md", "w").write("\n".join(md) + "\n")

# Table 2
C = json.load(open(ROOT / "results/mech/exp3/COX_ADJ.json"))
CV = {"SCAN-B": "Age, tumour size, nodal status, NHG grade, ER, HER2, PAM50", "METABRIC": "Age, NPI, ER, HER2, PAM50 (incl. claudin-low)",
      "GSE14520": "Age, sex, TNM stage, tumour size >5 cm, AFP >300 ng/ml, cirrhosis", "GSE76427": "Age, sex, BCLC stage, TNM stage",
      "GSE62452": "Stage, grade", "GSE57495": "Stage"}
order = ["SCAN-B", "METABRIC", "GSE14520", "GSE76427", "GSE57495", "GSE62452"]
rows2 = []
for coh in order:
    r = next(x for x in C if x["cohort"] == coh)
    rows2.append({"Cancer": r["organ"].capitalize(), "Cohort": coh, "n (events)": f"{r['n']} ({r['events']})", "Adjustment covariates": CV[coh],
                  "Unadjusted HR": f"{r['HR_unadj']:.2f}", "Adjusted HR (95% CI)": f"{r['HR_adj']:.2f} ({r['HR_lo']:.2f}–{r['HR_hi']:.2f})",
                  "P (likelihood ratio)": f"{r['p_LR']:.2g}"})
T2 = pd.DataFrame(rows2); T2.to_csv(OUTD / "Table2.csv", index=False)
md2 = ["**Table 2. Clinical-adjusted prognostic association of CellFactor risk.**", "", "| " + " | ".join(T2.columns) + " |", "|" + "---|" * len(T2.columns)]
for _, r in T2.iterrows(): md2.append("| " + " | ".join(str(v) for v in r.values) + " |")
md2 += ["", "Cox proportional hazards models for overall survival in external cohorts. HR per 1 SD increase in CellFactor risk (six-chain ensemble, standardized within cohort). "
        "Complete cases only. P from the likelihood-ratio test comparing covariates-only with covariates + CellFactor risk. "
        "GSE85916 had no clinical covariates and was not analysed. GSE57495 was adjusted for stage only. The analysis tests adjusted association, not incremental discrimination."]
open(OUTD / "Table2.md", "w").write("\n".join(md2) + "\n")
print("\n".join(md)); print(); print("\n".join(md2))
