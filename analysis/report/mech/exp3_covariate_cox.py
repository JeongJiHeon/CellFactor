#!/usr/bin/env python3
"""PLAN_MECH §3 — 임상 공변량 보정 Cox. 재학습 없음.
위험 = CellFactor c1–c6 앙상블(체인별 코호트 내 z 후 평균, Fig. 2), 다시 코호트 내 z(1 SD 당 HR).
보조 = 외부 기여 크기 상위 3 세포유형 기여 T_c(체인 평균, z) 를 하나씩.
환자 연결: SCAN-B = SCANB_BUNDLE 순서와 SCANB_CLINICAL_ALIGNED 의 Patient 일치 행만 · METABRIC = METABRIC_sample_ids ·
  GSE14520/GSE76427 = 원 전처리(liver_prep.py) 규칙 재현(GSM 순서), 생존시간 일치 확인 · GSE62452/GSE57495 = panc_stage23.load_ext 규칙 재현.
출력 results/mech/exp3/{COX_ADJ.csv, COX_ADJ.json, COX_ADJ.md} (+ results_json 미러)"""
import csv, gzip, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]; D0 = Path(os.environ.get("CELLFACTOR_DATA", "data"))
OUT = ROOT / "results/mech/exp3"; OUT.mkdir(parents=True, exist_ok=True); OUTJ = ROOT / "results_json/mech/exp3"; OUTJ.mkdir(parents=True, exist_ok=True)
CH = range(1, 7)


def z(x): x = np.asarray(x, float); return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def chain_parts(org, coh):
    out = []
    for c in CH:
        s = torch.load(ROOT / f"results/paper_acand3k/ext_states/{org}_c{c}.pt", weights_only=False)
        pi, M, W, a = (s["PM"][coh]["pi"].double().numpy(), s["PM"][coh]["M"].double().numpy(), s["W"].double().numpy(), s["alpha"].double().numpy())
        T = pi * a[None] + (pi[:, :, None] * W[None] * M).sum(2)   # (n, C) 세포유형 기여
        out.append(T)
    return out


def risk_and_types(org, coh):
    P = chain_parts(org, coh)
    r = np.mean([z(T.sum(1)) for T in P], 0)
    Tm = np.mean([T / T.sum(1).std() for T in P], 0)   # 체인 위험 SD 단위로 맞춘 뒤 평균
    return z(r), Tm


TYPES = {o: list(torch.load(ROOT / f"results/hvg_design/run1a/{o}/A_cand3k/c1/pred_bulk.pt", map_location="cpu", weights_only=False)["cell_types"]) for o in ("breast", "liver", "pancreas")}
TOP3 = lambda o: list(pd.read_csv(ROOT / f"results/mech/exp1/{o}_CELLTYPES.csv").cell_type.values[:3])
import os
ONLY = os.environ.get("COHORTS")


# ---------- 코호트별 (위험, 기여, 시간, 사건, 공변량 DataFrame) ----------
def cohort_scanb():
    b = torch.load(D0 / "external_bulk/scanb_16k/SCANB_BUNDLE.pt", weights_only=False)
    s = pd.read_csv(D0 / "external_bulk/scanb_3233/SCANB_CLINICAL_ALIGNED.csv")
    ok = s["Patient"].values == np.array(b["patient_ids"])
    t, e = np.asarray(b["endpoints"]["OS"]["time"], float), np.asarray(b["endpoints"]["OS"]["event"], float)
    cov = pd.DataFrame({"age": s.age_at_diagnosis, "tumor_size": s.tumor_size,
                        "node_pos": s.lymph_node_status.map(lambda v: 1.0 if str(v).startswith("NodePos") else (0.0 if str(v).startswith("NodeNeg") else np.nan)),
                        "nhg": s.nhg.map({"G1": 1.0, "G2": 2.0, "G3": 3.0}), "er_pos": pd.to_numeric(s.er_status, errors="coerce"),
                        "her2_pos": pd.to_numeric(s.her2_status, errors="coerce"), "pam50": s.pam50_subtype.astype(str)})
    cov.loc[~ok, :] = np.nan
    return "breast", "SCAN-B", t, e, cov, "PAM50 categorical (ref = LumA)"


def cohort_metabric():
    sids = json.load(open(D0 / "external_bulk/metabric_16k/METABRIC_sample_ids.json"))
    c = pd.read_csv(D0 / "external_bulk/metabric_16k/METABRIC_CLINICAL.csv").set_index("patient_id").reindex(sids)
    t = c.OS_MONTHS.values * 30.4375; e = c.OS_STATUS.astype(str).str.startswith("1").astype(float).values; e[pd.isna(c.OS_STATUS).values] = np.nan
    cov = pd.DataFrame({"age": c.AGE_AT_DIAGNOSIS.values, "npi": c.NPI.values,
                        "er_pos": c.ER_IHC.map({"Positve": 1.0, "Positive": 1.0, "Negative": 0.0}).values,
                        "her2_gain": c.HER2_SNP6.map({"GAIN": 1.0, "NEUTRAL": 0.0, "LOSS": 0.0}).values,
                        "pam50": c.CLAUDIN_SUBTYPE.astype(str).replace({"nan": np.nan, "NC": np.nan}).values})
    return "breast", "METABRIC", t, e, cov, "PAM50+claudin-low categorical (ref = LumA)"


def _series(path):
    gsm, ch = None, []
    for line in gzip.open(path, "rt", errors="ignore"):
        if line.startswith("!Sample_geo_accession"): gsm = [x.strip('"\n') for x in line.split("\t")[1:]]
        if line.startswith("!Sample_characteristics_ch1"): ch.append([x.strip('"\n') for x in line.split("\t")[1:]])
    return gsm, ch


def cohort_gse14520():
    gsm, _ = _series(D0 / "organs/liver/bulk/GSE14520/GSE14520-GPL3921_series_matrix.txt.gz")
    sup = list(csv.DictReader(gzip.open(D0 / "organs/liver/bulk/GSE14520/GSE14520_Extra_Supplement.txt.gz", "rt", errors="ignore"), delimiter="\t"))
    surv = {r["Affy_GSM"].strip(): r for r in sup if r["Tissue Type"] == "Tumor" and r["Affy_GSM"].strip().startswith("GSM")
            and r["Survival status"] in ("0", "1") and r["Survival months"] not in ("", "NA")}
    gs = [g for g in gsm if g in surv]; rows = [surv[g] for g in gs]
    L = np.load(ROOT / "results/liver/LIVER_BULK.npz", allow_pickle=True)
    t = np.array([float(r["Survival months"]) * 30.44 for r in rows]); e = np.array([float(r["Survival status"]) for r in rows])
    assert len(t) == len(L["g14520_t"]) and np.allclose(t, L["g14520_t"], atol=0.05) and np.array_equal(e, L["g14520_e"]), "GSE14520 order mismatch"
    tnm = {"I": 1, "II": 2, "IIIA": 3, "IIIB": 3, "IIIC": 3, "III": 3, "IV": 4}
    cov = pd.DataFrame({"age": [pd.to_numeric(r["Age"], errors="coerce") for r in rows], "male": [1.0 if r["Gender"] == "M" else (0.0 if r["Gender"] == "F" else np.nan) for r in rows],
                        "tnm": [tnm.get(r["TNM staging"].strip(), np.nan) for r in rows],
                        "size_gt5cm": [1.0 if r["Main Tumor Size (>/<=5 cm)"] == "large" else (0.0 if r["Main Tumor Size (>/<=5 cm)"] == "small" else np.nan) for r in rows],
                        "afp_gt300": [1.0 if r["AFP (>/<=300ng/ml)"] == "high" else (0.0 if r["AFP (>/<=300ng/ml)"] == "low" else np.nan) for r in rows],
                        "cirrhosis": [1.0 if r["Cirrhosis"] == "Y" else (0.0 if r["Cirrhosis"] == "N" else np.nan) for r in rows]})
    return "liver", "GSE14520", t, e, cov, "TNM ordinal I–III"


def cohort_gse76427():
    gsm, ch = _series(D0 / "organs/liver/bulk/GSE76427/GSE76427_series_matrix.txt.gz")
    def field(name):
        for v in ch:
            if v and v[0].split(":")[0].strip() == name: return [x.split(":", 1)[1].strip() if ":" in x else "" for x in v]
    tis, eo, to = field("tissue"), field("event_os"), field("duryears_os")
    keep = [i for i in range(len(gsm)) if tis and "tumor" in tis[i].lower() and eo[i] in ("0", "1") and to[i] not in ("", "NA")]
    L = np.load(ROOT / "results/liver/LIVER_BULK.npz", allow_pickle=True)
    t = np.array([float(to[i]) * 365.25 for i in keep]); e = np.array([float(eo[i]) for i in keep])
    assert len(t) == len(L["g76427_t"]) and np.allclose(t, L["g76427_t"], atol=0.05), "GSE76427 order mismatch"
    age, sex, bclc, tnm = field("age (years)"), field("gender (1=m, 2=f)"), field("bclc_staging"), field("tnm_staging_clinical")
    tm = {"I": 1, "II": 2, "IIIA": 3, "IIIB": 3, "IIIC": 3, "III": 3, "IV": 4, "IVA": 4, "IVB": 4}
    cov = pd.DataFrame({"age": [pd.to_numeric(age[i], errors="coerce") for i in keep], "male": [1.0 if sex[i] == "1" else (0.0 if sex[i] == "2" else np.nan) for i in keep],
                        "bclc": [{"0": 0, "A": 1, "B": 2, "C": 3, "D": 4}.get(bclc[i], np.nan) for i in keep], "tnm": [tm.get(tnm[i], np.nan) for i in keep]})
    return "liver", "GSE76427", t, e, cov, "BCLC 0–C and TNM ordinal"


def cohort_panc(acc):
    sys.path.insert(0, str(ROOT / "code/lib")); import panc_stage23 as P
    gpl, tk, unit, ek, dead = P.EXT[acc]; gsm, src, per, probes, X = P.series(acc)
    tum = np.array([i for i, s in enumerate(src) if any(w in s.lower() for w in ("tumor", "tumour", "carcinoma", "pdac", "cancer"))])
    if len(tum) == 0: tum = np.arange(len(gsm))
    t = np.array([P.num(x) for x in P.field(per, tk)])[tum] * unit; ev = P.field(per, ek)
    e = (np.array([1.0 if dead.lower() in str(v).lower() else (0.0 if v else np.nan) for v in ev]) if dead else np.array([P.num(x) for x in ev]))[tum]
    ok = np.isfinite(t) & np.isfinite(e) & (t > 0)
    def fld(name):
        v = P.field(per, name); return None if v is None else np.asarray(v, dtype=object)[tum][ok]
    st = {"I": 1, "IA": 1, "IB": 1, "II": 2, "IIA": 2, "IIB": 2, "III": 3, "IV": 4}
    stage = fld("stage"); grading = fld("grading")
    def stg(x):   # 'IIB' 또는 '2B' 형식
        x = str(x).strip().upper()
        if x[:1].isdigit(): return float(x[0])
        return st.get(x, np.nan)
    cov = pd.DataFrame({"stage": [stg(s) if s is not None else np.nan for s in (stage if stage is not None else [None] * ok.sum())]})
    if grading is not None: cov["grade"] = [pd.to_numeric(str(g).upper().replace("G", "").strip(), errors="coerce") for g in grading]
    cov = cov.loc[:, cov.notna().any()]   # 값이 없는 항목(GSE57495 grade) 제외
    return "pancreas", acc, t[ok], e[ok], cov, "stage ordinal I–IV" + (" + grade" if "grade" in cov else "")


def fit(df, cols, strata=None):
    for pen in (0.0, 0.01, 0.1):
        try:
            m = CoxPHFitter(penalizer=pen); m.fit(df[cols + ["T", "E"]], "T", "E"); return m, pen
        except Exception: continue
    raise RuntimeError("cox failed")


rows, meta = [], {}
for fn in (cohort_scanb, cohort_metabric, cohort_gse14520, cohort_gse76427, lambda: cohort_panc("GSE62452"), lambda: cohort_panc("GSE57495")):
    org, coh, t, e, cov, note = fn()
    if ONLY and coh not in ONLY.split(","): continue
    r, Tm = risk_and_types(org, coh)
    assert len(r) == len(t), (coh, len(r), len(t))
    df = cov.copy(); df["T"], df["E"], df["risk"] = t, e, r
    for nm in TOP3(org): df[f"T_{nm}"] = z(Tm[:, TYPES[org].index(nm)])
    cat = [c for c in df.columns if df[c].dtype == object]
    num = [c for c in cov.columns if c not in cat]
    df = df.dropna(subset=num + cat + ["T", "E"]); df = df[df["T"] > 0]
    for c in cat:   # 범주 → 더미(기준 = LumA, 없으면 최빈)
        ref = "LumA" if "LumA" in set(df[c]) else df[c].mode()[0]
        dm = pd.get_dummies(df[c], prefix=c, dtype=float).drop(columns=f"{c}_{ref}"); dm = dm.loc[:, dm.sum() >= 5]
        df = pd.concat([df.drop(columns=c), dm], axis=1)
    covcols = [c for c in df.columns if c not in ("T", "E", "risk") and not c.startswith("T_")]
    covcols = [c for c in covcols if df[c].std() > 0]
    m0, p0 = fit(df, covcols); m1, p1 = fit(df, covcols + ["risk"]); mr, _ = fit(df, ["risk"])
    lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_); p_lr = float(chi2.sf(lr, 1))
    hr, lo, hi = (float(np.exp(m1.params_["risk"])), *[float(np.exp(v)) for v in m1.confidence_intervals_.loc["risk"].values])
    c0 = concordance_index(df["T"], -m0.predict_partial_hazard(df), df["E"]); c1 = concordance_index(df["T"], -m1.predict_partial_hazard(df), df["E"])
    rec = {"organ": org, "cohort": coh, "n": int(len(df)), "events": int(df.E.sum()), "covariates": ", ".join(covcols), "note": note, "penalizer": p1,
           "HR_unadj": float(np.exp(mr.params_["risk"])), "HR_adj": hr, "HR_lo": lo, "HR_hi": hi, "p_wald": float(m1.summary.loc["risk", "p"]), "p_LR": p_lr,
           "C_cov_insample": float(c0), "C_cov_plus_risk_insample": float(c1),
           "verdict": "independent of covariates" if lo > 1 else "not shown"}
    for nm in TOP3(org):   # 보조: 세포유형 기여 하나씩
        mt, _ = fit(df, covcols + [f"T_{nm}"]); ci_ = mt.confidence_intervals_.loc[f"T_{nm}"].values
        rec[f"type::{nm}"] = {"HR": float(np.exp(mt.params_[f"T_{nm}"])), "lo": float(np.exp(ci_[0])), "hi": float(np.exp(ci_[1])), "p": float(mt.summary.loc[f"T_{nm}", "p"])}
    rows.append(rec); print(json.dumps(rec, indent=1), flush=True)

flat = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("type::")} for r in rows])
for d in (OUT, OUTJ):
    flat.to_csv(d / "COX_ADJ.csv", index=False, float_format="%.4g"); json.dump(rows, open(d / "COX_ADJ.json", "w"), indent=1)
md = ["# 임상 공변량 보정 Cox (PLAN_MECH §3)", "", "위험 = CellFactor c1–c6 앙상블, 코호트 내 1 SD 당 HR. 완전 사례. C 는 표본 내 값(낙관적).", "",
      "| 장기 | 코호트 | n (사건) | 공변량 | HR 보정 전 | HR 보정 후 (95% CI) | LR p | C 공변량 → +위험 | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['organ']} | {r['cohort']} | {r['n']} ({r['events']}) | {r['covariates']} | {r['HR_unadj']:.2f} | {r['HR_adj']:.2f} ({r['HR_lo']:.2f}–{r['HR_hi']:.2f}) | {r['p_LR']:.2g} | "
              f"{r['C_cov_insample']:.3f} → {r['C_cov_plus_risk_insample']:.3f} | {r['verdict']} |")
md += ["", "## 보조: 세포유형 기여(외부 기여 크기 상위 3) 하나씩, 공변량 보정", "", "| 코호트 | 세포유형 | HR (95% CI) | p |", "|---|---|---|---|"]
for r in rows:
    for k, v in r.items():
        if k.startswith("type::"): md.append(f"| {r['cohort']} | {k[6:]} | {v['HR']:.2f} ({v['lo']:.2f}–{v['hi']:.2f}) | {v['p']:.2g} |")
for d in (OUT, OUTJ): open(d / "COX_ADJ.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
