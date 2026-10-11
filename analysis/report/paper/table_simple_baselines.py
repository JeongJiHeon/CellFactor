#!/usr/bin/env python3
"""리뷰 대응 — 단순 기준선 C-index 표(저장 산출만 읽음, 재학습 없음).
열(모델)과 원천
  현행(CellFactor = A_cand3k c1–c6 와 같은 시기)
    CellFactor                 results_json/report_acand3k/{LIVER,BREAST,PANC}_ACAND3K_RISKS.json (환자별 위험도)
    Bulk Cox                   results/paper_acand3k/PAPER_NUMBERS.json Fig2 "Direct bulk Cox" (ridge Cox, 벌크 전체 유전자)
    Composition-only Cox       같은 파일 "Composition only" (π̂ 만 넣고 Cox 재적합)
  이전 파이프라인(9월, 배포 E2″ 시기 — CellFactor 와 같은 2A·같은 코호트, H/W 는 이전 모델)
    Bulk program-score Cox     results/w_exp/{EXP1_W_PERMUTATION_bulk, ORGAN_EXP123_*_e2pp}.json exp1 raw_signature_cox (분해 없음: 벌크에서 프로그램 점수 → ridge Cox)
    π̂ + bulk program score     results/report/MSUB_{org}.json pi_bulkscore (조성 + 벌크 프로그램 점수, 곱셈 가중 없는 가산 Cox)
    π̂, state removed (refit)   같은 파일 Mbar_refit (M̂ 을 학습 평균으로 치환, 재적합)
    State only (W fixed)       results/w_exp/{EXP2_COUNTERFACTUAL_bulk, ORGAN_EXP123_*_e2pp}.json exp2 state_fixedpi (π̂ 를 학습 평균으로 고정)
    Composition only (W fixed) 같은 파일 exp2 comp (M̂ 을 학습 평균으로 고정)
    Earlier model (full)       같은 파일 exp2 full — 위 두 반사실의 기준
짝 bootstrap: 환자별 위험도가 저장된 팔(MSUB npz)만 CellFactor 와 같은 환자 짝(시간·사건 벡터 일치 확인), 2000회, 외부 평균은 종점별 독립 재표집 후 평균.
출력 final_results/02_cindex/simple_baselines/{TABLE_simple_baselines.csv,.md, PAIRED_BOOT.csv, SOURCES.json}"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "final_results/02_cindex/simple_baselines"; OUT.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(ROOT / p))
B = 2000; RNG = np.random.default_rng(0)
ORG = {
    "breast": dict(risk="results_json/report_acand3k/BREAST_ACAND3K_RISKS.json", msub="results/report/MSUB_breast",
                   exp1=("results/w_exp/EXP1_W_PERMUTATION_bulk.json", None), exp2=("results/w_exp/EXP2_COUNTERFACTUAL_bulk.json", None),
                   held="TCGA-heldout|OS", ext=["METABRIC|OS", "METABRIC|RFS", "SCAN-B|BCFi", "SCAN-B|DRFi", "SCAN-B|OS", "SCAN-B|RFi"]),
    "liver": dict(risk="results_json/report_acand3k/LIVER_ACAND3K_RISKS.json", msub="results/report/MSUB_liver",
                  exp1=("results/w_exp/ORGAN_EXP123_liver_e2pp.json", "exp1"), exp2=("results/w_exp/ORGAN_EXP123_liver_e2pp.json", "exp2"),
                  held="TCGA-heldout|OS", ext=["GSE14520|OS", "GSE14520|RFS", "GSE76427|OS", "GSE76427|RFS"]),
    "pancreas": dict(risk="results_json/report_acand3k/PANC_ACAND3K_RISKS.json", msub="results/report/MSUB_pancreas",
                     exp1=("results/w_exp/ORGAN_EXP123_pancreas_e2pp.json", "exp1"), exp2=("results/w_exp/ORGAN_EXP123_pancreas_e2pp.json", "exp2"),
                     held="TCGA-heldout|OS", ext=["GSE57495|OS", "GSE62452|OS", "GSE85916|OS"]),
}
COLS = [("CellFactor", "current"), ("Bulk Cox", "current"), ("Composition-only Cox", "current"),
        ("Bulk program-score Cox", "earlier"), ("Composition + bulk program score (additive Cox)", "earlier"),
        ("Composition, state removed (refit)", "earlier"), ("State only (composition fixed, W fixed)", "earlier"),
        ("Composition only (state fixed, W fixed)", "earlier"), ("Earlier model, full", "earlier")]
MSUB_ARM = {"Composition + bulk program score (additive Cox)": "pi_bulkscore", "Composition, state removed (refit)": "Mbar_refit"}


def cidx(r, t, e):
    r, t, e = map(np.asarray, (r, t, e)); num = den = 0.0
    for i in np.where(e > 0)[0]:
        m = t > t[i]; den += m.sum(); num += (r[i] > r[m]).sum() + 0.5 * (r[i] == r[m]).sum()
    return num / den


def cidx_fast(r, t, e):
    r, t, e = map(np.asarray, (r, t, e)); ev = e > 0
    M = (t[None, :] > t[:, None]) & ev[:, None]
    return float(((r[:, None] > r[None, :]) * M).sum() + 0.5 * ((r[:, None] == r[None, :]) * M).sum()) / M.sum()


k_ = lambda k: k.replace("TCGA_heldout", "TCGA-heldout")
rows, boot_rows, src = [], [], {}
for o, cfg in ORG.items():
    R = J(cfg["risk"]); PN = J("results/paper_acand3k/PAPER_NUMBERS.json")["Fig2"][o]["point"]
    ms = J(cfg["msub"] + ".json"); mz = np.load(ROOT / (cfg["msub"] + ".npz"), allow_pickle=True)
    e1 = J(cfg["exp1"][0]); e1 = (e1[cfg["exp1"][1]] if cfg["exp1"][1] else e1)["variants"]["raw_signature_cox"]
    e2 = J(cfg["exp2"][0])
    if cfg["exp2"][1]: e2 = e2[cfg["exp2"][1]]; get2 = lambda v, ep: {k_(k): x for k, x in e2[v].items()}.get(ep)
    else: get2 = lambda v, ep: {k_(k): x[v] for k, x in e2["variants"].items()}.get(ep)
    e1 = {k_(k): v for k, v in e1.items()}
    eps = [cfg["held"]] + cfg["ext"]; cf_r, ms_r = {}, {}
    for ep in eps:
        coh, end = ep.split("|"); d = R[coh]; en = d["endpoints"][end]; r = np.asarray(d["risk"])[np.asarray(en["indices"])]
        t, e = np.asarray(en["time"]), np.asarray(en["event"]); cf_r[ep] = (r, t, e)
        me = ms["endpoints"][ep]; same = np.allclose(me["time"], t) and np.allclose(me["event"], e)
        for col, arm in MSUB_ARM.items():
            rr = mz[f"{arm}__{coh}"]; rr = rr.mean(0) if rr.ndim == 2 else rr; ms_r[(col, ep)] = rr[np.asarray(me["indices"])] if same else None
        v = {"CellFactor": cidx_fast(r, t, e),
             "Bulk Cox": PN["Direct bulk Cox"]["held"] if ep == cfg["held"] else PN["Direct bulk Cox"]["endpoints"][ep],
             "Composition-only Cox": PN["Composition only"]["held"] if ep == cfg["held"] else PN["Composition only"]["endpoints"][ep],
             "Bulk program-score Cox": e1.get(ep),
             "Composition + bulk program score (additive Cox)": ms["summary"]["pi_bulkscore"].get(ep),
             "Composition, state removed (refit)": ms["summary"]["Mbar_refit"].get(ep),
             "State only (composition fixed, W fixed)": get2("state_fixedpi", ep),
             "Composition only (state fixed, W fixed)": get2("comp", ep),
             "Earlier model, full": get2("full", ep)}
        rows.append({"organ": o, "set": "Held-out" if ep == cfg["held"] else "External", "endpoint": ep, **v})
        if not same: print(f"[경고] {o} {ep}: MSUB 환자 순서 불일치 → 짝 bootstrap 생략")
    df = pd.DataFrame([x for x in rows if x["organ"] == o and x["set"] == "External"])
    rows.append({"organ": o, "set": "External", "endpoint": "Mean of external endpoints", **{c: float(df[c].mean()) for c, _ in COLS}})
    # 짝 bootstrap (CellFactor − 팔)
    for col in MSUB_ARM:
        for scope in ["held", "ext"]:
            sel = [cfg["held"]] if scope == "held" else cfg["ext"]
            if any(ms_r[(col, ep)] is None for ep in sel): continue
            point = np.mean([cidx_fast(*cf_r[ep]) - cidx_fast(ms_r[(col, ep)], *cf_r[ep][1:]) for ep in sel]); bs = []
            for _ in range(B):
                dd = []
                for ep in sel:
                    r, t, e = cf_r[ep]; i = RNG.integers(0, len(t), len(t))
                    dd.append(cidx_fast(r[i], t[i], e[i]) - cidx_fast(ms_r[(col, ep)][i], t[i], e[i]))
                bs.append(np.mean(dd))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            boot_rows.append({"organ": o, "comparison": f"CellFactor − {col}", "scope": "held-out OS" if scope == "held" else "mean of external endpoints",
                              "diff": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                              "verdict": "CellFactor higher" if lo > 0 else ("baseline higher" if hi < 0 else "no difference detected (CI includes 0)")})
    for m in ("Direct bulk Cox", "Composition only"):   # 현행 팔 CI 는 PAPER_NUMBERS 에서 그대로
        cmp = J("results/paper_acand3k/PAPER_NUMBERS.json")["Fig2"][o]["compare_ext_held"][m]
        for sc, lab in (("held", "held-out OS"), ("ext", "mean of external endpoints")):
            p, lo, hi = cmp[sc]
            boot_rows.append({"organ": o, "comparison": f"CellFactor − {'Bulk Cox' if m == 'Direct bulk Cox' else 'Composition-only Cox'}", "scope": lab,
                              "diff": p, "ci_lo": lo, "ci_hi": hi,
                              "verdict": "CellFactor higher" if lo > 0 else ("baseline higher" if hi < 0 else "no difference detected (CI includes 0)")})
    src[o] = {"cellfactor_risk": cfg["risk"], "bulk_cox_and_composition_only": "results/paper_acand3k/PAPER_NUMBERS.json Fig2",
              "msub": cfg["msub"] + ".{json,npz}", "msub_reference_heads": {k: ms.get(k) for k in ("head_prefix", "zsuf", "h_root", "h_seeds", "ckd") if k in ms},
              "exp1_raw_signature_cox": cfg["exp1"][0], "exp1_lambda": e1.get("lambda"), "exp2_counterfactual": cfg["exp2"][0]}

T = pd.DataFrame(rows); T.to_csv(OUT / "TABLE_simple_baselines.csv", index=False, float_format="%.4f")
P = pd.DataFrame(boot_rows); P.to_csv(OUT / "PAIRED_BOOT.csv", index=False, float_format="%.4f")
# CellFactor 항등: final_results 표와 같은 값인지
FT = pd.read_csv(ROOT / "final_results/02_cindex/TABLE_cindex_per_cohort.csv")
src["identity_vs_TABLE_cindex_per_cohort"] = "CellFactor column recomputed from the same risk files; max |diff| reported below"
json.dump(src, open(OUT / "SOURCES.json", "w"), indent=1)

ORGN = {"breast": "Breast", "liver": "Liver", "pancreas": "Pancreas"}
f = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
md = ["# Simple baselines — C-index (review response)", "",
      "No retraining. All values are read from saved runs. Two groups:", "",
      "- **Current** = the same period as CellFactor (A_cand3k chains c1–c6). Bulk Cox and composition-only Cox are the Fig. 2 comparison arms.",
      "- **Earlier** = runs from the September E2″ pipeline. They share CellFactor's composition estimator (2A) and cohorts but use the earlier state estimator H and risk weights W. "
      "They answer the structural question (composition vs state, additive vs multiplicative). They are not CellFactor itself.", "",
      "| Organ | Set | Endpoint | " + " | ".join(f"{c}{' †' if g == 'earlier' else ''}" for c, g in COLS) + " |", "|" + "---|" * (3 + len(COLS))]
for _, r in T.iterrows():
    md.append(f"| {ORGN[r.organ]} | {r.set} | {r.endpoint.replace('|', ' ')} | " + " | ".join((f'**{f(r[c])}**' if r.endpoint.startswith('Mean') else f(r[c])) for c, _ in COLS) + " |")
md += ["", "† Earlier pipeline (E2″ era).", "",
       "The CellFactor column is recomputed from the patient risk files behind Fig. 2 and the paired bootstrap. "
       "It differs from TABLE_cindex_per_cohort (10/4 re-inference) by at most 0.007 per endpoint and 0.001 in the external means.", "",
       "## Model definitions", "",
       "| Model | Input to Cox | Uses deconvolution | Cell-type × state interaction (π·W·M) |", "|---|---|---|---|",
       "| CellFactor | Σ_c π̂_c(α_c + W_c·M̂_c) | yes | yes |",
       "| Bulk Cox | ridge Cox on all bulk genes | no | no |",
       "| Composition-only Cox | π̂ only, Cox refit | yes (composition) | no |",
       "| Bulk program-score Cox | mean expression of each program gene set in bulk, ridge Cox | no | no |",
       "| Composition + bulk program score | π̂·α + bulk program scores·γ, additive Cox | yes (composition) | no |",
       "| Composition, state removed (refit) | M̂ replaced by the training mean, Cox refit | yes | state removed |",
       "| State only (W fixed) | π̂ fixed at the training mean; M̂ varies; α, W unchanged | yes | composition removed |",
       "| Composition only (W fixed) | M̂ fixed at the training mean; π̂ varies; α, W unchanged | yes | state removed |",
       "| Earlier model, full | the earlier model's own risk; reference for the two W-fixed rows | yes | yes |", "",
       "Breast 'Composition only (W fixed)' is constant (0.500) because the earlier breast model fitted α = 0, so composition alone carries no risk.", "",
       "## Paired bootstrap: CellFactor minus baseline", "",
       "Patient bootstrap, 2,000 resamples. External = each endpoint resampled independently, then averaged. Only arms with saved patient-level risks are tested. "
       "Bulk Cox and composition-only CIs are the Fig. 2 values. For breast external cohorts, the earlier runs used a different patient filter and order (e.g. METABRIC 1,980 vs 1,979), "
       "so those pairs were not tested.", "",
       "| Organ | Comparison | Scope | Difference | 95% CI | Verdict |", "|---|---|---|---|---|---|"]
for _, r in P.iterrows():
    md.append(f"| {ORGN[r.organ]} | {r.comparison} | {r.scope} | {r['diff']:.3f} | {r.ci_lo:.3f} to {r.ci_hi:.3f} | {r.verdict} |")
md += ["", "## Not performed", "",
       "- A survival baseline fed with cell-type-specific expression from BayesPrism, StateScope or similar was not run. "
       "BayesPrism was used only to test state recovery (see 06_cell_state), not as a prognostic baseline.",
       "- A 'deconvolution → program scoring of each cell type's deconvolved expression → Cox' arm was not run. "
       "The closest existing arms are 'Bulk program-score Cox' (program scoring without deconvolution) and 'Composition + bulk program score' (composition from deconvolution plus bulk program scores, additive).",
       "- State-only and additive arms were not re-run with CellFactor's own H and W."]
open(OUT / "TABLE_simple_baselines.md", "w").write("\n".join(md) + "\n")

print(T.to_string()); print(P.to_string())
