#!/usr/bin/env python3
"""PLAN_MECH 사후 분석 H4s — TP53 변이와 악성 증식 기여: 조성 가중(π̂·w·M̂) vs 상태 단독(w·M̂), π̂_악성·(유방) PAM50 보정 OLS.
출력 results/mech/exp2/{TP53_STATE.json, TP53_STATE.md, TP53_PATIENTS.csv} (+ results_json 미러)"""
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "analysis/report/mech"))
PH = Path(os.path.join(os.environ.get("CELLFACTOR_DATA", "data"), "tcga_pheno"))
OUT = ROOT / "results/mech/exp2"; OUTJ = ROOT / "results_json/mech/exp2"
SPEC = {"breast": ("brca", "Cancer Epithelial"), "liver": ("lihc", "Malignant cells"), "pancreas": ("paad", "Epithelial")}
PROLIF = ["CS_PROLIFERATION", "HM_E2F_Targets", "HM_G2_M_Checkpoint"]
import sys as _sys; _sys.path.insert(0, str(ROOT)); import train as _T
def bulk_prolif(org):   # 사후 기준선(2026-10-10): 세 증식 프로그램 유전자의 표준화 bulk 발현 평균(프로그램별 z 후 평균), 분해 없음
    cfg = _T.ORGANS[org]; b = _T.load(cfg["registry"]); g = {str(x): i for i, x in enumerate(b["gene_names"])}; dct = json.load(open(ROOT / cfg["programs"])); x = b["bulk_x"].double().numpy()
    sc = [x[:, [g[t] for t in dct[k] if t in g]].mean(1) for k in PROLIF]; return np.mean([(v - v.mean()) / v.std() for v in sc], 0)
TH = pd.read_excel(PH / "thorsson_mmc2.xlsx").set_index("TCGA Participant Barcode"); MU = json.load(open(PH / "cbio_mutations_TP53_CTNNB1_KRAS.json"))


def ids(org):
    if org == "breast":   # 2026-10-10: CCE 학습에 쓴 정확 TCGA ID(1,089행 전부; 기존 레지스트리 ID 717개와 전부 일치). 이전 = 레지스트리 생존 일치 복구 ID(372 결측)
        return [str(x) for x in torch.load(ROOT / "results/stage2_redesign/run_20260828_p16k_stage2a_100k/BREAST_2A_SUPERVISION.pt", weights_only=False)["patient_ids"]]
    p = {"liver": ROOT / "results/liver/LIVER_STAGE1_DATA.pt", "pancreas": ROOT / "sandbox6/results/pancreas/PANC_STAGE1_DATA.pt"}[org]
    return [str(x) for x in torch.load(p, map_location="cpu", weights_only=False, mmap=True)["bulk_patient_ids"]]


res, recs = {}, []
for org, (study, mal) in SPEC.items():
    PB = [torch.load(ROOT / f"results/hvg_design/run1a/{org}/A_cand3k/c{c}/pred_bulk.pt", map_location="cpu", weights_only=False) for c in range(1, 7)]
    types, progs = list(PB[0]["cell_types"]), list(PB[0]["programs"]); im = types.index(mal); kp = [progs.index(p) for p in PROLIF]
    cw, st = [], []
    for p in PB:
        P, M, W, a = p["pi"].double().numpy(), p["M"].double().numpy(), p["W"].double().numpy(), p["alpha"].double().numpy()
        s = ((P * a[None]).sum(1) + (P[:, :, None] * W[None] * M).sum((1, 2))).std()
        sv = (W[im, kp][None] * M[:, im, kp]).sum(1) / s
        st.append(sv); cw.append(P[:, im] * sv)
    cw, st = np.mean(cw, 0), np.mean(st, 0); pim = np.mean([p["pi"][:, im].double().numpy() for p in PB], 0)
    pats = [x[:12] for x in ids(org)]; n = len(pats); held = np.zeros(n, bool); held[PB[0]["heldout_idx"].numpy()] = True
    seq = {s[:12] for s in MU[study]["sequenced"]}; mut = {m["sample"][:12] for m in MU[study]["mutations"] if m["gene"] == "TP53"}
    tp = np.array([(1.0 if x in mut else 0.0) if x in seq else np.nan for x in pats])
    sub = TH.reindex(pats)["TCGA Subtype"].astype(str).values
    df = pd.DataFrame({"organ": org, "patient": pats, "heldout": held, "TP53_mut": tp, "comp_weighted": cw, "state_only": st, "pi_malignant": pim, "subtype": sub, "bulk_prolif_score": bulk_prolif(org)})
    recs.append(df); res[org] = {}
    for scope, m in (("all", np.ones(n, bool)), ("heldout", held)):
        d = df[m & np.isfinite(tp)]; r = {"n_mut": int(d.TP53_mut.sum()), "n_wt": int((d.TP53_mut == 0).sum())}
        for v in ("comp_weighted", "state_only", "bulk_prolif_score"):
            a_, b_ = d[d.TP53_mut == 1][v], d[d.TP53_mut == 0][v]; u = mannwhitneyu(a_, b_, alternative="greater")
            r[v] = {"median_diff": float(a_.median() - b_.median()), "auc": float(u.statistic / (len(a_) * len(b_))), "p_one_sided": float(u.pvalue)}
        X = pd.DataFrame({"TP53_mut": d.TP53_mut, "pi_malignant": d.pi_malignant})
        if org == "breast":
            sb = d.subtype.where(d.subtype.str.startswith("BRCA."), np.nan)
            ok = sb.notna(); X = pd.concat([X[ok], pd.get_dummies(sb[ok], drop_first=True, dtype=float)], axis=1); yv = d.state_only[ok]
        else: yv = d.state_only
        fit = sm.OLS(yv.values, sm.add_constant(X.values.astype(float))).fit(cov_type="HC3")
        r["ols_state_only"] = {"covariates": list(X.columns), "n": int(len(yv)), "beta_TP53": float(fit.params[1]), "ci": [float(v) for v in fit.conf_int()[1]], "p_two_sided": float(fit.pvalues[1])}
        res[org][scope] = r
    print(org, json.dumps(res[org], indent=1), flush=True)
P = pd.concat(recs); P.to_csv(OUT / "TP53_PATIENTS.csv", index=False, float_format="%.5g")
for d in (OUT, OUTJ): json.dump(res, open(d / "TP53_STATE.json", "w"), indent=1)
md = ["# TP53 × 악성 증식 기여 — 조성 가중 vs 상태 단독 (사후 분석 H4s)", "", "AUC = P(변이군 값 > 야생형 값). OLS = 상태 단독 ~ TP53 + π̂_악성 (+ 유방 PAM50), HC3.", "",
      "| 장기 | 범위 | 변이/야생형 | 조성 가중 AUC (p) | 상태 단독 AUC (p) | bulk 증식 점수 AUC (p, 사후 기준선) | OLS β_TP53 (95% CI), p |", "|---|---|---|---|---|---|---|"]
for org, v in res.items():
    for sc, r in v.items():
        o = r["ols_state_only"]
        md.append(f"| {org} | {sc} | {r['n_mut']}/{r['n_wt']} | {r['comp_weighted']['auc']:.2f} ({r['comp_weighted']['p_one_sided']:.2g}) | {r['state_only']['auc']:.2f} ({r['state_only']['p_one_sided']:.2g}) | {r['bulk_prolif_score']['auc']:.2f} ({r['bulk_prolif_score']['p_one_sided']:.2g}) | "
                  f"{o['beta_TP53']:.3f} ({o['ci'][0]:.3f}–{o['ci'][1]:.3f}), {o['p_two_sided']:.2g} |")
for d in (OUT, OUTJ): open(d / "TP53_STATE.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
