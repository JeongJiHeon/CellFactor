#!/usr/bin/env python3
"""Fig. 2f · 보충 S1 의 CellFactor KM 입력을 체인 c1–c6 앙상블로 다시 만든다(유방·간; 췌장은 원래 c1–c6).
앙상블 = 체인 위험(pred_bulk.pt risk_cohorts, TCGA held-out = risk_all[heldout_idx])을 코호트 안에서 z 표준화 후 평균.
항등 점검: 같은 규칙의 앙상블이 RISKS.json(prog2, 2026-10-11 부터 c1–c6) 위험과 일치해야 진행.
종점·필터·상위 20% 규칙 = km_values_acand3k.py. 출력 final_results/04_prognosis/patients/*_patients.csv (이전 12체인 판은 patients_c1-12/ 로 보관), KM_SUMMARY_c1-6.json"""
import json, shutil
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[4]; FR = ROOT / "final_results/04_prognosis"; PAT = FR / "patients"; BK = FR / "patients_c1-12"
exec(compile(open(ROOT / "analysis/report/acand3k/km_values_acand3k.py").read().split("summary = {}")[0].split("os.makedirs(OUT")[0] + "\n", "kmfun", "exec"))
src = open(ROOT / "analysis/report/acand3k/km_values_acand3k.py").read(); ns = {"np": np}
exec(src[src.index("def km("):src.index("summary = {}")], ns); logrank = ns["logrank"]
zf = lambda r: (r - r.mean()) / (r.std() + 1e-12)
def ens(org, chains):
    P = [torch.load(ROOT / f"results/hvg_design/run1a/{org}/A_cand3k/c{c}/pred_bulk.pt", map_location="cpu", weights_only=False) for c in chains]
    out = {k: np.mean([zf(np.asarray(p["risk_cohorts"][k], float)) for p in P], 0) for k in P[0]["risk_cohorts"]}
    out["TCGA-heldout"] = np.mean([zf(np.asarray(p["risk_all"], float)[np.asarray(p["heldout_idx"], int)]) for p in P], 0); return out
if not BK.exists(): shutil.copytree(PAT, BK)
summary = {}
for organ, org, tag in (("Breast", "breast", "BREAST"), ("Liver", "liver", "LIVER"), ("Pancreas", "pancreas", "PANC")):
    R = json.load(open(ROOT / f"results_json/report_acand3k/prog2/{tag}_ACAND3K_RISKS.json"))
    nall = 6   # 2026-10-11: prog2 RISKS 도 c1–c6 (PLAN_ALPHA_FREE, 모든 모델 6체인)
    E_all, E6 = ens(org, range(1, nall + 1)), ens(org, range(1, 7))
    for coh, c in R.items():
        ref = np.asarray(c["risk"], float); assert np.allclose(E_all[coh], ref, atol=1e-5), (org, coh, float(np.abs(E_all[coh] - ref).max()))
        ep = "OS*" if coh == "GSE54236" else ("OS" if "OS" in c["endpoints"] else next(iter(c["endpoints"]))); v = c["endpoints"][ep]
        t, e = np.asarray(v["time"], float), np.asarray(v["event"], float); idx = np.asarray(v.get("indices", range(len(t))), int)
        r = E6[coh]; ri = r[idx] if len(idx) == len(t) else r
        m = np.isfinite(ri) & np.isfinite(t) & np.isfinite(e) & (t > 0); ri, t, e = ri[m], t[m], e[m]
        if len(t) < 20: continue
        hi = (ri >= np.quantile(ri, 0.8)).astype(int); p = logrank(t, e, hi)
        name = f"{organ}_{coh}_{ep}".replace(" ", "").replace("*", "star")
        with open(PAT / f"{name}_patients.csv", "w") as f:
            f.write("time_years,event,risk,high_risk_top20\n")
            for ti, ei, r_, h_ in zip(t, e, ri, hi): f.write(f"{ti/365.25:.6f},{int(ei)},{r_:.6f},{h_}\n")
        summary[name] = {"n": int(len(t)), "events": int(e.sum()), "logrank_p_c1-6": p}
json.dump(summary, open(FR / "KM_SUMMARY_c1-6.json", "w"), indent=1)
for k, v in summary.items(): print(k, v)
