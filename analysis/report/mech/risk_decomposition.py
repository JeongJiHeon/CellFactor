#!/usr/bin/env python3
"""CellFactor 위험 분해(보충 표) — 체인 c1–c6, 체인별 코호트 내 z 후 평균. 재학습 없음.
  전체 r = Σ_c π̂_c(α_c + s_c),  s_c = W_c·M̂_c
  조성 고정(state-only) = Σ_c π̄_c(α_c + s_c)        (π̄ = TCGA 학습 평균)
  조성 편차 항 = Σ_c (π̂_c − π̄_c)(α_c + s_c)        (전체 − 조성 고정)
  조성만 = Σ_c π̂_c(α_c + s̄_c)                      (s̄ = TCGA 학습 평균 상태)
외부 평균 = 확정 외부 종점 평균, held-out = TCGA OS. 95% CI = 코호트별 환자 재표집 2000회(seed 0, 성분 공통 표본).
출력 results/mech/decomp/RISK_DECOMPOSITION.{json,csv,md} (+ results_json 미러)"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from lifelines.utils import concordance_index as ci
ROOT = Path(__file__).resolve().parents[3]; OUT = ROOT / "results/mech/decomp"; OUT.mkdir(parents=True, exist_ok=True); OJ = ROOT / "results_json/mech/decomp"; OJ.mkdir(parents=True, exist_ok=True)
CFG = {"breast": ("BREAST", [("SCAN-B", "OS"), ("SCAN-B", "RFi"), ("SCAN-B", "DRFi"), ("SCAN-B", "BCFi"), ("METABRIC", "OS"), ("METABRIC", "RFS")]),
       "liver": ("LIVER", [("GSE14520", "OS"), ("GSE14520", "RFS"), ("GSE76427", "OS"), ("GSE76427", "RFS")]),
       "pancreas": ("PANC", [("GSE62452", "OS"), ("GSE57495", "OS"), ("GSE85916", "OS")])}
PARTS = ["CellFactor (full)", "Composition fixed (state-only)", "Composition deviation term", "Composition only (mean state)"]
z = lambda v: (v - v.mean()) / (v.std() + 1e-12); B = 2000
rows, res = [], {}
for org, (tag, EPS) in CFG.items():
    R = json.load(open(ROOT / f"results_json/report_acand3k/{tag}_ACAND3K_RISKS.json"))
    cohorts = sorted({c for c, _ in EPS}) + ["TCGA-heldout"]; acc = {p: {c: [] for c in cohorts} for p in PARTS}
    for ch in range(1, 7):
        pb = torch.load(ROOT / f"results/hvg_design/run1a/{org}/A_cand3k/c{ch}/pred_bulk.pt", map_location="cpu", weights_only=False)
        es = torch.load(ROOT / f"results/paper_acand3k/ext_states/{org}_c{ch}.pt", weights_only=False)
        W, a = pb["W"].double().numpy(), pb["alpha"].double().numpy(); tr, ho = pb["train_idx"].numpy(), pb["heldout_idx"].numpy()
        pbar = pb["pi"].double().numpy()[tr].mean(0); sbar = (pb["M"].double().numpy()[tr].mean(0) * W).sum(1)
        for c in cohorts:
            pi, M = (pb["pi"].double().numpy()[ho], pb["M"].double().numpy()[ho]) if c == "TCGA-heldout" else (es["PM"][c]["pi"].double().numpy(), es["PM"][c]["M"].double().numpy())
            g = a[None] + (M * W[None]).sum(2)
            for p, r in zip(PARTS, ((pi * g).sum(1), (pbar[None] * g).sum(1), ((pi - pbar[None]) * g).sum(1), (pi * (a + sbar)[None]).sum(1))): acc[p][c].append(z(r))
    ens = {p: {c: np.mean(v, 0) for c, v in d.items()} for p, d in acc.items()}
    END = {f"{c}|{e}": (c, *(lambda d: (np.asarray(d["indices"]), np.asarray(d["time"], float), np.asarray(d["event"], float)))(R[c]["endpoints"][e])) for c, e in EPS}
    END["TCGA-heldout|OS"] = ("TCGA-heldout", *(lambda d: (np.asarray(d["indices"]), np.asarray(d["time"], float), np.asarray(d["event"], float)))(R["TCGA-heldout"]["endpoints"]["OS"]))
    ncoh = {c: len(ens[PARTS[0]][c]) for c in cohorts}

    INV = {}
    for k, (c, ix, t, e) in END.items():
        assert len(np.unique(ix)) == len(ix); inv = -np.ones(ncoh[c], int); inv[ix] = np.arange(len(ix)); INV[k] = inv

    def evaluate(draw):
        POS = {}
        for k, (c, ix, t, e) in END.items():
            pos = np.arange(len(ix)) if draw is None else INV[k][draw[c]]; pos = pos[pos >= 0]
            POS[k] = pos[np.isfinite(t[pos]) & np.isfinite(e[pos])]
        out = {}
        for p in PARTS:
            v = {k: ci(t[POS[k]], -ens[p][c][ix[POS[k]]], e[POS[k]]) for k, (c, ix, t, e) in END.items()}
            out[p] = {"external_mean": float(np.mean([v[f"{c}|{e}"] for c, e in EPS])), "heldout": v["TCGA-heldout|OS"]}
        return out
    pt = evaluate(None); rng = np.random.default_rng(0); bs = [evaluate({c: rng.integers(0, ncoh[c], ncoh[c]) for c in cohorts}) for _ in range(B)]
    res[org] = {}
    for p in PARTS:
        res[org][p] = {}
        for k in ("external_mean", "heldout"):
            arr = np.array([b[p][k] for b in bs]); lo, hi = np.percentile(arr, [2.5, 97.5]); res[org][p][k] = {"point": pt[p][k], "ci95": [float(lo), float(hi)]}
            rows.append({"Cancer": org.capitalize(), "Component": p, "Set": "External mean" if k == "external_mean" else "Held-out (TCGA OS)", "C-index": pt[p][k], "CI low": lo, "CI high": hi})
        if p != PARTS[0]:
            for k in ("external_mean", "heldout"):
                d = np.array([b[PARTS[0]][k] - b[p][k] for b in bs]); res[org][p][f"full_minus_{k}"] = {"diff": pt[PARTS[0]][k] - pt[p][k], "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
    print(org, {p: round(pt[p]["external_mean"], 3) for p in PARTS}, flush=True)
T = pd.DataFrame(rows)
for d in (OUT, OJ): T.to_csv(d / "RISK_DECOMPOSITION.csv", index=False, float_format="%.4f"); json.dump(res, open(d / "RISK_DECOMPOSITION.json", "w"), indent=1)
md = ["# CellFactor risk decomposition (chains c1–c6)", "", "| Cancer | Component | External mean C (95% CI) | Held-out C (95% CI) | Full − component, external (95% CI) |", "|---|---|---|---|---|"]
for org in CFG:
    for p in PARTS:
        r = res[org][p]; f = lambda v: f"{v['point']:.3f} ({v['ci95'][0]:.3f}–{v['ci95'][1]:.3f})"
        dd = r.get("full_minus_external_mean"); md.append(f"| {org.capitalize()} | {p} | {f(r['external_mean'])} | {f(r['heldout'])} | " + (f"{dd['diff']:.3f} ({dd['ci95'][0]:.3f} to {dd['ci95'][1]:.3f})" if dd else "—") + " |")
for d in (OUT, OJ): open(d / "RISK_DECOMPOSITION.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
