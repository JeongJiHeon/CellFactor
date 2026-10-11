#!/usr/bin/env python3
"""PLAN_MECH §1 — 기여 충실도(deletion). 재학습 없음, 저장 산출만 읽음.
위험 r = Σ_c π̂_c(α_c + W_c·M̂_c), 체인 c1–c6, 앙상블 = 체인별 코호트 내 z 후 평균(Fig. 2).
삭제 = 기여를 코호트 평균으로 대체(주) · 환자 간 순열 20회 평균(보조). ΔC = C(전체) − C(삭제), 확정 외부 종점 평균.
원천 results/paper_acand3k/ext_states/{organ}_c*.pt, results_json/report_acand3k/{BREAST,LIVER,PANC}_ACAND3K_RISKS.json(종점·indices),
     results/hvg_design/run1a/{organ}/A_cand3k/c1/pred_bulk.pt(유형 이름)
출력 results/mech/exp1/{organ}_{PROGRAMS,CELLTYPES,SUMMARY}.{csv,json} (+ results_json 미러)"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, torch
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[3]; OUT = ROOT / "results/mech/exp1"; OUT.mkdir(parents=True, exist_ok=True)
OUTJ = ROOT / "results_json/mech/exp1"; OUTJ.mkdir(parents=True, exist_ok=True)
CH = range(1, 7); RNG = np.random.default_rng(0)
CFG = {"breast": ("BREAST", ["SCAN-B|OS", "SCAN-B|RFi", "SCAN-B|DRFi", "SCAN-B|BCFi", "METABRIC|OS", "METABRIC|RFS"], 0.6175),
       "liver": ("LIVER", ["GSE14520|OS", "GSE14520|RFS", "GSE76427|OS", "GSE76427|RFS"], 0.6000),
       "pancreas": ("PANC", ["GSE62452|OS", "GSE57495|OS", "GSE85916|OS"], 0.6213)}
ORGS = os.environ.get("ORGS", "breast,liver,pancreas").split(",")


def cidx(r, t, e): return concordance_index(t, -r, e)


def z(x): return (x - x.mean()) / (x.std() + 1e-12)


for org in ORGS:
    tag, EPS, c_ref = CFG[org]
    R = json.load(open(ROOT / f"results_json/report_acand3k/{tag}_ACAND3K_RISKS.json"))
    pb = torch.load(ROOT / f"results/hvg_design/run1a/{org}/A_cand3k/c1/pred_bulk.pt", map_location="cpu", weights_only=False)
    types = list(pb["cell_types"])
    S = [torch.load(ROOT / f"results/paper_acand3k/ext_states/{org}_c{c}.pt", weights_only=False) for c in CH]
    progs = list(S[0]["programs"]); K = len(progs); C = len(types)
    cohorts = sorted({e.split("|")[0] for e in EPS})
    # 종점: (cohort, idx, t, e) — 비유한 시간 제거
    END = {}
    for ep in EPS:
        coh, en = ep.split("|"); d = R[coh]["endpoints"][en]
        idx, t, e = np.asarray(d["indices"]), np.asarray(d["time"], float), np.asarray(d["event"], float)
        ok = np.isfinite(t) & np.isfinite(e); END[ep] = (coh, idx[ok], t[ok], e[ok])
    # 체인별 성분: alpha 항 A(n,C), 상태 항 B(n,C,K)
    PARTS = {}
    for coh in cohorts:
        P = []
        for s in S:
            pi, M, W, a = (s["PM"][coh]["pi"].double().numpy(), s["PM"][coh]["M"].double().numpy(), s["W"].double().numpy(), s["alpha"].double().numpy())
            P.append((pi * a[None], pi[:, :, None] * W[None] * M))
        PARTS[coh] = P

    def ens_risk(coh, mod=None):
        """체인별 위험(선택적으로 수정) → 코호트 내 z → 평균"""
        out = []
        for ch, (A, B) in enumerate(PARTS[coh]):
            r = A.sum(1) + B.sum((1, 2))
            if mod is not None: r = mod(ch, A, B, r)
            out.append(z(r))
        return np.mean(out, 0)

    def mean_c(mod=None, boot=None):
        rs = {coh: ens_risk(coh, mod) for coh in cohorts}; v = []
        for ep, (coh, idx, t, e) in END.items():
            ii = np.arange(len(idx)) if boot is None else boot[ep]
            v.append(cidx(rs[coh][idx][ii], t[ii], e[ii]))
        return float(np.mean(v))

    C_full = mean_c()
    assert abs(C_full - c_ref) < 1e-3, f"{org} identity {C_full:.4f} vs {c_ref}"
    print(f"[{org}] identity C {C_full:.4f} (ref {c_ref})", flush=True)

    # 기여 크기: 체인별·코호트별 SD(성분)/SD(위험), 코호트 평균 → 체인 평균
    def magnitude(fn):
        vals = []
        for ch in range(len(S)):
            vals.append(np.mean([fn(*PARTS[coh][ch]).std() / (PARTS[coh][ch][0].sum(1) + PARTS[coh][ch][1].sum((1, 2))).std() for coh in cohorts]))
        return float(np.mean(vals))
    prog_comp = lambda k: (lambda A, B: B[:, :, k].sum(1))
    type_comp = lambda c: (lambda A, B: A[:, c] + B[:, c, :].sum(1))

    def del_mean(compf):
        return lambda ch, A, B, r: r - compf(A, B) + compf(A, B).mean()

    def del_perm(compf, seed):
        def f(ch, A, B, r):
            x = compf(A, B); return r - x + x[np.random.default_rng(seed + 1000 * ch).permutation(len(x))]
        return f

    rows = []
    for k, p in enumerate(progs):
        cf = prog_comp(k)
        dmean = C_full - mean_c(del_mean(cf))
        dperm = C_full - np.mean([mean_c(del_perm(cf, s)) for s in range(20)])
        rows.append({"program": p, "named": not p.startswith(("free", "dim")), "magnitude": magnitude(cf), "dC_mean": dmean, "dC_perm": dperm})
    PG = pd.DataFrame(rows).sort_values("magnitude", ascending=False)
    trow = []
    for c, tname in enumerate(types):
        cf = type_comp(c)
        trow.append({"cell_type": tname, "magnitude": magnitude(cf), "dC_mean": C_full - mean_c(del_mean(cf)),
                     "dC_perm": C_full - np.mean([mean_c(del_perm(cf, s)) for s in range(20)])})
    TP = pd.DataFrame(trow).sort_values("magnitude", ascending=False)

    # F1: Spearman(크기, ΔC) + 축 순열 귀무
    from scipy.stats import spearmanr
    rho = spearmanr(PG.magnitude, PG.dC_mean).correlation
    null = np.array([spearmanr(PG.magnitude.values, RNG.permutation(PG.dC_mean.values)).correlation for _ in range(2000)])
    p_f1 = float((np.sum(null >= rho) + 1) / (len(null) + 1))
    rho_perm = spearmanr(PG.magnitude, PG.dC_perm).correlation
    # F2: 상위 5축 동시 삭제 vs 무작위 5축
    top5 = [progs.index(p) for p in PG.program.values[:5]]
    multi = lambda ks: (lambda A, B: B[:, :, ks].sum((1, 2)))
    d_top5 = C_full - mean_c(del_mean(multi(top5)))
    rnd = np.array([C_full - mean_c(del_mean(multi(list(RNG.choice(K, 5, replace=False))))) for _ in range(1000)])
    q95 = float(np.quantile(rnd, 0.95)); p_f2 = float((np.sum(rnd >= d_top5) + 1) / (len(rnd) + 1))
    # bootstrap CI: 상위 5축 삭제, 최대 기여 세포유형 삭제
    top_t = types.index(TP.cell_type.values[0])
    bt5, btt = [], []
    for b in range(1000):
        boot = {ep: RNG.integers(0, len(v[1]), len(v[1])) for ep, v in END.items()}
        cf_ = mean_c(boot=boot)
        bt5.append(cf_ - mean_c(del_mean(multi(top5)), boot)); btt.append(cf_ - mean_c(del_mean(type_comp(top_t)), boot))
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    rho_t = spearmanr(TP.magnitude, TP.dC_mean).correlation
    summ = {"organ": org, "C_full": C_full, "n_programs": K, "n_named": int(PG.named.sum()), "cell_types": types,
            "F1": {"spearman_magnitude_vs_dC_mean": float(rho), "p_axis_perm_one_sided": p_f1, "spearman_vs_dC_perm": float(rho_perm),
                   "verdict": "magnitude reflects predictive contribution" if (rho > 0 and p_f1 < 0.05) else "not shown"},
            "F2": {"top5": list(PG.program.values[:5]), "dC_top5": d_top5, "dC_top5_ci95": ci(bt5), "random5_mean": float(rnd.mean()), "random5_q95": q95, "p": p_f2,
                   "verdict": "top contributors needed for prediction" if d_top5 > q95 else "not shown"},
            "F3_celltype": {"spearman": float(rho_t), "top_type": TP.cell_type.values[0], "dC_top_type": float(TP.dC_mean.values[0]), "dC_top_type_ci95": ci(btt)}}
    for df, nm in ((PG, "PROGRAMS"), (TP, "CELLTYPES")):
        df.to_csv(OUT / f"{org}_{nm}.csv", index=False, float_format="%.5f"); df.to_csv(OUTJ / f"{org}_{nm}.csv", index=False, float_format="%.5f")
    for d in (OUT, OUTJ): json.dump(summ, open(d / f"{org}_SUMMARY.json", "w"), indent=1)
    np.savez(OUT / f"{org}_DRAWS.npz", random5=rnd, boot_top5=np.asarray(bt5), boot_toptype=np.asarray(btt))   # 그림용(6c)
    print(json.dumps(summ, indent=1), flush=True)
    print(TP.round(4).to_string(), flush=True); print(PG.head(10).round(4).to_string(), flush=True)
