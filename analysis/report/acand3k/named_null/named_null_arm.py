"""명명 축 문헌 부호 부합 + 순열 귀무 — 1a 팔(체인 pred_bulk.pt 의 π̂·M̂) 판. 정의는 code/eval/ablation_cp/named_axes_arms.py 그대로:
EXPECT(bio_literature_concordance.py) · 공선성 제외 |corr(M̂_ck, π_tumor)|>0.3 · Stage 3 = 장기 규약(유방 α=0 ridge 0.05 600 step · 간·췌장·폐 fit_stage3 seed 0) ·
(a) 체인 6개 평균 W 부합 (b) 체인별 부합 평균±SD (c) 체인 1–3 평균으로 관측·순열 p(R_NULL, rng 11, 생존 라벨 순열 후 재적합).
원 스크립트와 다른 점: H 추론 대신 체인이 저장한 전체 벌크 π̂·M̂(pred_bulk.pt: pi, M, train_idx, time_all, event_all)을 쓴다.
env ORG · ARM · CHAINS(1-6) · DEV · R_NULL(200) · OUT_JSON"""
import os, ast, json, sys, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import Counter
sys.path.insert(0, os.path.join(os.environ.get("CELLFACTOR_PROJECT", "."), "code/lib"))
from evaluate_stage2_redesign import Full, cox_nll, set_seed, fit_stage3
FINAL = Path(os.environ.get("CELLFACTOR_PROJECT", ".")); ORG = os.environ["ORG"]; ARM = os.environ["ARM"]; DEV = torch.device(os.environ.get("DEV", "cuda:1")); assert str(DEV) != "cuda:2"
R_NULL = int(os.environ.get("R_NULL", "200")); c0, c1 = (int(v) for v in os.environ.get("CHAINS", "1-6").split("-")); CH = list(range(c0, c1 + 1)); DEPC = CH[:3]
src = open(FINAL / "analysis/eval/bio_literature_concordance.py").read(); i0 = src.index("EXPECT={"); EXPECT = ast.literal_eval(src[i0 + 7:src.index("\n}\n", i0) + 2])[ORG]
TUMNAME = {"liver": "Malignant cells", "pancreas": "Epithelial", "breast": "Cancer Epithelial"}[ORG]
RUN = FINAL / "results/hvg_design/run1a" / ORG / ARM
PB = {c: torch.load(RUN / f"c{c}" / "pred_bulk.pt", map_location="cpu", weights_only=False) for c in CH}
p0 = PB[CH[0]]; types = list(p0["cell_types"]); PROG = list(p0["programs"]); tum = types.index(TUMNAME)
tr = np.asarray(p0["train_idx"], int); t_all = np.asarray(p0["time_all"], float); e_all = np.asarray(p0["event_all"], float)
ok = np.isfinite(t_all[tr]) & (t_all[tr] > 0) & np.isfinite(e_all[tr]); tr = tr[ok]
for c in CH: assert np.array_equal(np.asarray(PB[c]["train_idx"], int)[ok] if len(np.asarray(PB[c]["train_idx"])) == len(ok) else tr, tr) and list(PB[c]["programs"]) == PROG
T = torch.tensor(t_all[tr], dtype=torch.float32, device=DEV); E = torch.tensor(e_all[tr], dtype=torch.float32, device=DEV)
ALPHA_FREE_BREAST = os.environ.get("ALPHA_FREE", "1") == "1"   # 2026-10-11 유방 배포 CSRF = α 자유(PLAN_ALPHA_FREE). 0 = 이전 α=0 판 재현
def fitW(pi, M, t, e):
    if ORG == "breast" and not ALPHA_FREE_BREAST:
        set_seed(0); m = Full(len(types), M.shape[2]).to(DEV); m.alpha.data.zero_(); m.alpha.requires_grad_(False); opt = torch.optim.Adam([m.W], lr=1e-2, weight_decay=1e-4)
        for _ in range(600):
            loss = cox_nll(m.risk(pi, M), t, e) + 0.05 * m.W.square().sum(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return m.W.detach().cpu()
    return fit_stage3(pi, M, t, e, DEV, seed=0).W.detach().cpu()
PI = {c: torch.as_tensor(PB[c]["pi"]).float()[tr].to(DEV) for c in CH}; MS = {c: torch.as_tensor(PB[c]["M"]).float()[tr].to(DEV) for c in CH}
WS = {c: fitW(PI[c], MS[c], T, E) for c in CH}
w_saved = {c: float((WS[c] - torch.as_tensor(PB[c]["W"]).float()).abs().max()) for c in CH}   # 저장 W 와의 차(재적합 항등)
def keepset(pi_tr, M_tr):
    pit = pi_tr[:, tum].cpu().numpy(); kept, rows = [], []
    for ct, ax, sign, why in EXPECT:
        if ct not in types or ax not in PROG: rows.append({"cell_type": ct, "axis": ax, "status": "축 없음"}); continue
        c, k = types.index(ct), PROG.index(ax); coll = float(np.corrcoef(M_tr[:, c, k].cpu().numpy(), pit)[0, 1]); ex = abs(coll) > 0.3
        rows.append({"cell_type": ct, "axis": ax, "expected": "+" if sign > 0 else "−", "corr_with_pi_tumor": round(coll, 3), "excluded_collinear": bool(ex)})
        if not ex: kept.append((c, k, sign))
    return kept, rows
conc = lambda W, kept: float(np.mean([np.sign(float(W[c, k])) == s for c, k, s in kept])) if kept else float("nan")
ms = lambda v: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
per = []
for c in CH: kept, _ = keepset(PI[c], MS[c]); per.append(conc(WS[c], kept))
def agg(sl):
    pim = torch.stack([PI[c] for c in sl]).mean(0); Mm = torch.stack([MS[c] for c in sl]).mean(0); Wm = torch.stack([WS[c] for c in sl]).mean(0); kept, rows = keepset(pim, Mm); return kept, rows, Wm
kept6, rows6, W6 = agg(CH); kept3, rows3, W3 = agg(DEPC); obs3 = conc(W3, kept3)
null = []; rng = np.random.default_rng(11)
for i in range(R_NULL):
    o = torch.as_tensor(rng.permutation(len(tr)), device=DEV); null.append(conc(torch.stack([fitW(PI[c], MS[c], T[o], E[o]) for c in DEPC]).mean(0), kept3))
null = np.asarray(null)
rec = {"arm": ARM, "organ": ORG, "chains": CH, "null_chains": DEPC, "R_null": R_NULL, "n_expectations": len(EXPECT), "refit_vs_saved_W_maxabs": w_saved,
       "concordance_6_meanW": conc(W6, kept6), "n_tested_6": len(kept6), "k_agree_6": int(round(conc(W6, kept6) * len(kept6))) if kept6 else 0,
       "per_chain": ms(per), "per_chain_values": per, "concordance_3": obs3, "n_tested_3": len(kept3), "k_agree_3": int(round(obs3 * len(kept3))) if kept3 else 0,
       "null_mean_3": float(null.mean()), "null_sd_3": float(null.std()), "p_3": float((np.sum(null >= obs3) + 1) / (R_NULL + 1)),
       "per_expectation_6": [dict(r, fitted_W=(round(float(W6[types.index(r["cell_type"]), PROG.index(r["axis"])]), 4) if r.get("expected") else None),
                                  agrees=(bool(np.sign(float(W6[types.index(r["cell_type"]), PROG.index(r["axis"])])) == (1 if r["expected"] == "+" else -1)) if r.get("expected") else None)) for r in rows6]}
out = os.environ["OUT_JSON"]; Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(rec, open(out, "w"), ensure_ascii=False, indent=1)
print(f"[{ORG} {ARM}] 부합 6 {rec['k_agree_6']}/{rec['n_tested_6']} · 체인별 {rec['per_chain']['mean']:.3f}±{rec['per_chain']['sd']:.3f} · 1–3 {rec['k_agree_3']}/{rec['n_tested_3']} p={rec['p_3']:.3f} (귀무 {rec['null_mean_3']:.3f}) · 재적합 W 차 최대 {max(w_saved.values()):.2e}", flush=True)
