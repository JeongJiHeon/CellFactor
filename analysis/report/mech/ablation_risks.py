#!/usr/bin/env python3
"""PLAN_ABLATION_BASELINES — 단순화 모델 3종의 환자별 위험(CellFactor 구성요소로 재구현). env ORG · DEV.
출력 results/mech/ablation_risks/{ORG}.npz  키 '{unit}__{cohort}' = (구성원 수, n) 배열 · {ORG}_INFO.json"""
import os, sys, json
ORG = os.environ["ORG"]; DEV_S = os.environ.get("DEV", "cuda:0")
os.environ.update(ARM="A_cand3k", STEP="none", DEV=DEV_S, CHAIN="1")
sys.path.insert(0, os.path.join(os.environ.get("CELLFACTOR_PROJECT", "."), "code/eval/hvg_run1a"))
import numpy as np, torch
from pathlib import Path
from lifelines.utils import concordance_index
F = Path(os.environ.get("CELLFACTOR_PROJECT", ".")); OUT = F / "results/mech/ablation_risks"; OUT.mkdir(parents=True, exist_ok=True)
if ORG == "breast":
    import chain_breast as M
    _ctx, cohorts = M.protocol.endpoints(); X = {n: cd["x"].float() for n, cd in cohorts.items() if not n.startswith("TCGA")}
else:
    import chain_organ as M
    data = torch.load(M.L.STAGE_DATA, map_location="cpu", weights_only=False); cohorts = M.LS3.load_cohorts(M.bundle, data)
    X = {n: c["x"].float() for n, c in cohorts.items() if not n.startswith("TCGA")}
bundle = M.bundle; genes = [str(g) for g in bundle["gene_names"]]
PB = [torch.load(F / f"results/hvg_design/run1a/{ORG}/A_cand3k/c{c}/pred_bulk.pt", map_location="cpu", weights_only=False) for c in range(1, 7)]
ES = [torch.load(F / f"results/paper_acand3k/ext_states/{ORG}_c{c}.pt", weights_only=False) for c in range(1, 7)]
tr = PB[0]["train_idx"].numpy(); ho = PB[0]["heldout_idx"].numpy()
assert np.array_equal(tr, np.asarray(bundle["bulk_split"]["train"])) and np.array_equal(ho, np.asarray(bundle["bulk_split"]["heldout"])), "split mismatch"
Xall = bundle["bulk_x"].float(); X["TCGA-heldout"] = Xall[ho]; Xtr = Xall[tr]
t_tr, e_tr = PB[0]["time_all"].numpy()[tr], PB[0]["event_all"].numpy()[tr]
# 프로그램 점수
import common as C
k0 = json.load(open(C.ORG[ORG]["k0"] if Path(C.ORG[ORG]["k0"]).is_absolute() else (C.ORG[ORG]["cwd"] / C.ORG[ORG]["k0"])))
gi = {g: i for i, g in enumerate(genes)}
progs = [p for p in PB[0]["programs"] if not p.startswith("free") and p in k0]
idx = {p: [gi[g] for g in (k0[p] if isinstance(k0[p], list) else k0[p]["genes"]) if g in gi] for p in progs}; progs = [p for p in progs if len(idx[p]) >= 1]
sig = lambda x: np.stack([x[:, idx[p]].mean(1).numpy() for p in progs], 1)
Str = sig(Xtr); mu, sd = Str.mean(0), Str.std(0) + 1e-6; Z = lambda s: (s - mu) / sd
# π̂ (2A 는 체인 공통인지 확인)
assert all(torch.allclose(p["pi"], PB[0]["pi"]) for p in PB), "pi differs across chains"
PI = {n: ES[0]["PM"][n]["pi"].double().numpy() for n in ES[0]["PM"]}; PI["TCGA-heldout"] = PB[0]["pi"].double().numpy()[ho]; pi_tr = PB[0]["pi"].double().numpy()[tr]
pmu, psd = pi_tr.mean(0), pi_tr.std(0) + 1e-6; PZ = lambda p: (p - pmu) / psd


def cox_nll(r, t, e):
    o = torch.argsort(t, descending=True); r, e = r[o], e[o]; return -((r - torch.logcumsumexp(r, 0)) * e).sum() / e.sum().clamp_min(1)


def fitw(Xm, t, e, lam):
    X_ = torch.tensor(Xm, dtype=torch.float64); t = torch.tensor(t, dtype=torch.float64); e = torch.tensor(e, dtype=torch.float64)
    w = torch.zeros(X_.shape[1], dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); L = cox_nll(X_ @ w, t, e) + lam * (w * w).sum(); L.backward(); return L
    opt.step(cl); return w.detach().numpy()


def fit_cv(Xm):
    perm = np.random.default_rng(0).permutation(len(t_tr)); folds = np.array_split(perm, 5); cv = {}
    for lam in (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10):
        cs = []
        for f in folds:
            trn = np.setdiff1d(perm, f); w = fitw(Xm[trn], t_tr[trn], e_tr[trn], lam); cs.append(concordance_index(t_tr[f], -(Xm[f] @ w), e_tr[f]))
        cv[lam] = float(np.mean(cs))
    lam = max(cv, key=cv.get); return fitw(Xm, t_tr, e_tr, lam), lam, cv


out, info = {}, {"organ": ORG, "programs": progs, "n_programs": len(progs)}
w1, l1, cv1 = fit_cv(Z(Str)); info["program_score_cox"] = {"lambda": l1, "cv": cv1}
w2, l2, cv2 = fit_cv(np.hstack([PZ(pi_tr), Z(Str)])); info["composition_plus_program_score_cox"] = {"lambda": l2, "cv": cv2}
for n in [k for k in X if k in PI]:
    s = Z(sig(X[n])); p = PZ(PI[n]); assert len(s) == len(p), (n, len(s), len(p))
    out[f"programscore__{n}"] = (s @ w1)[None]; out[f"comp_programscore__{n}"] = (np.hstack([p, s]) @ w2)[None]
    st = []
    for c in range(6):
        pb, es = PB[c], ES[c]; pbar = pb["pi"].double().numpy()[tr].mean(0); W = pb["W"].double().numpy(); a = pb["alpha"].double().numpy()
        Mh = pb["M"].double().numpy()[ho] if n == "TCGA-heldout" else es["PM"][n]["M"].double().numpy()
        st.append((pbar[None] * (a[None] + (Mh * W[None]).sum(2))).sum(1))
    out[f"stateonly__{n}"] = np.stack(st)
np.savez(OUT / f"{ORG}.npz", **out); json.dump(info, open(OUT / f"{ORG}_INFO.json", "w"), indent=1)
print(ORG, "programs", len(progs), "lambda", l1, l2, {k: v.shape for k, v in out.items()})
