#!/usr/bin/env python3
"""PLAN_PSEUDO_BULK — 유사벌크 대 실제 벌크의 프로그램 앵커 feature(M̂) 비교. 재학습 없음, H 추론만.
도메인: P_tr(학습 도너 trainsel 가방) · P_ho(보류 도너 val+test 가방) · R(TCGA 전체) · R_ext(외부 코호트).
env ORG(breast|liver|pancreas) · DEV. 원 코드 경로(코호트 적재)를 쓰므로 PYTHONPATH=code/lib/experiments:code/lib, 췌장은 cwd sandbox6.
출력 results/mech/pseudo_vs_bulk/{ORG}.json (+ results_json/mech/pseudo_vs_bulk 미러)"""
import os, sys, json
ORG = os.environ["ORG"]; DEV_S = os.environ.get("DEV", "cuda:0"); assert DEV_S in ("cuda:0", "cuda:1")
os.environ.update(ARM="A_cand3k", STEP="none", DEV=DEV_S, CHAIN="1")
F = os.environ.get("CELLFACTOR_PROJECT", ".")
sys.path.insert(0, f"{F}/code/eval/hvg_run1a"); sys.path.insert(0, f"{F}")
import numpy as np, torch
from pathlib import Path
from scipy.stats import rankdata
import train as T
from model import load_state_estimator, fine_composition
F = Path(F); OUT = F / "results/mech/pseudo_vs_bulk"; OUT.mkdir(parents=True, exist_ok=True); OJ = F / "results_json/mech/pseudo_vs_bulk"; OJ.mkdir(parents=True, exist_ok=True)
dev = torch.device(DEV_S); cfg = T.ORGANS[ORG]

# ---------- 외부 코호트 x (원 체인 코드의 적재 경로) ----------
if ORG == "breast":
    import chain_breast as CM
    _ctx, cohorts = CM.protocol.endpoints(); EXT_X = {n: cd["x"].float() for n, cd in cohorts.items() if not n.startswith("TCGA")}
else:
    import chain_organ as CM
    data = torch.load(CM.L.STAGE_DATA, map_location="cpu", weights_only=False); cohorts = CM.LS3.load_cohorts(CM.bundle, data)
    EXT_X = {n: c["x"].float() for n, c in cohorts.items() if not n.startswith("TCGA")}
bundle = T.load(cfg["registry"]); genes = [str(g) for g in bundle["gene_names"]]; gi = {g: i for i, g in enumerate(genes)}
mapping = T.type_mapping(cfg, bundle); Cn = len(bundle["cell_type_names"]); TYPES = list(bundle["cell_type_names"])
A = T.load_a(cfg, None, dev)
progs = json.load(open(F / cfg["programs"]))
PB1 = torch.load(F / f"results/hvg_design/run1a/{ORG}/A_cand3k/c1/pred_bulk.pt", map_location="cpu", weights_only=False)
AX = list(PB1["programs"]); NAMED = [i for i, p in enumerate(AX) if not p.startswith("free")]; FREE = [i for i, p in enumerate(AX) if p.startswith("free")]
GIDX = {AX[k]: torch.as_tensor([gi[g] for g in progs[AX[k]] if g in gi]) for k in NAMED}
X_TCGA = bundle["bulk_x"].float()
print(f"[{ORG}] axes {len(AX)} (named {len(NAMED)}, free {len(FREE)}) · types {TYPES} · ext {list(EXT_X)}", flush=True)


def scores(x):
    """S_k: 표준화 입력의 프로그램 유전자 평균 (표본 × 이름 붙은 축)."""
    return torch.stack([x[:, GIDX[AX[k]]].mean(1) for k in NAMED], 1).numpy()


@torch.no_grad()
def infer_bag(h, G, expr, coarse):
    M, P = [], []
    for i in range(0, len(expr), 128):
        xb = expr[i:i + 128].to(dev); q, rel = A(xb); P.append(fine_composition(q, rel, mapping, Cn).cpu())
        M.append(h(xb[:, G], coarse[i:i + 128].to(dev)).cpu())
    return torch.cat(M).numpy().astype(np.float64), torch.cat(P).numpy().astype(np.float64)


def eff_rank(Z):
    Z = Z - Z.mean(0); s = np.linalg.svd(Z, compute_uv=False); p = s / max(s.sum(), 1e-12); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def eff_rank_sub(Z, n, rng, reps=200):
    if len(Z) <= n: return eff_rank(Z)
    return float(np.mean([eff_rank(Z[rng.choice(len(Z), n, replace=False)]) for _ in range(reps)]))


def axis_stats(Mr, Mp):
    """유형 × 축 d, SD 비 (Mr, Mp: 표본 × 유형 × 축)."""
    sp = Mp.std(0).clip(1e-6); return (Mr.mean(0) - Mp.mean(0)) / sp, Mr.std(0) / sp


def struct_sim(Mr, Mp, t):
    def ut(Z):
        C = np.corrcoef(Z[:, t, NAMED].T); return C[np.triu_indices_from(C, 1)]
    a, b = ut(Mr), ut(Mp); ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def spearman_cols(a, b):
    ra, rb = rankdata(a, axis=0), rankdata(b, axis=0); ra -= ra.mean(0); rb -= rb.mean(0)
    return (ra * rb).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0)).clip(1e-12)


def chain_eval(ch):
    hck = F / f"weights/hvg_run1a/{ORG}/A_cand3k/c{ch}"
    ck = torch.load(hck / "h.pt", map_location="cpu", weights_only=False); G = ck["gene_idx"]; h = load_state_estimator(hck / "h.pt", dev)
    B = {r: torch.load(hck / f"materialized/paired_{r}.pt", map_location="cpu", weights_only=False) for r in ("trainsel", "val", "test")}
    Mb, Pb = {}, {}
    for r, b in B.items(): Mb[r], Pb[r] = infer_bag(h, G, b["expression_a"], b["coarse_a"])
    ref = torch.load(F / f"results/hvg_design/run1a/{ORG}/A_cand3k/c{ch}/pred_bags.pt", map_location="cpu", weights_only=False)
    ident = max(float(np.abs(Mb[r] - ref[r]["M_hat"].numpy()).max()) for r in ("val", "test")); assert ident < 1e-5, (ch, ident)
    pb = torch.load(F / f"results/hvg_design/run1a/{ORG}/A_cand3k/c{ch}/pred_bulk.pt", map_location="cpu", weights_only=False)
    ep = F / f"results/paper_acand3k/ext_states/{ORG}_c{ch}.pt"   # 외부 상태는 c1–c6 만 있다(c7–c12 는 민감도 J1–J3 전용)
    es = torch.load(ep, weights_only=False) if ep.exists() else {"PM": {}}
    D = {"P_tr": (Mb["trainsel"], Pb["trainsel"], scores(B["trainsel"]["expression_a"])),
         "P_ho": (np.concatenate([Mb["val"], Mb["test"]]), np.concatenate([Pb["val"], Pb["test"]]), scores(torch.cat([B["val"]["expression_a"], B["test"]["expression_a"]]))),
         "R": (pb["M"].double().numpy(), pb["pi"].double().numpy(), scores(X_TCGA))}
    ho = pb["heldout_idx"].numpy(); D["R_heldout"] = tuple(v[ho] for v in D["R"])
    for n in es["PM"]: D[f"ext:{n}"] = (es["PM"][n]["M"].double().numpy(), es["PM"][n]["pi"].double().numpy(), scores(EXT_X[n]))
    # 조성 겹침(민감도): TCGA q 가 P_tr q 의 5–95 백분위 안
    qp = B["trainsel"]["coarse_a"].numpy(); qr = pb["q"].numpy(); lo, hi = np.percentile(qp, 5, 0), np.percentile(qp, 95, 0)
    ov = np.all((qr >= lo) & (qr <= hi), 1); D["R_overlap"] = tuple(v[ov] for v in D["R"])
    truth = (B["trainsel"]["target_a"].double().numpy(), B["trainsel"]["present_a"].numpy() > 0)
    return D, truth, int(ov.sum())


rng0 = np.random.default_rng(0)
res = {"organ": ORG, "types": TYPES, "named_axes": [AX[k] for k in NAMED], "free_axes": [AX[k] for k in FREE], "chains": {}}
A_store = {}   # M6 체인 평균용: domain -> list of z(A) (표본 × 이름 축)
CHAINS = list(range(1, 7)) + (list(range(7, 13)) if ORG in ("breast", "liver") else [])
for ch in CHAINS:
    D, (Mt, pres), n_ov = chain_eval(ch); Mp = D["P_tr"][0]; out = {"n": {k: len(v[0]) for k, v in D.items()}, "n_overlap": n_ov, "per_type": {}}
    n_min = min(len(D[k][0]) for k in ("P_tr", "P_ho", "R"))
    for t, tn in enumerate(TYPES):
        row = {}
        for dom in [k for k in D if k != "P_tr"]:
            d, r = axis_stats(D[dom][0], Mp)
            Mr = D[dom][0]; lo, hi = np.percentile(Mp[:, t], 1, 0), np.percentile(Mp[:, t], 99, 0)
            row[dom] = {"d_abs_med_named": float(np.median(np.abs(d[t, NAMED]))), "d_med_named": float(np.median(d[t, NAMED])),
                        "sd_ratio_med_named": float(np.median(r[t, NAMED])), "d_abs_med_free": float(np.median(np.abs(d[t, FREE]))),
                        "sd_ratio_med_free": float(np.median(r[t, FREE])),
                        "outside_1_99_med_named": float(np.median(((Mr[:, t] < lo) | (Mr[:, t] > hi)).mean(0)[NAMED]))}
            if dom in ("P_ho", "R", "R_overlap"): row[dom]["struct_sim"] = struct_sim(Mr, Mp, t)
        nm = min(n_min, len(D["R_overlap"][0])) if len(D["R_overlap"][0]) >= 20 else n_min
        rk = {dom: eff_rank_sub(D[dom][0][:, t][:, NAMED], n_min, rng0) for dom in ("P_tr", "P_ho", "R")}
        rk["R_overlap"] = eff_rank_sub(D["R_overlap"][0][:, t][:, NAMED], nm, rng0) if len(D["R_overlap"][0]) >= 20 else None
        rk["P_tr_overlapN"] = eff_rank_sub(Mp[:, t][:, NAMED], nm, rng0)
        pm = pres[:, t]; rk["P_tr_truth"] = eff_rank_sub(Mt[pm, t][:, NAMED], min(n_min, int(pm.sum())), rng0) if pm.sum() >= 10 else None
        sd_truth = Mt[pm, t].std(0) if pm.sum() >= 10 else None
        row["eff_rank_named"] = rk; row["n_min"] = n_min
        row["P_tr_sd_pred_over_truth_med_named"] = float(np.median(Mp[pm, t].std(0)[NAMED] / sd_truth[NAMED].clip(1e-6))) if sd_truth is not None else None
        out["per_type"][tn] = row
    # M6 (체인별): A_k = Σ_t π̂_t M̂_tk, ρ_k = Spearman(A_k, S_k)
    m6 = {}
    for dom, (M_, P_, S_) in D.items():
        Ak = np.einsum("bt,btk->bk", P_, M_[:, :, NAMED]); m6[dom] = float(np.median(spearman_cols(Ak, S_)))
        if ch <= 6: A_store.setdefault(dom, {})[ch] = (Ak - Ak.mean(0)) / Ak.std(0).clip(1e-9)
    # 참조: 정답 상태(있는 유형만, π̂ 를 그 유형들로 재정규화)
    w = D["P_tr"][1] * pres; w = w / w.sum(1, keepdims=True).clip(1e-12)
    m6["P_tr_truth_ref"] = float(np.median(spearman_cols(np.einsum("bt,btk->bk", w, Mt[:, :, NAMED]), D["P_tr"][2])))
    out["M6_rho_med_named"] = m6
    res["chains"][ch] = out
    print(f"[{ORG} c{ch}] n {out['n']} overlap {n_ov} · M6 " + " ".join(f"{k}={v:.3f}" for k, v in m6.items()), flush=True)

# ---------- 판정 J1–J3 (c1–c6), 민감도 c7–c12 ----------
def judge(chs):
    J = {}
    for tn in TYPES:
        g = lambda dom, key: np.array([res["chains"][c]["per_type"][tn][dom][key] for c in chs])
        sr, srh = g("R", "sd_ratio_med_named"), g("P_ho", "sd_ratio_med_named")
        da, dah = g("R", "d_abs_med_named"), g("P_ho", "d_abs_med_named")
        rr = np.array([res["chains"][c]["per_type"][tn]["eff_rank_named"]["R"] for c in chs]); rp = np.array([res["chains"][c]["per_type"][tn]["eff_rank_named"]["P_tr"] for c in chs])
        need = 5 if len(chs) == 6 else int(np.ceil(5 / 6 * len(chs)))
        J[tn] = {"J1_shrink": bool(((sr < 0.8).sum() >= need) and ((sr < srh).sum() >= need)), "sd_ratio_R": [float(np.median(sr)), float(sr.min()), float(sr.max())],
                 "sd_ratio_Pho": [float(np.median(srh)), float(srh.min()), float(srh.max())],
                 "J2_shift": bool(((da > 0.5).sum() >= need) and ((da > dah).sum() >= need)), "d_abs_R": [float(np.median(da)), float(da.min()), float(da.max())],
                 "d_abs_Pho": [float(np.median(dah)), float(dah.min()), float(dah.max())],
                 "J3_rank_down": bool((rr < rp).sum() >= need), "eff_rank_R": float(np.median(rr)), "eff_rank_Ptr": float(np.median(rp)),
                 "eff_rank_Pho": float(np.median([res["chains"][c]["per_type"][tn]["eff_rank_named"]["P_ho"] for c in chs])),
                 "eff_rank_truth": (float(np.median([v for v in [res["chains"][c]["per_type"][tn]["eff_rank_named"]["P_tr_truth"] for c in chs] if v is not None])) if any(res["chains"][c]["per_type"][tn]["eff_rank_named"]["P_tr_truth"] is not None for c in chs) else None)}
    return J


res["judgement_c1_6"] = judge(list(range(1, 7)))
if ORG in ("breast", "liver"): res["sensitivity_c7_12"] = judge(list(range(7, 13)))

# ---------- J4: 체인 평균 A_k 로 ρ_k, 도메인 독립 bootstrap ----------
def ens(dom, chs): return np.mean([A_store[dom][c] for c in chs], 0)
S = {}
D1, _, _ = chain_eval(1)   # S_k 는 체인 무관
for dom in A_store: S[dom] = D1[dom][2]
Ar, Ap = ens("R", range(1, 7)), ens("P_tr", range(1, 7)); rho_r, rho_p = spearman_cols(Ar, S["R"]), spearman_cols(Ap, S["P_tr"])
rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    ir = rng.integers(0, len(Ar), len(Ar)); ip = rng.integers(0, len(Ap), len(Ap))
    diffs.append(np.median(spearman_cols(Ar[ir], S["R"][ir])) - np.median(spearman_cols(Ap[ip], S["P_tr"][ip])))
lo, hi = np.percentile(diffs, [2.5, 97.5])
res["J4"] = {"rho_med_R": float(np.median(rho_r)), "rho_med_Ptr": float(np.median(rho_p)), "diff": float(np.median(rho_r) - np.median(rho_p)), "ci95": [float(lo), float(hi)],
             "verdict": "lower in real bulk" if hi < 0 else ("higher in real bulk" if lo > 0 else "no difference detected"),
             "per_program": {AX[k]: {"R": float(rho_r[j]), "P_tr": float(rho_p[j])} for j, k in enumerate(NAMED)},
             "other_domains_rho_med": {dom: float(np.median(spearman_cols(ens(dom, range(1, 7)), S[dom]))) for dom in A_store}}
for d in (OUT, OJ): json.dump(res, open(d / f"{ORG}.json", "w"), indent=1)
print(json.dumps({"J": {t: {k: v for k, v in j.items() if k.startswith("J")} for t, j in res["judgement_c1_6"].items()}, "J4": {k: res["J4"][k] for k in ("rho_med_R", "rho_med_Ptr", "diff", "ci95", "verdict")}}, indent=1))
