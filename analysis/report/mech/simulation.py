#!/usr/bin/env python3
"""PLAN_SIMULATION — 알려진 위험 기여를 가진 semi-synthetic 코호트(GSE161529 세포)에서 CellFactor 분해 정확도.
PACE·CCE·CCSE 고정(논문 c1–c6), CSRF 만 가상 생존으로 다시 적합. 정답은 모델 좌표·학습 계수와 무관(유전자 집합 점수).
env SCEN(A|B|C0|C5|C9|D) · REPS(예 0-9) · DEV(cuda:0|cuda:1). PYTHONPATH=code/lib.
출력 results/mech/simulation/{SCEN}_r{r}.json (ALPHA_FREE=0 이면 {SCEN}_r{r}_a0.json)"""
import os, sys, json, math, time
SCEN = os.environ["SCEN"]; DEV_S = os.environ.get("DEV", "cuda:0"); assert DEV_S in ("cuda:0", "cuda:1")
a, b = (int(x) for x in os.environ.get("REPS", "0-9").split("-")); REPS = list(range(a, b + 1))
os.environ["EXT_CENTER"] = "1"
F = os.environ.get("CELLFACTOR_PROJECT", ".")
sys.path.insert(0, f"{F}"); sys.path.insert(0, f"{F}/code/lib")
import numpy as np, torch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import train as T
from model import load_state_estimator, fine_composition
from run_breast_stage3_preprocess_v2 import prepare_external

F = Path(F); OUT = F / "results/mech/simulation"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device(DEV_S); cfg = T.ORGANS["breast"]; CHAINS = list(range(1, 7))
N_TR, N_TE, N_CELL, TAU, EV_RATE = 500, 500, 1024, 0.5, 0.40
ALPHA_FREE = os.environ.get("ALPHA_FREE", "1") == "1"; SUF = "" if ALPHA_FREE else "_a0"   # CSRF 재적합 규칙 = 배포(α 자유, PLAN_ALPHA_FREE). 0 = 초판(α=0) 재현
LAMS = (30., 100., 300., 1000., 3000., 10000.)

# ---------------------------------------------------------------- 고정 자원
t0 = time.time()
D = torch.load(F / "results/panel16k_g161/G161_STAGE1_DATA.pt", map_location="cpu", weights_only=False, mmap=True)
bundle = T.load(cfg["registry"]); TYPES = list(bundle["cell_type_names"]); Cn = len(TYPES)
genes = [str(g) for g in bundle["gene_names"]]; assert genes == [str(g) for g in D["gene_names"]] and list(D["cell_type_names"]) == TYPES
mapping = T.type_mapping(cfg, bundle)
A = T.load_a(cfg, None, dev)
H, G = {}, {}; S = {}
for c in CHAINS:
    p = F / f"weights/hvg_run1a/breast/A_cand3k/c{c}/h.pt"; ck = torch.load(p, map_location="cpu", weights_only=False)
    H[c] = load_state_estimator(p, dev); G[c] = torch.as_tensor(ck["gene_idx"]).to(dev); S[c] = torch.as_tensor(ck["s"]).float()
pb1 = torch.load(F / "results/hvg_design/run1a/breast/A_cand3k/c1/pred_bulk.pt", map_location="cpu", weights_only=False)
AX = list(pb1["programs"]); LAT = len(AX)
P_BAR = pb1["pi"][pb1["train_idx"]].double().mean(0).numpy()

# 식별 점검: 논문 c1 의 TCGA π̂·M̂ 재현
with torch.no_grad():
    x0 = bundle["bulk_x"][:512].float().to(dev); q0, r0 = A(x0)
    d_pi = float((fine_composition(q0, r0, mapping, Cn).cpu() - pb1["pi"][:512]).abs().max()); d_M = float((H[1](x0[:, G[1]], q0).cpu() - pb1["M"][:512]).abs().max())
assert d_pi < 1e-4 and d_M < 1e-3, (d_pi, d_M)

# 효과 표(사전 고정)
ti = TYPES.index
CE, TC, CAF, EN = ti("Cancer Epithelial"), ti("T-cells"), ti("CAFs"), ti("Endothelial")
MOD = [(CE, "CS_PROLIFERATION"), (CE, "HM_Estrogen_Response_Early"), (CE, "CS_HYPOXIA"), (TC, "T_CYTOTOXIC"), (CAF, "HM_TGF_beta_Signaling"), (EN, "CS_HYPOXIA")]
SIGN = {(CE, "CS_PROLIFERATION"): 1, (CE, "HM_Estrogen_Response_Early"): -1, (TC, "T_CYTOTOXIC"): -1, (CAF, "HM_TGF_beta_Signaling"): 1}
if SCEN.startswith("C"): SIGN[(CE, "CS_HYPOXIA")] = 1
RHO_C = {"C0": 0.0, "C5": 0.5, "C9": 0.9}.get(SCEN)
PROGS = sorted({m for _, m in MOD})
progs = json.load(open(F / cfg["programs"])); gi = {g: i for i, g in enumerate(genes)}
PG = {m: torch.as_tensor([gi[g] for g in progs[m] if g in gi]) for m in PROGS}

# 단일세포: GPU 상주 count, 세포별 프로그램 점수(log1p CP10k 평균, 16,509 유전자 library)
counts = D["sc_counts"]; counts = counts.view(torch.int16) if counts.dtype == torch.uint16 else counts
CNT = counts.to(dev); NC = len(CNT)
ctype = D["sc_cell_type"].numpy(); donor = D["sc_donor"].numpy(); n_donor = len(D["donor_names"])


def rows_float(idx):
    x = CNT[idx].float(); x[x < 0] += 65536.; return x


RAW = torch.zeros(NC, len(PROGS))
with torch.no_grad():
    for i in range(0, NC, 20000):
        x = rows_float(torch.arange(i, min(i + 20000, NC), device=dev)); lib = x.sum(1, keepdim=True).clamp_min(1)
        ln = torch.log1p(x / lib * 1e4)
        RAW[i:i + len(x)] = torch.stack([ln[:, PG[m].to(dev)].mean(1) for m in PROGS], 1).cpu()
RAW = RAW.numpy().astype(np.float64)
# 체인별 calibrated z′ (oracle 상태 M*)
ZC = {}
for c in CHAINS:
    z = torch.load(F / f"results/report_acand3k/g161/zprime_A_cand3k_c{c}.pt", map_location="cpu", weights_only=False)["z"].float()
    ZC[c] = (z / S[c][None]).clamp(-12, 12).to(dev)
# TCGA 학습 환자 log2(1+CPM), 16,509 유전자 안 재정규화(D 오프셋)
tl = bundle["bulk_log2_cpm"][bundle["bulk_split"]["train"]].double(); tc = (2 ** tl - 1).clamp_min(0); tc = tc / tc.sum(1, keepdim=True) * 1e6
TCGA_LOG_MEAN = torch.log2(1 + tc).mean(0).float()
print(f"[sim {SCEN}] ready {time.time() - t0:.0f}s · ident π {d_pi:.1e} M {d_M:.1e} · p̄ {np.round(P_BAR, 3).tolist()}", flush=True)


# ---------------------------------------------------------------- 생성
def allocate(p):
    raw = p * N_CELL; n = np.floor(raw).astype(int); rem = N_CELL - n.sum()
    if rem: n[np.argsort(-(raw - n))[:rem]] += 1
    return n


def make_patients(n, donors, rng, zstat):
    """조성·잠재값 생성 → 세포 추출. 반환: 세포 index, 실현 조성, 정답 상태(원척도 평균 z), 잠재값."""
    comp = rng.dirichlet(20 * P_BAR, size=n)
    U = rng.standard_normal((n, len(MOD))); e = rng.standard_normal(n); f, e1, e2 = rng.standard_normal((3, n))   # 난수 흐름은 시나리오와 무관
    if SCEN == "B":
        lg = np.log(comp[:, CE] / (1 - comp[:, CE]).clip(1e-9)); lg = (lg - lg.mean()) / lg.std(); U[:, 0] = 0.6 * lg + math.sqrt(1 - .36) * e
    if RHO_C is not None:
        U[:, 2] = math.sqrt(RHO_C) * f + math.sqrt(1 - RHO_C) * e1; U[:, 5] = math.sqrt(RHO_C) * f + math.sqrt(1 - RHO_C) * e2
    pool = {k: np.where(np.isin(donor, donors) & (ctype == k))[0] for k in range(Cn)}
    zpool = {k: (RAW[pool[k]] - zstat[k][0]) / zstat[k][1] for k in range(Cn)}
    tg = torch.Generator(device="cpu").manual_seed(int(rng.integers(2 ** 31)))
    IDX, P, ST = [], np.zeros((n, Cn)), np.zeros((n, len(MOD)))
    for i in range(n):
        alloc = allocate(comp[i]); P[i] = alloc / N_CELL; sel = []
        for k in range(Cn):
            if alloc[k] == 0: continue
            j = [jj for jj, (kk, _) in enumerate(MOD) if kk == k]
            if j:
                lw = TAU * sum(U[i, jj] * zpool[k][:, PROGS.index(MOD[jj][1])] for jj in j); w = torch.as_tensor(np.exp(lw - lw.max()))
                pick = torch.multinomial(w, int(alloc[k]), replacement=True, generator=tg).numpy()
            else:
                pick = torch.randint(len(pool[k]), (int(alloc[k]),), generator=tg).numpy()
            sel.append(pool[k][pick])
            for jj in j: ST[i, jj] = zpool[k][pick, PROGS.index(MOD[jj][1])].mean()
        IDX.append(np.concatenate(sel))
    return IDX, P, ST, U


@torch.no_grad()
def bulk_log2cpm(IDX):
    out = torch.zeros(len(IDX), len(genes))
    for i, idx in enumerate(IDX):
        s = rows_float(torch.as_tensor(idx, device=dev)).sum(0); out[i] = torch.log2(1 + s / s.sum() * 1e6).cpu()
    return out


def prep(L):
    return prepare_external(L, torch.ones(L.shape[1], dtype=torch.bool), bundle).float()


@torch.no_grad()
def infer(x):
    P, Q = [], []
    for i in range(0, len(x), 512):
        xb = x[i:i + 512].to(dev); q, r = A(xb); P.append(fine_composition(q, r, mapping, Cn).cpu()); Q.append(q.cpu())
    Q = torch.cat(Q); M = {}
    for c in CHAINS:
        M[c] = torch.cat([H[c](x[i:i + 512].to(dev)[:, G[c]], Q[i:i + 512].to(dev)).cpu() for i in range(0, len(x), 512)])
    return torch.cat(P), M


@torch.no_grad()
def oracle_state(IDX, c):
    M = torch.zeros(len(IDX), Cn, LAT); ct = torch.as_tensor(ctype)
    for i, idx in enumerate(IDX):
        t = ct[idx]; z = ZC[c][torch.as_tensor(idx, device=dev)].cpu(); cnt = torch.bincount(t, minlength=Cn).clamp(min=1)
        M[i].index_add_(0, t, z); M[i] /= cnt[:, None]
    return M


def cindex(t, e, r):
    from lifelines.utils import concordance_index
    return float(concordance_index(t, -r, e))


def fit_bulk_cox(X, t, e, lam):
    Z = torch.as_tensor(X, dtype=torch.float32, device=dev); tt = torch.as_tensor(t, dtype=torch.float32, device=dev); ee = torch.as_tensor(e, dtype=torch.float32, device=dev)
    o = torch.argsort(tt, descending=True); Z, ee = Z[o], ee[o]
    w = torch.zeros(Z.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=400, lr=.5, history_size=20, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); r = Z @ w; L = -(r - torch.logcumsumexp(r, 0))[ee > 0].sum() / ee.sum().clamp_min(1) + lam * w.pow(2).sum() / Z.shape[1]; L.backward(); return L
    opt.step(cl); return w.detach().cpu().numpy()


def bulk_cox(Xtr, ttr, etr, Xte, rng):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8; Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    folds = rng.permutation(len(Ztr)) % 5; cv = {}
    for lam in LAMS:
        cs = []
        for f in range(5):
            tr, va = folds != f, folds == f; w = fit_bulk_cox(Ztr[tr], ttr[tr], etr[tr], lam); cs.append(cindex(ttr[va], etr[va], Ztr[va] @ w))
        cv[lam] = float(np.mean(cs))
    lam = max(cv, key=cv.get); w = fit_bulk_cox(Ztr, ttr, etr, lam)
    return Zte @ w, lam, cv


# ---------------------------------------------------------------- 평가
def sp(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return None
    return float(spearmanr(a, b)[0])


def evaluate(name, pi_tr, M_tr, pi_te, M_te, truth, ttr, etr, tte, ete):
    """M_tr/M_te: {chain: tensor}. pi: tensor 또는 {chain: tensor}. 앙상블 기여 = 체인 기여 / 체인 시험 위험 SD 의 평균."""
    contrib = np.zeros((len(tte), Cn, LAT)); base = np.zeros((len(tte), Cn)); eff_w = np.zeros((Cn, LAT)); risk = np.zeros(len(tte)); per_chain_c = []
    for c in CHAINS:
        ptr = pi_tr[c] if isinstance(pi_tr, dict) else pi_tr; pte = pi_te[c] if isinstance(pi_te, dict) else pi_te
        m = T.fit_stage3(ptr.float(), M_tr[c].float(), torch.tensor(ttr, dtype=torch.float32), torch.tensor(etr, dtype=torch.float32), ALPHA_FREE, dev)
        W = m.W.detach().cpu().double().numpy(); Al = m.alpha.detach().cpu().double().numpy()
        cb = pte.double().numpy()[:, :, None] * M_te[c].double().numpy() * W[None]; ca = pte.double().numpy() * Al[None]   # 프로그램 기여 · 유형 기저(π α) 기여
        r = cb.sum((1, 2)) + ca.sum(1); sdr = r.std() + 1e-12; per_chain_c.append(cindex(tte, ete, r))
        contrib += cb / sdr / len(CHAINS); base += ca / sdr / len(CHAINS); risk += (r - r.mean()) / sdr / len(CHAINS)
        eff_w += W * M_tr[c].double().numpy().std(0) / len(CHAINS)
    Ttype, Tterm, eta = truth["T_type"], truth["T_term"], truth["eta"]
    R_type = contrib.sum(2) + base; out = {"c_index": cindex(tte, ete, risk), "c_index_chains": per_chain_c, "pearson_eta": float(pearsonr(risk, eta)[0])}
    eff_types = sorted({k for (k, _) in SIGN}); out["type_spearman"] = {TYPES[k]: sp(R_type[:, k], Ttype[:, k]) for k in eff_types}
    v = R_type.var(0); out["attribution_share"] = float(v[eff_types].sum() / v.sum()); out["type_var_share"] = {TYPES[k]: float(v[k] / v.sum()) for k in range(Cn)}
    terms = {}
    for (k, mname), sg in SIGN.items():
        j = AX.index(mname); terms[f"{TYPES[k]}|{mname}"] = {"spearman": sp(contrib[:, k, j], Tterm[(k, mname)]), "sign_ok": bool(np.sign(eff_w[k, j]) == sg),
                                                            "eff_w": float(eff_w[k, j]), "contrib_sd": float(contrib[:, k, j].std())}
    out["terms"] = terms
    csd = contrib.std(0); K = len(SIGN); top = np.argsort(-csd.ravel())[:K]
    truth_set = {k * LAT + AX.index(m) for (k, m) in SIGN}; out["topK_precision"] = float(np.mean([t in truth_set for t in top]))
    out["topK"] = [f"{TYPES[t // LAT]}|{AX[t % LAT]}" for t in top]
    jh = AX.index("CS_HYPOXIA"); e_en, e_ce = csd[EN, jh], csd[CE, jh]; w_en, w_ce = abs(eff_w[EN, jh]), abs(eff_w[CE, jh])
    out["P3_endo_share_contrib"] = float(e_en / (e_en + e_ce + 1e-12)); out["P3_endo_share_W"] = float(w_en / (w_en + w_ce + 1e-12))
    out["endo_hypoxia"] = {"spearman_with_endo_state": sp(contrib[:, EN, jh], truth["S_te"][:, MOD.index((EN, "CS_HYPOXIA"))]), "eff_w": float(eff_w[EN, jh])}
    return out


def run(r):
    t1 = time.time(); rng = np.random.default_rng(r)
    perm = rng.permutation(n_donor); dtr, dte = np.sort(perm[:n_donor // 2]), np.sort(perm[n_donor // 2:])
    trc = np.isin(donor, dtr); zstat = {k: (RAW[trc & (ctype == k)].mean(0), RAW[trc & (ctype == k)].std(0) + 1e-9) for k in range(Cn)}
    rg_tr, rg_te = np.random.default_rng(1000 + r), np.random.default_rng(2000 + r)
    Itr, Ptr, Str, _ = make_patients(N_TR, dtr, rg_tr, zstat); Ite, Pte, Ste, _ = make_patients(N_TE, dte, rg_te, zstat)
    mu_s, sd_s = Str.mean(0), Str.std(0) + 1e-9; Str, Ste = (Str - mu_s) / sd_s, (Ste - mu_s) / sd_s
    beta = {key: sg / Ptr[:, key[0]].mean() for key, sg in SIGN.items()}

    def truth_of(P, St):
        Tterm = {key: P[:, key[0]] * beta[key] * St[:, MOD.index(key)] for key in SIGN}
        Ttype = np.zeros((len(P), Cn))
        for key, v in Tterm.items(): Ttype[:, key[0]] += v
        return Tterm, Ttype
    Tt_tr, Ty_tr = truth_of(Ptr, Str); Tt_te, Ty_te = truth_of(Pte, Ste)
    eta_tr, eta_te = Ty_tr.sum(1), Ty_te.sum(1); sc = eta_tr.std(); eta_tr, eta_te = eta_tr / sc, eta_te / sc
    Tt_te = {k: v / sc for k, v in Tt_te.items()}; Ty_te = Ty_te / sc
    rs = np.random.default_rng(3000 + r)
    Ttr_, Tte_ = rs.exponential(size=N_TR) / np.exp(eta_tr), rs.exponential(size=N_TE) / np.exp(eta_te)
    Ctr_u, Cte_u = rs.uniform(size=N_TR), rs.uniform(size=N_TE)
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        c = math.sqrt(lo * hi); rate = (Ttr_ <= Ctr_u * c).mean()
        lo, hi = (c, hi) if rate < EV_RATE else (lo, c)
    c = math.sqrt(lo * hi)
    ttr, etr = np.minimum(Ttr_, Ctr_u * c), (Ttr_ <= Ctr_u * c).astype(float); tte, ete = np.minimum(Tte_, Cte_u * c), (Tte_ <= Cte_u * c).astype(float)
    # bulk
    Ltr, Lte = bulk_log2cpm(Itr), bulk_log2cpm(Ite)
    if SCEN == "D":
        off = TCGA_LOG_MEAN - Ltr.mean(0); rn = torch.Generator().manual_seed(4000 + r)
        Ltr = Ltr + off + 0.3 * torch.randn(Ltr.shape, generator=rn); Lte = Lte + off + 0.3 * torch.randn(Lte.shape, generator=rn)
    Xtr, Xte = prep(Ltr), prep(Lte)
    pi_tr, M_tr = infer(Xtr); pi_te, M_te = infer(Xte)
    Ms_tr = {c: oracle_state(Itr, c) for c in CHAINS}; Ms_te = {c: oracle_state(Ite, c) for c in CHAINS}
    Ptr_t, Pte_t = torch.as_tensor(Ptr).float(), torch.as_tensor(Pte).float()
    truth = {"T_type": Ty_te, "T_term": Tt_te, "eta": eta_te, "S_te": Ste}
    res = {"scenario": SCEN, "rep": r, "train_donors": dtr.tolist(), "test_donors": dte.tolist(), "event_rate_train": float(etr.mean()), "event_rate_test": float(ete.mean()),
           "c_true_eta": cindex(tte, ete, eta_te), "methods": {}}
    for nm, a_ in (("CellFactor", (pi_tr, M_tr, pi_te, M_te)), ("oracle_comp", (Ptr_t, M_tr, Pte_t, M_te)),
                   ("oracle_state", (pi_tr, Ms_tr, pi_te, Ms_te)), ("oracle_both", (Ptr_t, Ms_tr, Pte_t, Ms_te))):
        res["methods"][nm] = evaluate(nm, *a_, truth, ttr, etr, tte, ete)
    bc, lam, cv = bulk_cox(Xtr.numpy().astype(np.float64), ttr, etr, Xte.numpy().astype(np.float64), np.random.default_rng(5000 + r))
    res["bulk_cox"] = {"c_index": cindex(tte, ete, bc), "lambda": lam, "cv": {str(k): v for k, v in cv.items()}, "pearson_eta": float(pearsonr(bc, eta_te)[0])}
    pnp = pi_te.double().numpy()
    res["composition"] = {"mae_mean": float(np.abs(pnp - Pte).mean(0).mean()), "mae": {TYPES[k]: float(np.abs(pnp[:, k] - Pte[:, k]).mean()) for k in range(Cn)},
                          "spearman": {TYPES[k]: sp(pnp[:, k], Pte[:, k]) for k in range(Cn)}}
    Mbar = np.mean([M_te[c].double().numpy() for c in CHAINS], 0); Msbar = np.mean([Ms_te[c].double().numpy() for c in CHAINS], 0)
    res["state"] = {}
    for jj, (k, m) in enumerate(MOD):
        has = Pte[:, k] > 0
        res["state"][f"{TYPES[k]}|{m}"] = {"M_hat": sp(Mbar[has, k, AX.index(m)], Ste[has, jj]), "M_star": sp(Msbar[has, k, AX.index(m)], Ste[has, jj]), "n": int(has.sum())}
    res["seconds"] = time.time() - t1
    json.dump(res, open(OUT / f"{SCEN}_r{r}{SUF}.json", "w"), ensure_ascii=False, indent=1)
    cf = res["methods"]["CellFactor"]
    print(f"[sim {SCEN} r{r}] {res['seconds']:.0f}s · ev {etr.mean():.2f}/{ete.mean():.2f} · C true {res['c_true_eta']:.3f} CF {cf['c_index']:.3f} bulk {res['bulk_cox']['c_index']:.3f} · "
          f"type ρ {cf['type_spearman']} · share {cf['attribution_share']:.2f} · topK {cf['topK_precision']:.2f} · P3 {cf['P3_endo_share_contrib']:.2f}", flush=True)


for r in REPS:
    if (OUT / f"{SCEN}_r{r}{SUF}.json").exists() and not os.environ.get("FORCE"): print(f"[sim {SCEN} r{r}] exists", flush=True); continue
    run(r)
