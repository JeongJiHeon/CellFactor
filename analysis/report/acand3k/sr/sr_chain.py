"""벌크 → 세포상태 직접 복원 — 1a 체인 팔(A_cand3k · F · DEPLOYED 같은 경로). 정의는 code/eval/ablation_cp/state_recovery_arms.py(v7) 그대로:
 도너 전 세포 pseudobulk(log1p CPM, 전체 패널) → 2A(배포) q · 체인 H(목록 열) → M̂; 정답 M_true = 그 도너·유형 세포 z′ 평균(유형당 ≥20세포);
 (유형, 축)별 도너 간 Pearson(유방 ≥8 · 간 ≥5 도너) · 도너쌍 Δ부호 일치 → 유형별 중위 · 전체 중위; raw 기준선 = pseudobulk 시그니처 점수(H 무관).
 차이: z′·H 는 체인마다 다르다(체인별 좌표계) → 체인별 계산 후 평균 ± SD(체인 간 M̂ 평균 '앙상블'은 정의하지 않음).
 유방 = GSE161529 52도너(외부): M_true = G161 세포를 체인 Stage 1′ 인코더(교사: F = G161_TEACHER 전체 패널, A_cand3k = G161_cand3k_TEACHER)로 인코딩. 입력 표준화 = 체인이 학습한 AREG(prepared_v2_all26) 규약.
 간 = 보류 val+test 도너: M_true = 체인 z′(LIVER_STAGE1_DATA 전 세포 포함).
env ORG · DEV · ARMS(쉼표, 기본 DEPLOYED,F,A_cand3k) · 출력 results/report_acand3k/state_recovery/{org}_chains.json"""
import os, sys, json, math, numpy as np, torch
from pathlib import Path
from scipy.stats import pearsonr
F = Path(os.environ.get("CELLFACTOR_PROJECT", ".")); sys.path.insert(0, str(F / "code/eval/hvg_run1a")); import common as C
ORG = os.environ["ORG"]; DEVS = os.environ.get("DEV", "cuda:1"); C.dev_check(DEVS); DEV = torch.device(DEVS)
ARMS = os.environ.get("ARMS", "DEPLOYED,F,A_cand3k").split(","); MIN = 20; OUT = F / "results/report_acand3k/state_recovery"; OUT.mkdir(parents=True, exist_ok=True)
def chains(a): return [1, 2, 3] if a == "DEPLOYED" else list(range(1, 7))
if ORG == "breast":
    for k, v in dict(EXT_SUFFIX="16k", EXT_CENTER="1", A_CKPT_DIR=str(F / "results/stage2_redesign/run_20260828_p16k_stage2a_100k/stage2a_composition/breast/checkpoints")).items(): os.environ.setdefault(k, v)
    import run_breast_online_domain_adaptive_h as e2, run_breast_stage3_preprocess_v2 as protocol
    AREG = F / C.ORG["breast"]["areg"]; protocol.REGISTRY = AREG; e2.REGISTRY = AREG
    bundle = torch.load(AREG, map_location="cpu", weights_only=False); genes = [str(g) for g in bundle["gene_names"]]; Cn = len(bundle["cell_type_names"]); types = list(bundle["cell_type_names"])
    a = protocol.load_a(bundle, DEV); st = bundle["preprocessing"]["pseudo"]; clamp = float(bundle["preprocessing"]["clamp"])
    D = torch.load(F / "results/panel16k_g161/G161_STAGE1_DATA.pt", map_location="cpu", weights_only=False, mmap=True); assert [str(g) for g in D["gene_names"]] == genes
    ct = np.asarray(D["sc_cell_type"]); dn = np.asarray(D["sc_donor"]); cnt = D["sc_counts"]; cnt = cnt.view(torch.int16) if cnt.dtype == torch.uint16 else cnt
    donors = np.unique(dn); PMIN = 8
    def prep(PB): X = torch.stack(PB); return (((X / math.log(2.0)) - st["mean"]) / st["sd"]).clamp(-clamp, clamp)
    def coarse(x): co, im = a(x); return co
    def build_h(ck, G): bh = dict(bundle); bh["gene_names"] = [genes[i] for i in G.tolist()]; return e2.build_model(bh, DEV, ck["model_config"])
    def h_path(arm, c): return (F / f"weights/breast/stage2d_e2pp_all26_4k/h_seed{ {1: 20271701, 2: 20271702, 3: 20271703}[c]}.pt") if arm == "DEPLOYED" else C.w_dir("breast", arm, c) / "h.pt"
    TEACH = {"F": F / "results/panel16k_g161/G161_TEACHER.pt", "A_cand3k": F / "results/report_acand3k/g161/G161_cand3k_TEACHER.pt"}
    _TE = {}
    def encode_g161(arm, c):
        """체인 Stage 1′ 인코더(사후 평균)로 G161 세포 z′. DEPLOYED 는 기존 G161_ZPRIME_E2PP_FROM_G176."""
        if arm == "DEPLOYED": return torch.load(F / "results/panel16k_g161/G161_ZPRIME_E2PP_FROM_G176.pt", map_location="cpu", weights_only=False)["z"].float()
        cache = F / f"results/report_acand3k/g161/zprime_{arm}_c{c}.pt"
        if cache.exists(): return torch.load(cache)["z"].float()
        if arm not in _TE: _TE[arm] = torch.load(TEACH[arm], map_location="cpu", weights_only=False)["teacher_emb"].float()
        ck = torch.load(C.w_dir("breast", arm, c) / "stage1" / "G176_STAGE1_PRIME.pt", map_location="cpu", weights_only=False); sd = ck["state_dict"]
        post = torch.nn.Sequential(torch.nn.LayerNorm(512), torch.nn.Linear(512, 512), torch.nn.SiLU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, 256), torch.nn.LayerNorm(256), torch.nn.SiLU())
        post.load_state_dict({k[len("posterior."):]: v for k, v in sd.items() if k.startswith("posterior.")}); zm = torch.nn.Linear(256, sd["z_mean.weight"].shape[0]); zm.load_state_dict({"weight": sd["z_mean.weight"], "bias": sd["z_mean.bias"]})
        post, zm = post.to(DEV).eval(), zm.to(DEV).eval(); cen = sd["teacher_center"].to(DEV); T = _TE[arm]; out = []
        with torch.no_grad():
            for s in range(0, len(T), 8192): out.append(zm(post(T[s:s + 8192].to(DEV) - cen)).cpu())
        z = torch.cat(out); s_ = torch.tensor(json.load(open(C.run_dir("breast", arm, c) / "calibration.json"))["s"], dtype=torch.float32); z = (z / s_[None, :]).clamp(-12., 12.)   # chain_breast 의 Zc 와 같은 척도
        torch.save({"z": z.half(), "arm": arm, "chain": c, "teacher": str(TEACH[arm]), "stage1": str(C.w_dir("breast", arm, c) / "stage1")}, cache); return z
else:
    import run_organ_e2_prime as E2P
    L = E2P.L; bundle = L.prepare_registry(); genes = [str(g) for g in bundle["gene_names"]]; a = L.load_a(bundle, DEV); Cn = len(bundle["cell_type_names"]); types = list(bundle["cell_type_names"])
    data = torch.load(L.STAGE_DATA, map_location="cpu", weights_only=False); ct = np.asarray(data["sc_cell_type"]); dn = np.asarray(data["sc_donor"]); cnt = data["sc_counts"]
    donors = np.asarray(data["donor_split"]["val"] + data["donor_split"]["test"]); PMIN = 5
    def prep(PB): return L.standardize_online(bundle, torch.stack(PB))
    def coarse(x): q, rel = a(x); return q
    def build_h(ck, G): bh = dict(bundle); bh["gene_names"] = [genes[i] for i in G.tolist()]; return L.build_h(bh, DEV, ck["model_config"])
    def h_path(arm, c): return (F / f"weights/{ORG}/stage2d_e2pp/state_estimator_seed{ {1: 20271701, 2: 20271702, 3: 20271703}[c]}_e2pp.pt") if arm == "DEPLOYED" else C.w_dir(ORG, arm, c) / "h.pt"
    def encode_g161(arm, c):
        if arm == "DEPLOYED": return E2P.ZP["z"].float()
        return torch.load(C.w_dir(ORG, arm, c) / "stage1" / f"{C.ORG[ORG]['pfx']}_ZPRIME.pt", map_location="cpu", weights_only=False)["z"].float()
def programs(arm, c):
    if arm == "DEPLOYED":
        if ORG == "breast": zp = torch.load(F / "results/panel16k_g176/G176_ZPRIME.pt", map_location="cpu", weights_only=False); return list(zp["program_names"]) + [f"free_{i}" for i in range(zp["free_dims"])], json.load(open(C.ORG[ORG]["k0"]))
        return list(E2P.PROGRAMS), json.load(open(F / f"results/{ORG}/{E2P.PFX}_PROGRAM_SETS.json"))
    zp = torch.load(C.w_dir(ORG, arm, c) / "stage1" / f"{C.ORG[ORG]['pfx']}_ZPRIME.pt", map_location="cpu", weights_only=False)
    return list(zp["program_names"]) + [f"free_{i}" for i in range(zp["free_dims"])], json.load(open(C.dict_for(ORG, arm)))
# ── 도너 pseudobulk(팔 무관)
PB = []; idxs = []
for d in donors:
    idx = np.where(dn == d)[0]; idxs.append(idx); s = torch.zeros(cnt.shape[1])
    for j in range(0, len(idx), 4096): s += cnt[torch.as_tensor(idx[j:j + 4096])].float().sum(0)
    PB.append(torch.log1p(s / s.sum().clamp_min(1) * 1e6))
ncell = np.array([[int((ct[idx] == c).sum()) for c in range(Cn)] for idx in idxs])
x = prep(PB).to(DEV)
with torch.no_grad(): q = coarse(x)
Xp = torch.stack(PB).numpy(); Xz = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-6); gi = {g: i for i, g in enumerate(genes)}
def corr(u, v):
    m = np.isfinite(u) & np.isfinite(v); return float(pearsonr(u[m], v[m]).statistic) if m.sum() >= PMIN and np.std(u[m]) > 0 and np.std(v[m]) > 0 else np.nan
def metrics(Mb, Mt, prog, named):
    K = len(prog); R = np.full((Cn, K), np.nan); AGR = np.full((Cn, K), np.nan)
    for c in range(Cn):
        ok = ncell[:, c] >= MIN
        for k in range(K):
            u, v = Mb[ok, c, k], Mt[ok, c, k]; R[c, k] = corr(u, v)
            if ok.sum() >= 3: du = u[:, None] - u[None, :]; dv = v[:, None] - v[None, :]; tri = np.triu_indices(len(u), 1); AGR[c, k] = float(np.mean(np.sign(du[tri]) == np.sign(dv[tri])))
    med = lambda A: float(np.nanmedian(A)) if np.isfinite(A).any() else None
    nk = [k for k, p in enumerate(prog) if p in named]
    return {"per_type": {types[c]: {"n_donors": int((ncell[:, c] >= MIN).sum()), "r_median": med(R[c]), "delta_sign_median": med(AGR[c])} for c in range(Cn)},
            "r_median_all": med(R), "delta_sign_all": med(AGR), "r_median_common_named": med(R[:, nk]) if nk else None, "delta_sign_common_named": med(AGR[:, nk]) if nk else None, "n_common_named": len(nk)}
# 공통 명명 축 = 모든 팔 사전의 교집합(자유 축 제외)
PR = {(a, c): programs(a, c) for a in ARMS for c in chains(a)[:1]}
common = set.intersection(*[set(p for p in v[0] if not p.startswith("free_")) for v in PR.values()])
out = {"organ": ORG, "n_donors": int(len(donors)), "donor_source": "GSE161529 (external)" if ORG == "breast" else "held-out val+test donors", "min_cells": MIN, "pearson_min_donors": PMIN,
       "n_common_named_axes": len(common), "arms": {}, "raw_baseline": {}}
for arm in ARMS:
    per = {}
    for c in chains(arm):
        prog, sets = programs(arm, c); Z = encode_g161(arm, c); Z = Z.float()
        Mt = np.full((len(donors), Cn, Z.shape[1]), np.nan, np.float32)
        for i, idx in enumerate(idxs):
            for cc in range(Cn):
                m = idx[ct[idx] == cc]
                if len(m) >= MIN: Mt[i, cc] = Z[torch.as_tensor(m)].mean(0).numpy()
        ck = torch.load(h_path(arm, c), map_location="cpu", weights_only=False); G = C.arm_genes(ORG, "F" if arm == "DEPLOYED" else arm, genes)[0]
        h = build_h(ck, G); h.load_state_dict(ck["state_dict"]); h.requires_grad_(False); h.eval()
        xin = x if len(G) == len(genes) else x[:, G.to(DEV)]
        with torch.no_grad(): Mh = h(xin, q.detach()).cpu().numpy()
        assert Mh.shape[2] == Mt.shape[2] == len(prog), (Mh.shape, Mt.shape, len(prog))
        per[c] = metrics(Mh, Mt, prog, common)
        if c == chains(arm)[0]:
            raw = np.stack([Xz[:, [gi[g] for g in sets[p] if g in gi]].mean(1) if p in sets and any(g in gi for g in sets[p]) else np.full(len(donors), np.nan) for p in prog], 1)
            Rr = np.full((Cn, len(prog)), np.nan)
            for cc in range(Cn):
                ok = ncell[:, cc] >= MIN
                for k in range(len(prog)): Rr[cc, k] = corr(raw[ok, k], Mt[ok, cc, k])
            out["raw_baseline"][arm] = {"r_median_all": float(np.nanmedian(Rr)), "per_type": {types[cc]: (float(np.nanmedian(Rr[cc])) if np.isfinite(Rr[cc]).any() else None) for cc in range(Cn)}, "chain": c}
        print(f"[{ORG} {arm} c{c}] r {per[c]['r_median_all']:.3f} Δ {per[c]['delta_sign_all']:.3f} · 공통명명 r {per[c]['r_median_common_named']}", flush=True)
    rec = {"chains": chains(arm), "per_chain": per}
    for key in ("r_median_all", "delta_sign_all", "r_median_common_named", "delta_sign_common_named"):
        v = [per[c][key] for c in per if per[c][key] is not None]; rec[key] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)} if v else None
    rec["per_type_mean"] = {t: {"n_donors": per[chains(arm)[0]]["per_type"][t]["n_donors"],
                               "r": (lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} if v else None)([per[c]["per_type"][t]["r_median"] for c in per if per[c]["per_type"][t]["r_median"] is not None]),
                               "delta_sign": (lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} if v else None)([per[c]["per_type"][t]["delta_sign_median"] for c in per if per[c]["per_type"][t]["delta_sign_median"] is not None])} for t in types}
    out["arms"][arm] = rec
json.dump(out, open(OUT / f"{ORG}_chains.json", "w"), ensure_ascii=False, indent=1); print("[saved]", OUT / f"{ORG}_chains.json", flush=True)
