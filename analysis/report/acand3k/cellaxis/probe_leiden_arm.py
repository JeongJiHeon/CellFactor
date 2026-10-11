"""[A_cand3k 사본 — code/train/var_e2pp_liver/organ_probe_leiden_e2pp.py] env ARM(기본 DEPLOYED=원 경로) · CHAINS(1-6): ARM 이 1a 팔이면 체인별 z′(Stage 1 재학습, calibration s 로 나눔)·H(목록 열)를 쓴다.
간·췌장 표준 Leiden 표적 — CP-PCH 프로브(라벨 기준 bag) + 5모델 채점. sstar_all8_organ.py 규약, 표적만 교체.
베이스라인: SIDISH·DEGAS 는 저장 모델 전방 채점, scSurv·Scissor 는 저장된 transductive 점수(군집 정의와 무관) 재사용."""
import os, sys, json
sys.path.insert(0, "archive/recent_code/experiments")
import numpy as np, torch, torch.nn as nn, pandas as pd
torch.set_num_interop_threads = lambda n: None
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from bench_stateonly_cells import fit_bce
from diag_composition_only import Full
from g161.bags import make_bags_from_labels
DEV = torch.device(os.environ.get("DEV", "cuda:2")); ORG = os.environ["ORG"]
import run_organ_e2_prime2 as P2; P2.install(DEV)
import run_organ_e2_prime as E2P
from pathlib import Path
if os.environ.get("E2ROOT"): P2.E2P.ROOT = Path(os.environ["E2ROOT"]); E2P.ROOT = P2.E2P.ROOT
SEEDS = [int(x) for x in os.environ["PSEEDS"].split(",")]
L = E2P.L; LS3 = E2P.LS3
bundle = L.prepare_registry(); E2P.patch_cache(bundle)
a = L.load_a(bundle, DEV); mapping = LS3.fine_mapping(bundle)
data = torch.load(L.STAGE_DATA, map_location="cpu", weights_only=False)
PFX = {"liver": "LIVER", "pancreas": "PANC"}[ORG]; FOCUS = {"liver": "T cells", "pancreas": "Fibroblast"}[ORG]
names = list(data["cell_type_names"]); focus = names.index(FOCUS); genes = [str(g) for g in data["gene_names"]]
ct, donor = data["sc_cell_type"].numpy(), data["sc_donor"].numpy(); train = np.asarray(data["donor_split"]["train"])
H = np.load(os.environ.get("LEIDEN_NPZ", f"data/spatial/leiden_{ORG}.npz")); tf, ev, tr_sid, memb = H["tf"], H["ev"], H["tr_sid"], H["memb_ev"]
Atf, Aev = H["Atf"], H["Aev"]; K = len(np.unique(tr_sid))
assert len(data["sc_donor"]) == len(np.asarray(data["sc_cell_type"]))
t_eval = torch.full((len(ev),), focus, dtype=torch.long, device=DEV)
ARM = os.environ.get("ARM", "DEPLOYED")
AXN, ZSD, COX = [], [], []   # 체인별 z′ 축 이름 · 학습 도너 초점 세포 z′ SD (1a 팔에서만)
# PLAN_PROGRAM_AXIS(2026-10-05) 선택 옵션 — 비우면 기존 동작 그대로:
#   AXIS_MASK = named | free | named_rand4 (+ MASK_REP=r): 판독기 학습·채점 모두 해당 z′ 축만 사용(나머지 축은 0으로 가림)
#   DUMP_W = 경로(.pt): 군집·체인·fit 별 W_focus·α_focus·축 이름·z′ SD 저장
#   RAW_GENE = 1: 같은 bag 으로 유전자 수준 로지스틱 판독기를 맞춰 세포 발현에 적용한 참고 행(rawgene_int) 추가
FAIR = os.environ.get("FAIR") == "1"   # E5(2026-10-07): 위험도 축(Cox W 세포 위험, 방향 무관) + SIDISH·DEGAS 은닉층 bag 판독기
AXIS_MASK = os.environ.get("AXIS_MASK", ""); MASK_REP = int(os.environ.get("MASK_REP", "0")); DUMP_W = os.environ.get("DUMP_W", ""); RAW_GENE = os.environ.get("RAW_GENE") == "1"
assert AXIS_MASK in ("", "named", "free", "named_rand4"), AXIS_MASK
if ARM == "DEPLOYED":
    zp = E2P.ZP["z"].float(); z_eval = zp[ev].to(DEV)
    MEM = [(z_eval, E2P.load_hp(bundle, sd, DEV), E2P.LAT) for sd in SEEDS]
else:   # 1a 팔(예: A_cand3k): 체인마다 자기 z′ 와 H
    sys.path.insert(0, os.path.join(os.environ.get("CELLFACTOR_PROJECT", "."), "code/eval/hvg_run1a")); import common as C1
    def axis_names(C1, ch, LAT):
        """z′ 차원 이름: 체인 헤드 pred_bulk.pt 의 programs/active_dims. 매핑 불가 차원은 'dim{k}'(명명·자유 어디에도 넣지 않음)."""
        hp = C1.run_dir(ORG, ARM, ch) / "pred_bulk.pt"
        if not hp.exists(): return [f"dim{k}" for k in range(LAT)]
        hd = torch.load(hp, map_location="cpu", weights_only=False); pr = [str(x) for x in (hd.get("programs") or [])]; ad = hd.get("active_dims")
        if len(pr) == LAT and (ad is None or len(ad) == LAT): return pr
        nm = [f"dim{k}" for k in range(LAT)]
        if ad is not None and len(pr) == len(ad):
            for i, k in enumerate(np.asarray(ad).astype(int).tolist()): nm[k] = pr[i]
        return nm
    c0, c1_ = (int(v) for v in os.environ.get("CHAINS", "1-6").split("-")); G, gsha = C1.arm_genes(ORG, ARM, genes)
    bh = dict(bundle); bh["gene_names"] = [genes[i] for i in G.tolist()]; MEM = []
    for ch in range(c0, c1_ + 1):
        zpp = torch.load(C1.w_dir(ORG, ARM, ch) / "stage1" / f"{PFX}_ZPRIME.pt", map_location="cpu", weights_only=False)
        s_ = torch.tensor(json.load(open(C1.run_dir(ORG, ARM, ch) / "calibration.json"))["s"]).float()
        zc = zpp["z"].float() / s_[None, :]; assert len(zc) == len(ct)
        ck = torch.load(C1.w_dir(ORG, ARM, ch) / "h.pt", map_location="cpu", weights_only=False); hm = L.build_h(bh, DEV, ck["model_config"]); hm.load_state_dict(ck["state_dict"]); hm.eval().requires_grad_(False)
        Gd = G.to(DEV)
        MEM.append((zc[ev].to(DEV), (lambda x, q, hm=hm, Gd=Gd: hm(x.index_select(1, Gd), q)), zc.shape[1]))
        AXN.append(axis_names(C1, ch, zc.shape[1])); ZSD.append(zc[tf].std(0).numpy())
        if os.environ.get("FAIR") == "1":
            pbk = torch.load(C1.run_dir(ORG, ARM, ch) / "pred_bulk.pt", map_location="cpu", weights_only=False)
            COX.append((float(pbk["alpha"][focus]), pbk["W"][focus].float().numpy()))
    print(f"[arm] {ARM} 체인 {c0}-{c1_} · 목록 {len(G)} ({gsha}) · LAT {MEM[0][2]}", flush=True)
MASKS, MASK_NAMES = [None] * len(MEM), None
if AXIS_MASK or DUMP_W:
    assert AXN, "AXIS_MASK/DUMP_W 는 1a 팔(ARM≠DEPLOYED)에서만 지원"
    is_free = lambda n: n.startswith("free")
    is_named = lambda n: not is_free(n) and not n.startswith("dim")
    if AXIS_MASK == "named_rand4":
        common_named = sorted(set.intersection(*[{n for n in nm if is_named(n)} for nm in AXN]))
        MASK_NAMES = sorted(np.random.default_rng(1000 + MASK_REP).choice(common_named, 4, replace=False).tolist())
    if AXIS_MASK:
        sel = {"named": is_named, "free": is_free, "named_rand4": lambda n: n in MASK_NAMES}[AXIS_MASK]
        MASKS = [torch.tensor([1.0 if sel(n) else 0.0 for n in nm], device=DEV) for nm in AXN]
        print(f"[mask] {AXIS_MASK} rep {MASK_REP} · 축 수 체인별 {[int(m.sum()) for m in MASKS]}" + (f" · {MASK_NAMES}" if MASK_NAMES else ""), flush=True)
        assert all(m.sum() > 0 for m in MASKS), "빈 마스크"
DUMP = {"arm": ARM, "organ": ORG, "axis_names": AXN, "z_sd_train_focus": ZSD, "axis_mask": AXIS_MASK, "mask_names": MASK_NAMES, "clusters": {}}
# ── 베이스라인 세포 점수 (군집 정의와 무관) ──
counts = data["sc_counts"]; ce = counts[torch.as_tensor(ev)].float()
class DC(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.encoder_layer = nn.Sequential(nn.Sequential(nn.Linear(G, 512), nn.Softplus(), nn.Linear(512, 128), nn.Softplus()), nn.Linear(128, 32))
        self.new_layer = nn.Linear(32, 128); self.final_layer = nn.Linear(128, 1, bias=False); self.t = nn.Tanh()
    def forward(self, x): return self.final_layer(self.t(self.new_layer(self.t(self.encoder_layer(x)))))
nat = torch.log1p(ce / ce.sum(1, keepdim=True).clamp_min(1) * 1e6); x6 = torch.log1p(torch.expm1(torch.nan_to_num(nat)) / 100.0)
sid = []
for sd in range(3):
    ck = torch.load(f"results/{ORG}/{PFX}_SIDISH_OFFICIAL_deepcox_seed{sd}.pt", map_location="cpu", weights_only=False)
    m = DC(len(ck["hvg"])); m.load_state_dict(ck["state_dict"], strict=True); m.eval()
    with torch.no_grad(): sid.append(m(x6[:, ck["hvg"]]).flatten().numpy())
import scanpy as sc_, anndata as ad
def prep(m):
    x = np.power(1.5, np.log2(m + 1.0)); x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-3)
    return ((x - x.min(1, keepdims=True)) / (x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-3)).astype(np.float32)
def fwd(X, d):
    Th = [np.loadtxt(f"{d}/Theta{i}.csv", delimiter=",") for i in range(1, 5)]; Bi = [np.loadtxt(f"{d}/Bias{i}.csv", delimiter=",") for i in range(1, 5)]
    h = X
    for k in range(3): h = 1 / (1 + np.exp(-(h @ Th[k] + Bi[k])))
    return (1 / (1 + np.exp(-(h @ Th[3] + Bi[3])))).ravel()
pan = np.load(f"results/baseline_cellscores/scissor_{ORG}/panel_idx.npy"); cp_ = counts[torch.as_tensor(pan)].float()
cp10k = torch.log1p(cp_ / cp_.sum(1, keepdim=True).clamp_min(1) * 1e4)
A = ad.AnnData(cp10k.numpy().astype(np.float32)); A.var_names = genes; sc_.pp.highly_variable_genes(A, flavor="seurat")
hvg = np.where(A.var.highly_variable.values)[0][:2000]
ref = np.loadtxt(f"results/baseline_cellscores/degas_{ORG}/scExp.csv", delimiter=",", skiprows=1)
err = float(np.abs(prep(torch.expm1(cp10k)[:, hvg].numpy()) - ref).max()); assert err < 1e-4, err
Xe = prep(torch.expm1(torch.log1p(ce / ce.sum(1, keepdim=True).clamp_min(1) * 1e4))[:, hvg].numpy())
deg = [fwd(Xe, f"results/baseline_cellscores/degas_{ORG}/s{k}") for k in range(3)]
sv = np.load(f"results/report/EXT_SCSURV_EVSCORES_{ORG}.npz"); assert np.array_equal(sv["ev"], ev); scs = list(sv["scores"])
lab = pd.read_csv(f"data/baselines/{ORG}/scissor_ext/scissor_labels.csv")["label"].to_numpy(float); sci = [lab[len(lab) - len(ev):]]
BASE = {"scSurv": scs, "SIDISH": sid, "Scissor": sci, "DEGAS": deg}
if FAIR:
    SID_M = []
    for sd in range(3):
        ck = torch.load(f"results/{ORG}/{PFX}_SIDISH_OFFICIAL_deepcox_seed{sd}.pt", map_location="cpu", weights_only=False)
        m = DC(len(ck["hvg"])); m.load_state_dict(ck["state_dict"], strict=True); m.eval(); SID_M.append((m, np.asarray(ck["hvg"])))
    def sid_hidden(m, xin):   # 출력 직전 은닉층(점수 = final_layer(이 표현))
        with torch.no_grad(): return m.t(m.new_layer(m.t(m.encoder_layer(xin)))).numpy()
    def degas_hidden(X, d):
        Th = [np.loadtxt(f"{d}/Theta{i}.csv", delimiter=",") for i in range(1, 5)]; Bi = [np.loadtxt(f"{d}/Bias{i}.csv", delimiter=",") for i in range(1, 5)]
        h = X
        for k in range(3): h = 1 / (1 + np.exp(-(h @ Th[k] + Bi[k])))
        return h
    x6e = x6   # 세포: log1p CP10k
    SID_E = [sid_hidden(m, x6e[:, hv]) for m, hv in SID_M]
    DEG_E = [degas_hidden(Xe, f"results/baseline_cellscores/degas_{ORG}/s{k}") for k in range(3)]
    def fit_lin(Htr, y, seed):
        torch.manual_seed(seed); Ht = torch.tensor(Htr, dtype=torch.float32); yt = torch.as_tensor(y, dtype=torch.float32)
        w = torch.zeros(Ht.shape[1], requires_grad=True); a0 = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([w, a0], lr=1e-2, weight_decay=1e-4)
        for _ in range(500):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a0 + Ht @ w, yt) + 0.05 * w.square().sum(); opt.zero_grad(); loss.backward(); opt.step()
        return w.detach().numpy(), float(a0)
print(f"[{ORG}] Leiden K={K} · 평가 {len(ev)} · DEGAS scExp 오차 {err:.1e}", flush=True)
out = {"arm": ARM, "organ": ORG, "focus": FOCUS, "space": "leiden", "n_int": len(ev), "clusters": {}}
for c in range(K):
    n_tr, n_ev = int((tr_sid == c).sum()), int((memb == c).sum())
    if n_tr < 200 or n_ev < 20 or n_ev > len(ev) - 20: print(f"군집{c}: skip (train {n_tr}, eval {n_ev})", flush=True); continue
    pb, prop, yb = make_bags_from_labels(data, focus, tf, tr_sid, c, train, n_bags=240, N=512, seed=0)
    x = L.standardize_online(bundle, pb); yb = yb.to(DEV); mem = (memb == c).astype(float)
    with torch.no_grad(): co, rel = a(x.to(DEV)); fine = LS3.make_fine(co, rel, mapping, len(names))
    aucs = []; dW = []
    for i_m, (z_eval, h, LATm) in enumerate(MEM):
        with torch.no_grad(): st = h(x.to(DEV), co.detach())
        mk = MASKS[i_m]
        if mk is not None: st = st * mk; z_eval = z_eval * mk   # 가린 축은 판독기 입력·채점 모두 0
        loc = []; wl = []
        for fs in range(3):
            scm = fit_bce(Full(len(names), LATm).to(DEV), fine, st, yb, seed=fs)
            with torch.no_grad(): s_ = (scm.alpha[t_eval] + (scm.W[t_eval] * z_eval).sum(1)).cpu().numpy()
            loc.append(float(roc_auc_score(mem, s_)))
            if DUMP_W: wl.append({"W_focus": scm.W[focus].detach().cpu().numpy(), "alpha_focus": float(scm.alpha[focus]), "auc": loc[-1]})
        aucs.append(float(np.mean(loc))); dW.append(wl)
    if DUMP_W: DUMP["clusters"][str(c)] = {"fits": dW, "n_train": n_tr, "n_int": n_ev}
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Atf, (tr_sid == c).astype(int))
    row = {"CP-PCH_int": [float(np.mean(aucs)), float(np.std(aucs))], "ceiling_int": float(roc_auc_score(mem, lr.decision_function(Aev))),
           "n_train": n_tr, "n_int": n_ev}
    for k, sc in BASE.items():
        vv = [float(roc_auc_score(mem, s)) for s in sc]; row[f"{k}_int"] = [float(np.mean(vv)), float(np.std(vv))]
    if FAIR:
        if COX:
            hz = [float(roc_auc_score(mem, a_ + z_eval_c.cpu().numpy() @ w_)) for (a_, w_), (z_eval_c, _, _) in zip(COX, MEM)]
            row["hazard_CF_int"] = hz; row["hazard_CF_abs"] = float(np.mean([abs(v - 0.5) for v in hz]))
        for k_, sc_ in BASE.items(): row[f"{k_}_abs"] = float(np.mean([abs(float(roc_auc_score(mem, s_)) - 0.5) for s_ in sc_]))
        pbc = torch.expm1(pb.cpu()).clamp_min(0) / 100.0   # bag: log1p CPM → CP10k
        x6b = torch.log1p(pbc); yb_np = yb.cpu().numpy()
        pv = []
        for (m, hv), He in zip(SID_M, SID_E):
            Hb = sid_hidden(m, x6b[:, hv])
            for fs in range(3): w_, a_ = fit_lin(Hb, yb_np, fs); pv.append(float(roc_auc_score(mem, a_ + He @ w_)))
        row["probe_SIDISH_int"] = [float(np.mean(pv)), float(np.std(pv))]
        Xb = prep(pbc[:, hvg].numpy()); pv = []
        for k, He in enumerate(DEG_E):
            Hb = degas_hidden(Xb, f"results/baseline_cellscores/degas_{ORG}/s{k}")
            for fs in range(3): w_, a_ = fit_lin(Hb, yb_np, fs); pv.append(float(roc_auc_score(mem, a_ + He @ w_)))
        row["probe_DEGAS_int"] = [float(np.mean(pv)), float(np.std(pv))]
    if RAW_GENE:   # 참고: 같은 bag 의 표준화 유전자 → 로지스틱(L2) → 평가 세포의 같은 표준화 발현에 적용
        if "XE_STD" not in globals():
            XE_STD = L.standardize_online(bundle, torch.log1p(ce / ce.sum(1, keepdim=True).clamp_min(1) * 1e6)).numpy()
        rg = LogisticRegression(max_iter=3000, C=1.0).fit(x.cpu().numpy(), yb.cpu().numpy().astype(int))
        row["rawgene_int"] = float(roc_auc_score(mem, rg.decision_function(XE_STD)))
    out["clusters"][str(c)] = row
    print(f"군집{c} (tr {n_tr}/ev {n_ev}): CP-PCH {row['CP-PCH_int'][0]:.3f} / 상한 {row['ceiling_int']:.3f} · " + " ".join(f"{k} {row[k+'_int'][0]:.3f}" for k in BASE), flush=True)
cl = out["clusters"]
for k in ["CP-PCH"] + list(BASE):
    v = [r[f"{k}_int"][0] for r in cl.values()]; out[f"{k}_int_mean"] = [float(np.mean(v)), float(np.std(v)), len(v)]
out["ceiling_int_mean"] = float(np.mean([r["ceiling_int"] for r in cl.values()]))
print(f"[{ORG} leiden] " + " · ".join(f"{k} {out[k+'_int_mean'][0]:.3f}±{out[k+'_int_mean'][1]:.3f}" for k in ["CP-PCH"] + list(BASE)) + f" · 적격 {len(cl)}/{K}", flush=True)
if FAIR:
    for k_ in ["hazard_CF_abs"] + [f"{m}_abs" for m in BASE]:
        v = [r[k_] for r in cl.values() if k_ in r]
        if v: out[f"{k_}_mean"] = [float(np.mean(v)), float(np.std(v)), len(v)]
    for k_ in ("probe_SIDISH_int", "probe_DEGAS_int"):
        v = [r[k_][0] for r in cl.values()]; out[f"{k_}_mean"] = [float(np.mean(v)), float(np.std(v)), len(v)]
    print("[E5] " + " · ".join(f"{k}={out[k][0]:.3f}" for k in out if k.endswith("_mean") and ("abs" in k or "probe" in k)), flush=True)
if RAW_GENE: out["rawgene_int_mean"] = [float(np.mean([r["rawgene_int"] for r in cl.values()])), float(np.std([r["rawgene_int"] for r in cl.values()])), len(cl)]
out["axis_mask"] = AXIS_MASK; out["mask_rep"] = MASK_REP; out["mask_names"] = MASK_NAMES
if DUMP_W: torch.save(DUMP, DUMP_W); print(f"[saved] {DUMP_W}", flush=True)
out["K"] = int(K); out["leiden_npz"] = os.environ.get("LEIDEN_NPZ", f"data/spatial/leiden_{ORG}.npz"); out["leiden_seed"] = int(H["leiden_seed"]) if "leiden_seed" in H.files else None
OUTJ = os.environ["LEIDEN_JSON"]   # 사본: 기본 경로(results/report) 쓰기 금지
json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
print(f"[saved] {OUTJ}", flush=True)
