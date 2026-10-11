#!/usr/bin/env python3
"""PLAN_DOMAIN_GAP — (A) 유사벌크 대 TCGA 실제 벌크 입력 비교, (B) CORAL 켬(A_cand3k)/끔(A_cand3k__NO_CORAL)의 H 표현 정렬·상태 보존·예후.
재학습 없음. env ORG · DEV. 출력 results/mech/domain_gap/{ORG}.json, {ORG}_fig.npz (+ results_json 미러)"""
import os, sys, json, math
ORG = os.environ["ORG"]; DEV_S = os.environ.get("DEV", "cuda:0"); assert DEV_S in ("cuda:0", "cuda:1")
F = os.environ.get("CELLFACTOR_PROJECT", "."); sys.path.insert(0, f"{F}")
import numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import train as T
from model import load_state_estimator, conditional_coral
F = Path(F); OUT = F / "results/mech/domain_gap"; OUT.mkdir(parents=True, exist_ok=True); OJ = F / "results_json/mech/domain_gap"; OJ.mkdir(parents=True, exist_ok=True)
dev = torch.device(DEV_S); cfg = T.ORGANS[ORG]; RUN = F / "results/hvg_design/run1a" / ORG; WTS = F / "weights/hvg_run1a" / ORG
bundle = T.load(cfg["registry"]); genes = [str(g) for g in bundle["gene_names"]]
G = T.index_of(T.panel_genes(cfg), genes).numpy()
A = T.load_a(cfg, None, dev)
res = {"organ": ORG}; fig = {}


def auroc_cv(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0); out = {}
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    out["logistic"] = float(roc_auc_score(y, cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]))
    rf = RandomForestClassifier(500, random_state=0, n_jobs=8)
    out["random_forest"] = float(roc_auc_score(y, cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]))
    return out


def pca(X, k=10):
    Z = (X - X.mean(0)) / X.std(0).clip(1e-8); U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return U[:, :k] * S[:k], (S ** 2 / (S ** 2).sum())[:k]


def sep_auc(score, y):
    a = roc_auc_score(y, score); return float(max(a, 1 - a))


# ============================================================ A. 입력 공간
D = T.load(cfg["stage1_data"], mmap=True)
smp = T.BagSampler(D, torch.zeros(len(D["sc_counts"]), 1), bundle["donor_split"]["train"], len(bundle["cell_type_names"]), bundle, 1024, cfg["bag_seeds"]["trainsel"], dev)
raw_list = []; _std = smp._expression
def _raw(idx):
    s = smp.counts[idx.to(smp.dev)].sum(0, dtype=torch.float32); ln = torch.log1p(s / s.sum().clamp_min(1) * 1e6).cpu(); raw_list.append(ln / math.log(2.0))
    return ((ln / math.log(2.0) - smp.mean) / smp.sd).clamp(-smp.clamp, smp.clamp)
smp._expression = _raw
bags = smp.sample(cfg["bag_n"]["trainsel"])
P_raw_all = torch.stack(raw_list[0::2]).numpy()          # draw A 만 (A, B 순서로 쌓임)
stored = torch.load(WTS / "A_cand3k/c1/materialized/paired_trainsel.pt", map_location="cpu", weights_only=False)
ident = float((bags["expression_a"] - stored["expression_a"]).abs().max()); assert ident < 1e-4, ident
res["A_identity_maxabs"] = ident
v = bundle["bulk_log2_cpm"].double().numpy(); lin = np.power(2.0, v) - 1.0          # 간·췌장 = count, 유방 = CPM
R_raw_all = np.log2(1 + lin / lin.sum(1, keepdims=True) * 1e6)
P_raw, R_raw = P_raw_all[:, G], R_raw_all[:, G]
P_in, R_in = stored["expression_a"].numpy()[:, G].astype(np.float64), bundle["bulk_x"].double().numpy()[:, G]
y = np.r_[np.zeros(len(P_raw)), np.ones(len(R_raw))]
with torch.no_grad(): qR = torch.cat([A(bundle["bulk_x"][i:i + 256].float().to(dev))[0].cpu() for i in range(0, len(R_raw), 256)]).numpy()
qP = stored["coarse_a"].numpy(); lo, hi = np.percentile(qP, 5, 0), np.percentile(qP, 95, 0); ov = np.all((qR >= lo) & (qR <= hi), 1)
res["A"] = {"n_P": int(len(P_raw)), "n_R": int(len(R_raw)), "n_R_overlap": int(ov.sum()), "views": {}}
for view, (Pv, Rv) in {"raw_log2cpm": (P_raw, R_raw), "model_input": (P_in, R_in)}.items():
    X = np.r_[Pv, Rv]; pcs, ve = pca(X); fig[f"A_{view}_pc"] = pcs[:, :2]; fig[f"A_{view}_ve"] = ve
    mP, mR = Pv.mean(0), Rv.mean(0)
    r = {"pca_var_explained_pc1_2": [float(ve[0]), float(ve[1])], "pc_domain_auc": [sep_auc(pcs[:, k], y) for k in range(10)],
         "gene_mean_pearson": float(np.corrcoef(mP, mR)[0, 1]), "gene_mean_spearman": float(np.corrcoef(np.argsort(np.argsort(mP)), np.argsort(np.argsort(mR)))[0, 1]),
         "value_quantiles_P": np.percentile(Pv, [5, 25, 50, 75, 95]).tolist(), "value_quantiles_R": np.percentile(Rv, [5, 25, 50, 75, 95]).tolist(),
         "gene_sd_ratio_R_over_P_median": float(np.median(Rv.std(0) / Pv.std(0).clip(1e-8))),
         "auroc": auroc_cv(X, y), "auroc_overlap": auroc_cv(np.r_[Pv, Rv[ov]], np.r_[np.zeros(len(Pv)), np.ones(int(ov.sum()))])}
    if view == "raw_log2cpm":
        detP, detR = (Pv > 0).mean(0), (Rv > 0).mean(0)
        r.update({"zero_fraction_P": float((Pv == 0).mean()), "zero_fraction_R": float((Rv == 0).mean()),
                  "detection_rate_median_P": float(np.median(detP)), "detection_rate_median_R": float(np.median(detR)),
                  "genes_detected_in_lt50pct_P": int((detP < .5).sum()), "genes_detected_in_lt50pct_R": int((detR < .5).sum())})
        fig["A_gene_mean_P"], fig["A_gene_mean_R"] = mP, mR; fig["A_det_P"], fig["A_det_R"] = detP, detR
        fig["A_gene_sd_P"], fig["A_gene_sd_R"] = Pv.std(0), Rv.std(0)
    else:
        r["clamp_fraction_P"], r["clamp_fraction_R"] = float((np.abs(Pv) >= 6).mean()), float((np.abs(Rv) >= 6).mean())
    res["A"]["views"][view] = r
    print(f"[{ORG} A {view}] PC1/2 {ve[0]:.3f}/{ve[1]:.3f} · AUROC {r['auroc']} · overlap {r['auroc_overlap']} · gene-mean r {r['gene_mean_pearson']:.3f}", flush=True)
fig["A_y"] = y

# ============================================================ B. CORAL 켬/끔
pb1 = torch.load(RUN / "A_cand3k/c1/pred_bulk.pt", map_location="cpu", weights_only=False)
held = pb1["heldout_idx"].numpy()
fit = np.asarray(bundle["bulk_split"]["train"]) if cfg["real_rows"] == "train" else T.tcga_fit_val_split(bundle)[0]
ho_b = {r: torch.load(WTS / f"A_cand3k/c1/materialized/paired_{r}.pt", map_location="cpu", weights_only=False) for r in ("val", "test")}
X_ho = torch.cat([ho_b["val"]["expression_a"], ho_b["test"]["expression_a"]]); q_ho = torch.cat([ho_b["val"]["coarse_a"], ho_b["test"]["coarse_a"]])
X_P = stored["expression_a"]; q_P = stored["coarse_a"]; X_R = bundle["bulk_x"].float(); q_Rt = torch.as_tensor(qR)


@torch.no_grad()
def enc(h, X, q_unused=None):
    return torch.cat([h.encode(X[i:i + 256][:, G].to(dev)).cpu() for i in range(0, len(X), 256)]).double()


def pooled_resid(fp, qp, fr, qr):
    f = torch.cat([fp, fr]); q = torch.cat([qp, qr]).double(); d = torch.cat([torch.ones(len(q), 1, dtype=torch.float64), q[:, :2]], 1)
    beta = torch.linalg.lstsq(d, f).solution; e = (f - d @ beta).numpy(); return e[:len(fp)], e[len(fp):]


def b_metrics(fp, qp, fr, qr, auroc=True):
    out = {"coral_objective": float(conditional_coral(fp, qp.double(), fr, qr.double()))}
    ep, er = pooled_resid(fp, qp, fr, qr); sd = np.r_[ep, er].std(0).clip(1e-8)
    out["mean_diff"] = float(np.mean(np.abs(ep.mean(0) - er.mean(0)) / sd))
    Cp, Cr = np.cov(ep.T), np.cov(er.T); out["cov_rel_diff"] = float(np.linalg.norm(Cp - Cr) / np.linalg.norm(Cr))
    if auroc: out["auroc"] = auroc_cv(np.r_[ep, er], np.r_[np.zeros(len(ep)), np.ones(len(er))])
    return out, ep, er


ARMS = {"CORAL_on": "A_cand3k", "CORAL_off": "A_cand3k__NO_CORAL"}
res["B"] = {a: {} for a in ARMS}
for arm, d in ARMS.items():
    for ch in range(1, 7):
        h = load_state_estimator(WTS / f"{d}/c{ch}/h.pt", dev)
        fP, fho, fR = enc(h, X_P), enc(h, X_ho), enc(h, X_R)
        row = {}
        row["P_vs_Runseen"], ep, er = b_metrics(fP, q_P, fR[held], q_Rt[held])
        row["P_vs_Rfit"], _, _ = b_metrics(fP, q_P, fR[fit], q_Rt[fit])
        row["Pho_vs_Runseen"], _, _ = b_metrics(fho, q_ho, fR[held], q_Rt[held])
        if ch == 1:
            pcs, ve = pca(np.r_[ep, er]); fig[f"B_{arm}_pc"] = pcs[:, :2]; fig[f"B_{arm}_ve"] = ve; fig["B_y"] = np.r_[np.zeros(len(ep)), np.ones(len(er))]
        ev = json.load(open(RUN / f"{d}/c{ch}/EVAL.json"))
        row["state"] = {"test_wmse": ev["state_zspace"]["test_wmse"], "mae_ratio_mean": ev["state_zspace"]["mae_ratio_mean"], "mae_ratio_by_type": ev["state_zspace"]["mae_ratio_by_type"]}
        res["B"][arm][ch] = row
        m = row["P_vs_Runseen"]; print(f"[{ORG} B {arm} c{ch}] CORAL obj {m['coral_objective']:.4f} · mean diff {m['mean_diff']:.3f} · cov rel {m['cov_rel_diff']:.3f} · AUROC {m['auroc']} · state MAE ratio {row['state']['mae_ratio_mean']:.3f}", flush=True)

# ---- B6 예후: 앙상블 C 와 짝 bootstrap
EXT = {"breast": [("SCAN-B", e) for e in ("OS", "RFi", "DRFi", "BCFi")] + [("METABRIC", "OS"), ("METABRIC", "RFS")],
       "liver": [("GSE14520", "OS"), ("GSE14520", "RFS"), ("GSE76427", "OS"), ("GSE76427", "RFS")],
       "pancreas": [("GSE62452", "OS"), ("GSE57495", "OS"), ("GSE85916", "OS")]}[ORG]
zf = lambda r: (r - r.mean()) / (r.std() + 1e-12)
ENS, END = {}, {}
for arm, d in ARMS.items():
    P = [torch.load(RUN / f"{d}/c{ch}/pred_bulk.pt", map_location="cpu", weights_only=False) for ch in range(1, 7)]
    ENS[arm] = {c: np.mean([zf(np.asarray(p["risk_cohorts"][c], float)) for p in P], 0) for c in sorted({c for c, _ in EXT})}
    ENS[arm]["TCGA-heldout"] = np.mean([zf(np.asarray(p["risk_all"], float)[p["heldout_idx"].numpy()]) for p in P], 0)
for c, e in EXT:
    v = pb1["endpoints"][c][e]; END[f"{c}|{e}"] = (c, np.asarray(v["indices"], int), np.asarray(v["time"], float), np.asarray(v["event"], float))
END["TCGA-heldout|OS"] = ("TCGA-heldout", np.arange(len(held)), pb1["time_all"].numpy()[held].astype(float), pb1["event_all"].numpy()[held].astype(float))
ncoh = {c: len(ENS["CORAL_on"][c]) for c in ENS["CORAL_on"]}
INV = {}
for k, (c, ix, t, e) in END.items():
    assert len(np.unique(ix)) == len(ix); inv = -np.ones(ncoh[c], int); inv[ix] = np.arange(len(ix)); INV[k] = inv


def evaluate(draw):
    out = {}
    for arm in ARMS:
        vals = {}
        for k, (c, ix, t, e) in END.items():
            pos = np.arange(len(ix)) if draw is None else INV[k][draw[c]]; pos = pos[pos >= 0]; pos = pos[np.isfinite(t[pos]) & (t[pos] > 0) & np.isfinite(e[pos])]
            vals[k] = concordance_index(t[pos], -ENS[arm][c][ix[pos]], e[pos])
        out[arm] = {"external_mean": float(np.mean([vals[f"{c}|{e}"] for c, e in EXT])), "heldout": vals["TCGA-heldout|OS"]}
    return out


pt = evaluate(None)
PAPER = {"breast": float(os.environ.get("C_REF_BREAST", "0.6175")), "liver": 0.600, "pancreas": 0.621}[ORG]   # 유방 2026-10-11 α 자유(PLAN_ALPHA_FREE)
assert abs(pt["CORAL_on"]["external_mean"] - PAPER) < 1.5e-3, (pt, PAPER)
rng = np.random.default_rng(0); BS = [evaluate({c: rng.integers(0, n, n) for c, n in ncoh.items()}) for _ in range(2000)]
res["B6"] = {"point": pt}
for k in ("external_mean", "heldout"):
    dd = np.array([b["CORAL_on"][k] - b["CORAL_off"][k] for b in BS]); lo_, hi_ = np.percentile(dd, [2.5, 97.5])
    res["B6"][k] = {"diff_on_minus_off": pt["CORAL_on"][k] - pt["CORAL_off"][k], "ci95": [float(lo_), float(hi_)],
                    "ci95_on": np.percentile([b["CORAL_on"][k] for b in BS], [2.5, 97.5]).tolist(), "ci95_off": np.percentile([b["CORAL_off"][k] for b in BS], [2.5, 97.5]).tolist()}

# ---- 판정
on, off = res["B"]["CORAL_on"], res["B"]["CORAL_off"]; chs = range(1, 7)
obj_lower = sum(on[c]["P_vs_Runseen"]["coral_objective"] < off[c]["P_vs_Runseen"]["coral_objective"] for c in chs)
auc_lower = sum(on[c]["P_vs_Runseen"]["auroc"]["logistic"] < off[c]["P_vs_Runseen"]["auroc"]["logistic"] for c in chs)
jb = "CORAL reduces domain difference" if obj_lower >= 5 and auc_lower >= 5 else ("partial" if obj_lower >= 5 or auc_lower >= 5 else "not confirmed")
sm_on = np.array([on[c]["state"]["mae_ratio_mean"] for c in chs]); sm_off = np.array([off[c]["state"]["mae_ratio_mean"] for c in chs]); rel = np.median(sm_on / sm_off - 1)
js = "state information loss" if ((sm_on > sm_off).sum() >= 5 and rel > .05) else ("state recovery improved" if ((sm_on < sm_off).sum() >= 5 and rel < -.05) else "small difference")
ci = res["B6"]["external_mean"]["ci95"]; jp = "CORAL higher" if ci[0] > 0 else ("CORAL lower" if ci[1] < 0 else "no difference detected")
res["judgement"] = {"J_B": jb, "coral_obj_lower_chains": obj_lower, "auroc_lower_chains": auc_lower, "J_S": js, "state_rel_change_median": float(rel),
                    "state_on_higher_chains": int((sm_on > sm_off).sum()), "J_P": jp}
for dd in (OUT, OJ): json.dump(res, open(dd / f"{ORG}.json", "w"), indent=1)
np.savez_compressed(OUT / f"{ORG}_fig.npz", **{k: np.asarray(v) for k, v in fig.items()}, A_ov=ov)
print(json.dumps({"judgement": res["judgement"], "B6": res["B6"]}, indent=1))
