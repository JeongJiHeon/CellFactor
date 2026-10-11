#!/usr/bin/env python3
"""PLAN_DOMAIN_GAP 사후 진단(결과 확인 후, 변경 기록 참조) — 세 질문.
 Q1 CORAL 목적값은 주는데 공동 잔차 평균 차는 왜 커지는가: 목적값을 평균 항·공분산 항으로 분해, 원 단위 평균 차와 척도(합동 SD) 분리.
 Q2 CORAL 뒤에도 분류 AUROC 가 1 에 가까운 이유: 도메인별 잔차(평균 제거)에서 AUROC, 대조(유사벌크를 무작위 반으로 나눔).
 Q3 모델 입력 로지스틱 AUROC < 0.5: 라벨·점수 방향 확인, fold 안에서 도메인 표준화를 다시 해 누출 제거 후 AUROC, 라벨 순열 귀무.
 + 그림용: 유사벌크로만 맞춘 PCA 좌표(유사벌크 PC SD 단위)에 실제 벌크를 사영 — 두 모델을 같은 정의로 비교.
env ORG · DEV. 출력 results/mech/domain_gap/{ORG}_diag.json, {ORG}_diag_fig.npz"""
import os, sys, json
ORG = os.environ["ORG"]; DEV_S = os.environ.get("DEV", "cuda:0")
F = os.environ.get("CELLFACTOR_PROJECT", "."); sys.path.insert(0, f"{F}")
import numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import train as T
from model import load_state_estimator
F = Path(F); OUT = F / "results/mech/domain_gap"; dev = torch.device(DEV_S); cfg = T.ORGANS[ORG]; RUN = F / "results/hvg_design/run1a" / ORG; WTS = F / "weights/hvg_run1a" / ORG
bundle = T.load(cfg["registry"]); G = T.index_of(T.panel_genes(cfg), [str(g) for g in bundle["gene_names"]]).numpy()
A = T.load_a(cfg, None, dev)
P = torch.load(WTS / "A_cand3k/c1/materialized/paired_trainsel.pt", map_location="cpu", weights_only=False)
pb1 = torch.load(RUN / "A_cand3k/c1/pred_bulk.pt", map_location="cpu", weights_only=False); held = pb1["heldout_idx"].numpy()
X_R = bundle["bulk_x"].float()
with torch.no_grad(): qR = torch.cat([A(X_R[i:i + 256].to(dev))[0].cpu() for i in range(0, len(X_R), 256)]).double()
X_P, qP = P["expression_a"], P["coarse_a"].double()
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def lr_auc(X, y):
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return float(roc_auc_score(y, cross_val_predict(lr, X, y, cv=CV, method="predict_proba")[:, 1]))


def design(q): return torch.cat([torch.ones(len(q), 1, dtype=torch.float64), q[:, :2]], 1)
def resid(f, q):
    d = design(q); return f - d @ torch.linalg.lstsq(d, f).solution


res = {"organ": ORG}
# ---------------- Q3 모델 입력 로지스틱 AUROC
Xp_in, Xr_in = X_P.numpy()[:, G].astype(np.float64), X_R.numpy()[:, G].astype(np.float64)
X = np.r_[Xp_in, Xr_in]; y = np.r_[np.zeros(len(Xp_in)), np.ones(len(Xr_in))]
q3 = {"label_coding": "0 = pseudo-bulk, 1 = TCGA; score = P(TCGA)", "auc_as_reported": lr_auc(X, y),
      "per_domain_gene_mean_abs_max": [float(np.abs(Xp_in.mean(0)).max()), float(np.abs(Xr_in.mean(0)).max())]}
# 누출 없는 판: fold 마다 학습 fold 의 도메인별 평균·SD 로 두 도메인을 각각 다시 표준화(모델의 도메인별 표준화를 fold 안에서 재현)
s = np.zeros(len(y))
for tr, te in CV.split(X, y):
    Xt = X.copy()
    for d_ in (0, 1):
        m = tr[y[tr] == d_]; mu, sd = X[m].mean(0), X[m].std(0).clip(1e-6); idx = np.where(y == d_)[0]; Xt[idx] = (X[idx] - mu) / sd
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xt[tr], y[tr]); s[te] = clf.predict_proba(Xt[te])[:, 1]
q3["auc_foldwise_restandardised"] = float(roc_auc_score(y, s))
rng = np.random.default_rng(0); q3["auc_label_permutation"] = [lr_auc(X, rng.permutation(y)) for _ in range(5)]
res["Q3_model_input_logistic"] = q3
print(f"[{ORG} Q3] reported {q3['auc_as_reported']:.3f} · fold-wise restandardised {q3['auc_foldwise_restandardised']:.3f} · permuted {np.round(q3['auc_label_permutation'], 3)}", flush=True)

# ---------------- Q1·Q2 CORAL 켬/끔
ARMS = {"CORAL_off": "A_cand3k__NO_CORAL", "CORAL_on": "A_cand3k"}; fig = {}
res["arms"] = {a: {} for a in ARMS}
for arm, d in ARMS.items():
    for ch in range(1, 7):
        h = load_state_estimator(WTS / f"{d}/c{ch}/h.pt", dev)
        with torch.no_grad():
            fP = torch.cat([h.encode(X_P[i:i + 256][:, G].to(dev)).cpu() for i in range(0, len(X_P), 256)]).double()
            fR = torch.cat([h.encode(X_R[i:i + 256][:, G].to(dev)).cpu() for i in range(0, len(X_R), 256)]).double()[held]
        qRh = qR[held]
        # CORAL 목적값 분해(학습 손실과 같은 계산)
        p, r = resid(fP, qP), resid(fR, qRh); sc = torch.cat([p, r]).std(0).clamp_min(0.1); ps, rs = p / sc, r / sc
        mean_term = float((ps.mean(0) - rs.mean(0)).square().mean())
        cp = (ps - ps.mean(0)).T @ (ps - ps.mean(0)) / (len(ps) - 1); cr = (rs - rs.mean(0)).T @ (rs - rs.mean(0)) / (len(rs) - 1); cov_term = float((cp - cr).square().mean())
        # 공동 잔차: 원 단위 평균 차 · 합동 SD · 표준화 평균 차
        f = torch.cat([fP, fR]); q = torch.cat([qP, qRh]); e = resid(f, q).numpy(); ep, er = e[:len(fP)], e[len(fP):]
        raw_md = float(np.mean(np.abs(ep.mean(0) - er.mean(0)))); psd = float(np.mean(e.std(0))); std_md = float(np.mean(np.abs(ep.mean(0) - er.mean(0)) / e.std(0).clip(1e-8)))
        # 원 특징 크기(조성 보정 전): 도메인별 총분산
        tvP, tvR = float(fP.var(0).sum()), float(fR.var(0).sum())
        # Q2: 도메인별 잔차(평균 제거)와 공동 잔차의 분류 AUROC, 대조 = 유사벌크 무작위 반반
        yy = np.r_[np.zeros(len(ep)), np.ones(len(er))]
        auc_pooled = lr_auc(e, yy); auc_perdomain = lr_auc(np.r_[p.numpy(), r.numpy()], yy)
        # 누출 없는 판: fold 마다 학습 fold 로 도메인별 회귀를 적합해 시험 fold 잔차를 만든 뒤 LR·RF (평균 제거 후 남는 분리 = 공분산·고차 차이)
        Fa, Qa = torch.cat([fP, fR]).numpy(), torch.cat([qP, qRh]).numpy(); Da = np.c_[np.ones(len(Qa)), Qa[:, :2]]; sl, sr_ = np.zeros(len(yy)), np.zeros(len(yy))
        for tr, te in CV.split(Fa, yy):
            Ea = np.zeros_like(Fa)
            for d_ in (0, 1):
                m = tr[yy[tr] == d_]; B = np.linalg.lstsq(Da[m], Fa[m], rcond=None)[0]; idx = np.where(yy == d_)[0]; Ea[idx] = Fa[idx] - Da[idx] @ B
            sl[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Ea[tr], yy[tr]).predict_proba(Ea[te])[:, 1]
            sr_[te] = RandomForestClassifier(500, random_state=0, n_jobs=8).fit(Ea[tr], yy[tr]).predict_proba(Ea[te])[:, 1]
        auc_pd_lr, auc_pd_rf = float(roc_auc_score(yy, sl)), float(roc_auc_score(yy, sr_))
        half = np.random.default_rng(ch).permutation(len(ep)) < len(ep) // 2; auc_ctrl = lr_auc(ep, half.astype(float))
        # 평균 방향 하나만 쓰는 분류(학습 fold 의 평균 차 방향으로 사영) — 평균 이동이 분리를 설명하는지
        s = np.zeros(len(yy))
        for tr, te in CV.split(e, yy):
            w = e[tr][yy[tr] == 1].mean(0) - e[tr][yy[tr] == 0].mean(0); s[te] = e[te] @ w
        auc_meandir = float(roc_auc_score(yy, s))
        res["arms"][arm][ch] = {"coral_objective": mean_term + cov_term, "coral_mean_term": mean_term, "coral_cov_term": cov_term,
                                "pooled_raw_mean_diff": raw_md, "pooled_sd_mean": psd, "pooled_std_mean_diff": std_md, "total_var_P": tvP, "total_var_R": tvR,
                                "auc_pooled": auc_pooled, "auc_per_domain_resid": auc_perdomain, "auc_per_domain_foldwise_lr": auc_pd_lr, "auc_per_domain_foldwise_rf": auc_pd_rf, "auc_mean_direction_only": auc_meandir, "auc_control_pseudo_halves": auc_ctrl}
        print(f"[{ORG} {arm} c{ch}] obj mean {mean_term:.2e} cov {cov_term:.4f} · pooled raw Δμ {raw_md:.4f} SD {psd:.4f} std Δμ {std_md:.3f} · var P/R {tvP:.2f}/{tvR:.2f} · "
              f"AUC pooled {auc_pooled:.3f} per-domain {auc_perdomain:.3f} fold-wise LR {auc_pd_lr:.3f} RF {auc_pd_rf:.3f} mean-dir {auc_meandir:.3f} ctrl {auc_ctrl:.3f}", flush=True)
        if ch == 1:
            # 유사벌크로만 맞춘 PCA(공동 잔차), 실제 벌크 사영, 유사벌크 PC SD 단위
            mu = ep.mean(0); U, S_, Vt = np.linalg.svd(ep - mu, full_matrices=False); sdp = (S_ / np.sqrt(len(ep) - 1))[:2]
            fig[f"{arm}_P"] = (ep - mu) @ Vt[:2].T / sdp; fig[f"{arm}_R"] = (er - mu) @ Vt[:2].T / sdp
json.dump(res, open(OUT / f"{ORG}_diag.json", "w"), indent=1); np.savez_compressed(OUT / f"{ORG}_diag_fig.npz", **fig)
F2 = F / "results_json/mech/domain_gap"; json.dump(res, open(F2 / f"{ORG}_diag.json", "w"), indent=1)
