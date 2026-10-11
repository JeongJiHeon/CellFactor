#!/usr/bin/env python3
"""Fig 2a–c 유방 UMAP 고위험 상위 10% 재채점 (PLAN_ALPHA_FREE). UMAP 좌표·발현 PC 는 그대로 두고 점수·선정만 다시 계산한다.
원 생성기(code/scripts/umap_acand3k_cells.py)는 .pyc 만 남아 있어, MANIFEST 의 정의를 그대로 다시 구현했다:
  세포 점수 = mean_c1..c6( α_c[type] + W_c[type]·z_cal ),  z_cal = clamp(z[:, active_dims] / s, ±12)(유방),
  선정 = 점수 내림차순 상위 ceil(N·0.10), 동점은 원 행 번호 오름차순.
항등 점검: 백업(α=0) W 로 다시 계산한 체인 점수가 기존 cells.csv.gz 의 risk_c* 와 같아야 한다.
env ORGAN(기본 breast) · BACKUP(기본 results/hvg_design/alpha0_backup/breast/A_cand3k)"""
import os, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[4]; ORG = os.environ.get("ORGAN", "breast"); assert ORG == "breast"
D = ROOT / "final_results/01_umap" / ORG; RUN = ROOT / "results/hvg_design/run1a" / ORG / "A_cand3k"
BK = ROOT / os.environ.get("BACKUP", "results/hvg_design/alpha0_backup/breast/A_cand3k")
C = pd.read_csv(D / "cells.csv.gz"); N = len(C)


def chain_scores(pb_path, c):
    pb = torch.load(pb_path, map_location="cpu", weights_only=False)
    zp = torch.load(ROOT / f"weights/hvg_run1a/{ORG}/A_cand3k/c{c}/stage1/G176_ZPRIME.pt", map_location="cpu", weights_only=False)
    z = zp["z"].float()[:, torch.as_tensor(pb["active_dims"])] / torch.as_tensor(pb["s"]).float()[None]
    z = z.clamp(-12, 12)[torch.as_tensor(C.cell_row_index.values)]
    types = list(pb["cell_types"]); t = torch.as_tensor([types.index(x) for x in C.cell_type])
    W, A = pb["W"].float(), pb["alpha"].float()
    return (A[t] + (z * W[t]).sum(1)).double().numpy()


ident = max(float(np.abs(chain_scores(BK / f"c{c}" / "pred_bulk.pt", c) - C[f"risk_c{c}"].values).max()) for c in range(1, 7))
assert ident < 1e-4, ident
old_sel = C.high_risk_top10.values.astype(bool).copy()
for c in range(1, 7): C[f"risk_c{c}"] = chain_scores(RUN / f"c{c}" / "pred_bulk.pt", c)
C["risk"] = C[[f"risk_c{c}" for c in range(1, 7)]].mean(1)
k = math.ceil(N * 0.10); order = np.lexsort((C.cell_row_index.values, -C.risk.values)); sel = np.zeros(N, bool); sel[order[:k]] = True
C["high_risk_top10"] = sel
C.to_csv(D / "cells.csv.gz", index=False, compression="gzip")
T = C.groupby("cell_type").agg(total_cells=("cell_type", "size"), high_risk_cells=("high_risk_top10", "sum")).reset_index()
T["percent_of_high_risk_cells"] = 100 * T.high_risk_cells / k; T["percent_high_within_type"] = 100 * T.high_risk_cells / T.total_cells
T.sort_values("percent_of_high_risk_cells", ascending=False).to_csv(D / "cell_type_counts.csv", index=False)
M = json.load(open(D / "MANIFEST.json")); bscore = float(C.risk.values[order[k - 1]])
M.update({"boundary_score": bscore, "boundary_tie_count": int((C.risk.values == bscore).sum()), "high_risk_cells": int(k),
          "rescored": {"date": "2026-10-11", "reason": "breast CSRF alpha free (PLAN_ALPHA_FREE)", "identity_vs_previous_alpha0_maxabs": ident,
                       "selection_overlap_with_previous": int((sel & old_sel).sum()), "script": "analysis/report/paper/v2/umap_rescore.py"}})
json.dump(M, open(D / "MANIFEST.json", "w"), ensure_ascii=False, indent=2)
print(f"[umap {ORG}] 항등 {ident:.1e} · 선정 {k} · 이전 선정과 겹침 {(sel & old_sel).sum()} · 경계 {bscore:.4f}")
print(T.sort_values("percent_of_high_risk_cells", ascending=False).to_string(index=False))
