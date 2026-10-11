#!/usr/bin/env python3
"""PLAN_PROGRAM_AXIS E1 — 모델과 독립인 Leiden 군집 표지(CPU, 학습 없음).
시드마다 학습 도너 초점 세포(npz tf · tr_sid)에서
  Δ_k(c) = (군집 c 세포의 K0 프로그램 k 점수 평균 − 나머지 평균) / 풀린 SD   [점수 = log1p CP10k 세트 평균 — Stage 1 앵커와 같은 정의]
  마커 = 군집 c vs 나머지 Cohen's d 상위 50 (군집 내 검출률 ≥10% 유전자)
를 만든다. 모델 출력(z′·W)을 쓰지 않는다.
유방(2026-10-06 추가): GSE176078 학습 도너 T세포(g161_common.setup 과 같은 선택 tf), 표적 = data/spatial/leiden_sweep/g161_leiden_s{k}.npz 의 tr_sid.
usage: ORG=liver|pancreas|breast SEEDS="0 1 ... 9" python3 cluster_program_signature.py
출력 results/report_acand3k/cellaxis/program_axis/signature_{ORG}_s{k}.json"""
import os, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.join(os.environ.get("CELLFACTOR_PROJECT", "."), "code/eval/hvg_run1a")); import common as C

F = C.FINAL; ORG = os.environ["ORG"]; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]
OUT = F / "results/report_acand3k/cellaxis/program_axis"; OUT.mkdir(parents=True, exist_ok=True)
D = C.load_stage1_data(ORG); counts = D["sc_counts"]
if counts.dtype == torch.uint16: counts = counts.view(torch.int16)
genes = [str(g) for g in D["gene_names"]]; gi = {g: i for i, g in enumerate(genes)}
progs = json.load(open(C.ORG[ORG]["k0"]))
pidx = {k: [gi[g] for g in v if g in gi] for k, v in progs.items()}; pidx = {k: v for k, v in pidx.items() if len(v) >= 3}

for s in SEEDS:
    js = OUT / f"signature_{ORG}_s{s}.json"
    if js.exists(): print("[skip]", js); continue
    if ORG == "breast":   # npz 에 tf 가 없음 → setup() 과 같은 규칙(학습 도너 ∩ T-cells, 인덱스 오름차순)
        H = np.load(F / f"data/spatial/leiden_sweep/g161_leiden_s{s}.npz"); sid = H["tr_sid"]
        ct_, dn_ = np.asarray(D["sc_cell_type"]), np.asarray(D["sc_donor"])
        tf = np.where(np.isin(dn_, np.asarray(D["donor_split"]["train"])) & (ct_ == list(D["cell_type_names"]).index("T-cells")))[0]
        assert len(tf) == len(sid), (len(tf), len(sid))
    else:
        H = np.load(F / f"data/spatial/leiden_sweep/leiden_{ORG}_s{s}.npz"); tf, sid = H["tf"], H["tr_sid"]
    X = counts[torch.as_tensor(tf)].float(); lib = X.sum(1, keepdim=True).clamp_min(1)
    L = torch.log1p(X / lib * 1e4)                                     # 초점 학습 세포 × 전체 패널
    P = torch.stack([L[:, v].mean(1) for v in pidx.values()], 1).numpy(); pn = list(pidx)
    Ln = L.numpy(); det = (X > 0).numpy()
    rec = {"organ": ORG, "seed": s, "n_cells": int(len(tf)), "programs": pn, "clusters": {}}
    for c in np.unique(sid):
        m = sid == c; n1, n0 = int(m.sum()), int((~m).sum())
        if n1 < 20 or n0 < 20: continue
        sd = np.sqrt(((n1 - 1) * P[m].var(0, ddof=1) + (n0 - 1) * P[~m].var(0, ddof=1)) / (n1 + n0 - 2)) + 1e-9
        delta = (P[m].mean(0) - P[~m].mean(0)) / sd
        g_ok = det[m].mean(0) >= 0.10
        gsd = np.sqrt(((n1 - 1) * Ln[m].var(0, ddof=1) + (n0 - 1) * Ln[~m].var(0, ddof=1)) / (n1 + n0 - 2)) + 1e-9
        d = np.where(g_ok, (Ln[m].mean(0) - Ln[~m].mean(0)) / gsd, -np.inf)
        top = np.argsort(-d)[:50]
        rec["clusters"][str(int(c))] = {"n": n1, "delta": {k: float(v) for k, v in zip(pn, delta)},
                                        "markers": [genes[i] for i in top], "marker_d": [float(d[i]) for i in top]}
    json.dump(rec, open(js, "w"), ensure_ascii=False, indent=1)
    best = {c: max(v["delta"], key=v["delta"].get) for c, v in rec["clusters"].items()}
    print(f"[{ORG} s{s}] 군집 {len(rec['clusters'])} · 최대 Δ 프로그램 {best}", flush=True)
