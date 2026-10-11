#!/usr/bin/env python3
"""C-index 환자 bootstrap B=2000 — 모델별 점추정·bootstrap 평균·SE·95% percentile CI, CellFactor − 모델 짝 차이.
위험·종점·앙상블 정의는 prog2_acand3k.py(코호트별 z-평균 앙상블 위험의 C)와 같다: 그 파일의 적재부를 그대로 실행한다.
재표집: 코호트별 환자 복원추출, 코호트의 모든 종점에 같은 표본, 모든 모델에 같은 표본(rng seed 0).
외부 평균 = 확정 외부 종점(간은 GSE14520·GSE76427 OS/RFS 4종점) 평균. held-out = TCGA OS.
env ORG · B. 출력 final_results/02_cindex/bootstrap_ci_B2000/{ORG}_BOOT.json"""
import os, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; os.chdir(ROOT)
ORG = os.environ["ORG"]; B = int(os.environ.get("B", "2000"))
src = open("analysis/report/acand3k/prog2_acand3k.py").read()
pre = src[:src.index("full = {k: np.arange")]
g = {"__name__": "boot", "__file__": "prog2_acand3k.py"}; exec(compile(pre.replace('os.makedirs(OUTD, exist_ok=True)', 'pass'), "prog2(pre)", "exec"), g)
ends, EPS, EXT_SETS, ENS, c1, TK = g["ends"], g["EPS"], g["EXT_SETS"], g["ENS"], g["c1"], g["TK"]
EXT = EXT_SETS["ext_primary4" if ORG == "liver" else "ext"]
SID = "SIDISH_ens" if ORG == "breast" else "SIDISH"
UNITS = [("CellFactor", "acand3k"), ("CellFactor (c1–c6)", "acand3k_c1-6"), ("SIDISH", SID), ("scSurv", "scSurv"), ("Scissor", "Scissor"), ("DEGAS", "DEGAS"),
         ("Bulk Cox", "directbulk"), ("Composition-only Cox", "pi_only")]
# 단순화 모델 3종(PLAN_ABLATION_BASELINES, CellFactor 구성요소 재구현) — 있으면 추가. 재표집 열(rng)은 모델 수와 무관하다.
_ab = ROOT / f"results/mech/ablation_risks/{ORG}.npz"
if _ab.exists():
    Z_ = np.load(_ab); zf = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    for lab, key in (("Program-score Cox", "programscore"), ("Composition + program-score Cox", "comp_programscore"), ("State-only model", "stateonly"), ("Proliferation-score Cox (post hoc)", "prolifscore")):
        ENS[key] = {c: np.mean([zf(np.asarray(r, float)) for r in Z_[f"{key}__{c}"]], 0) for c in next(iter(ENS.values())) if f"{key}__{c}" in Z_.files}
        UNITS.append((lab, key))
UNITS = [(a, u) for a, u in UNITS if u in ENS]
cohorts = sorted({ends[k]["cohort"] for k in EPS})
ncoh = {c: len(next(iter(ENS.values()))[c]) for c in cohorts}


def evaluate(pos):
    out = {}
    for lab, u in UNITS:
        d = {k: c1(ENS[u][ends[k]["cohort"]], k, pos[k]) for k in EPS}
        d["external_mean"] = float(np.mean([d[k] for k in EXT])); d["heldout"] = d[TK]; out[lab] = d
    return out


def positions(draw):
    """코호트 환자 재표집(draw[cohort] = 환자 인덱스) → 종점별 위치 배열."""
    pos = {}
    for k in EPS:
        ix = np.asarray(ends[k]["indices"], int); c = ends[k]["cohort"]
        if draw is None: pos[k] = np.arange(len(ix)); continue
        where = {}
        for j, p in enumerate(ix): where.setdefault(int(p), []).append(j)
        pos[k] = np.asarray([j for p in draw[c] for j in where.get(int(p), [])], int)
    return pos


point = evaluate(positions(None))
rng = np.random.default_rng(0); BS = []
for b in range(B):
    draw = {c: rng.integers(0, ncoh[c], ncoh[c]) for c in cohorts}
    BS.append(evaluate(positions(draw)))
    if b % 200 == 0: print(f"[{ORG}] {b}/{B}", flush=True)
keys = EPS + ["external_mean", "heldout"]
res = {"organ": ORG, "B": B, "external_endpoints": EXT, "heldout_endpoint": TK, "models": {}, "paired_vs_CellFactor": {}}
for lab, _ in UNITS:
    res["models"][lab] = {}
    for k in keys:
        v = np.array([bs[lab][k] for bs in BS])
        res["models"][lab][k] = {"point": point[lab][k], "boot_mean": float(v.mean()), "se": float(v.std(ddof=1)), "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])]}
    if lab.startswith("CellFactor"): continue
    res["paired_vs_CellFactor"][lab] = {}
    for k in ("external_mean", "heldout"):
        d = np.array([bs["CellFactor"][k] - bs[lab][k] for bs in BS]); lo, hi = np.percentile(d, [2.5, 97.5])
        res["paired_vs_CellFactor"][lab][k] = {"diff": point["CellFactor"][k] - point[lab][k], "ci95": [float(lo), float(hi)],
                                               "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                                               "verdict": "CellFactor higher" if lo > 0 else ("baseline higher" if hi < 0 else "no difference detected")}
# 2026-10-10 추가: 체인 c1–c6 앙상블 기준 짝 차이(CellFactor (c1–c6) − 모델), 모든 종점 + external_mean + heldout. 같은 재표집 열.
REF6 = "CellFactor (c1–c6)"; res["paired_vs_CellFactor_c1_6"] = {}
if REF6 in point:
    for lab, _ in UNITS:
        if lab.startswith("CellFactor"): continue
        res["paired_vs_CellFactor_c1_6"][lab] = {}
        for k in keys:
            d = np.array([bs[REF6][k] - bs[lab][k] for bs in BS]); lo, hi = np.percentile(d, [2.5, 97.5])
            res["paired_vs_CellFactor_c1_6"][lab][k] = {"diff": point[REF6][k] - point[lab][k], "ci95": [float(lo), float(hi)],
                                                        "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
OUT = ROOT / "final_results/02_cindex/bootstrap_ci_B2000"; OUT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT / f"{ORG}_BOOT.json", "w"), indent=1)
for lab, _ in UNITS:
    m = res["models"][lab]
    print(f"{ORG} {lab:22s} held {m['heldout']['point']:.3f} [{m['heldout']['ci95'][0]:.3f}, {m['heldout']['ci95'][1]:.3f}]  ext {m['external_mean']['point']:.3f} [{m['external_mean']['ci95'][0]:.3f}, {m['external_mean']['ci95'][1]:.3f}]")
