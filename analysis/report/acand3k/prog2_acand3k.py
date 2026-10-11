"""사용자 검토 4번(2026-09-30) — A_cand3k 예후 표 재구성: C 정의를 섞지 않는다.
두 평가량(항상 표기): (i) 체인 평균 C = 체인·시드별 C 의 평균(v7 규약; 비교 모델은 시드별 평균), (ii) 앙상블 C = 체인·시드 위험을 코호트마다 z-평균한 위험의 C.
주표 = 가용 체인 전부(유방·간 c1–c12, 췌장 c1–c6). c1–c6 · c7–c12 따로.
환자 짝 bootstrap(코호트별 환자 복원추출, 코호트의 모든 종점에 같은 표본, B=2000, rng 0 — boot_e2pp_vs_baselines.py 와 같은 표본열):
  - 앙상블 C 기준 A_cand3k(전체 체인) − 각 모델
  - 체인 평균 C 기준 A_cand3k(전체 체인) − 각 모델(연속성) · 배포 − 각 모델(항등 점검: 원 BOOT_E2PP_vs_BL 재현)
체인 구성 점검 = 체인 평균 C 의 c1–c3 / c1–c6 / c1–c12(/c7–c12). 앙상블 크기 점검 = 같은 중첩 체인의 앙상블 C 3 / 6 / 12, 차이의 bootstrap.
env ORG(breast|liver|pancreas) · B. 출력 results/report_acand3k/prog2/PROG2_{ORG}.json"""
import os, json, numpy as np, torch
from lifelines.utils import concordance_index as _ci
ORG = os.environ["ORG"]; B = int(os.environ.get("B", "2000")); R = "results/report/"; OUTD = "results/report_acand3k/prog2"; os.makedirs(OUTD, exist_ok=True)
ends = json.load(open(R + f"MSUB_{ORG}.json"))["endpoints"]; M = np.load(R + f"MSUB_{ORG}.npz")
L = np.load(f"results/cppch/baseline_compare/BL_breast_ALL26_860.npz" if ORG == "breast" else f"results/cppch/baseline_compare/BL_{ORG}.npz")
NCH = sorted(int(p.split("/c")[-1].split("/")[0]) for p in __import__("glob").glob(f"results/hvg_design/run1a/{ORG}/A_cand3k/c*/pred_bulk.pt"))
NCH = [c for c in NCH if c <= int(os.environ.get("MAXCHAIN", "6"))]   # 2026-10-11: 모든 모델 6체인 동일 조건(PLAN_ALPHA_FREE). MAXCHAIN=12 = 이전 판
risk = {}
def chain_risks(chains):
    out = {}
    for c in chains:
        p = torch.load(f"results/hvg_design/run1a/{ORG}/A_cand3k/c{c}/pred_bulk.pt", map_location="cpu", weights_only=False)
        rc = {k: np.asarray(v, float) for k, v in p["risk_cohorts"].items()}; rc["TCGA-heldout"] = np.asarray(p["risk_all"], float)[np.asarray(p["heldout_idx"], int)]
        for k, v in rc.items(): out.setdefault(k, []).append(v)
    return out
risk["acand3k"] = chain_risks(NCH)
for m in ("deployed", "directbulk", "pi_only"):
    for k in M.files:
        if k.startswith(m + "__"): risk.setdefault(m, {})[k.split("__")[1]] = list(M[k])
bl = ["scSurv", "SIDISH_ens" if ORG == "breast" else "SIDISH", "Scissor", "DEGAS"] + (["SIDISH_NNLS"] if ORG != "breast" else [])
for m in bl:
    for k in L.files:
        mm, c, s = k.split("__")
        if mm == m: risk.setdefault(m, {}).setdefault(c, []).append(L[k])
EPS = [k for k in ends if all(k.split("|")[0] in risk[m] for m in risk)]
EXT_SETS = {"ext": [k for k in EPS if not k.startswith("TCGA")]}
if ORG == "liver": EXT_SETS["ext_primary4"] = [k for k in EXT_SETS["ext"] if k not in ("GSE54236|OS*", "GSE10143|OS")]
TK = "TCGA-heldout|OS"
z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
# 평가 단위: (모델명, 체인 부분집합) — 체인 평균·앙상블
SUB = {"acand3k": list(range(len(NCH)))}
nA = len(NCH)
for lab, idx in (("c1-3", [0, 1, 2]), ("c1-6", list(range(6))), ("c7-12", list(range(6, 12)))):
    if max(idx) < nA: SUB[f"acand3k_{lab}"] = idx
UNITS = {"acand3k": ("acand3k", SUB["acand3k"])}
for k, v in SUB.items(): UNITS[k] = ("acand3k", v)
for m in risk:
    if m != "acand3k": UNITS[m] = (m, list(range(len(next(iter(risk[m].values()))))))
ENS = {u: {c: np.mean([z(np.asarray(risk[m][c][i], float)) for i in idx], 0) for c in risk[m]} for u, (m, idx) in UNITS.items()}
def c1(r, key, pos):
    v = ends[key]; ix = np.asarray(v["indices"], int)[pos]; t = np.asarray(v["time"], float)[pos]; e = np.asarray(v["event"], float)[pos]; ri = r[ix]
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(ri) & (t > 0); return float(_ci(t[ok], -ri[ok], e[ok]))
def table(pos, chain_units, ens_units):
    out = {"chain": {}, "ens": {}}
    for u in chain_units:
        m, idx = UNITS[u]; out["chain"][u] = {k: float(np.mean([c1(np.asarray(risk[m][ends[k]["cohort"]][i], float), k, pos[k]) for i in idx])) for k in EPS}
    for u in ens_units: out["ens"][u] = {k: c1(ENS[u][ends[k]["cohort"]], k, pos[k]) for k in EPS}
    return out
full = {k: np.arange(len(ends[k]["indices"])) for k in EPS}
PT = table(full, list(UNITS), list(UNITS))
def summ(d):
    s = {nm: float(np.mean([d[k] for k in ks])) for nm, ks in EXT_SETS.items()}; s["held"] = d[TK]; return s
res = {"organ": ORG, "chains": NCH, "B": B, "ext_sets": EXT_SETS, "definitions": {"chain": "체인·시드별 C 평균", "ens": "코호트별 z-평균 앙상블 위험의 C"},
       "point": {q: {u: {**PT[q][u], **summ(PT[q][u])} for u in PT[q]} for q in PT}, "n_members": {u: len(idx) for u, (m, idx) in UNITS.items()}}
# bootstrap (원 스크립트와 같은 표본열)
rng = np.random.default_rng(0); cohorts = sorted({ends[k]["cohort"] for k in EPS}); pos_of = {k: {int(i): j for j, i in enumerate(ends[k]["indices"])} for k in EPS}
CHU = ["acand3k", "deployed"] + [m for m in risk if m not in ("acand3k", "deployed")]
ENU = list(UNITS)
SAMPS = [{c: rng.choice(len(risk["acand3k"][c][0]), len(risk["acand3k"][c][0]), replace=True) for c in cohorts} for b in range(B)]   # 원 스크립트와 같은 표본열
NSH = int(os.environ.get("NSHARD", "1")); SH = int(os.environ.get("SHARD", "0"))
def run_reps(ids):
    out = []
    for b in ids:
        samp = SAMPS[b]; pe = {k: np.array([pos_of[k][i] for i in samp[ends[k]["cohort"]] if i in pos_of[k]]) for k in EPS}
        T = table(pe, CHU, ENU); out.append({q: {u: summ(T[q][u]) for u in T[q]} for q in T})
        if b % 100 == 0: print(ORG, b, flush=True)
    return out
if NSH > 1 and SH >= 0:
    json.dump(run_reps(range(SH, B, NSH)), open(f"{OUTD}/_shard_{ORG}_{SH}.json", "w")); raise SystemExit
if NSH > 1:   # SHARD=-1 → 원래 순서로 합치기
    parts = [json.load(open(f"{OUTD}/_shard_{ORG}_{i}.json")) for i in range(NSH)]; BS = [None] * B
    for i, pr in enumerate(parts):
        for j, r in zip(range(i, B, NSH), pr): BS[j] = r
else: BS = run_reps(range(B))
def ci(q, a, bb, met):
    d = np.array([r[q][a][met] - r[q][bb][met] for r in BS]); pt = res["point"][q][a][met] - res["point"][q][bb][met]
    lo, hi = np.percentile(d, [2.5, 97.5]); return {"diff": float(pt), "lo": float(lo), "hi": float(hi)}
mets = list(EXT_SETS) + ["held"]
res["compare"] = {"ens": {f"acand3k - {m}": {mt: ci("ens", "acand3k", m, mt) for mt in mets} for m in UNITS if not m.startswith("acand3k")},
                  "chain": {f"acand3k - {m}": {mt: ci("chain", "acand3k", m, mt) for mt in mets} for m in CHU if m != "acand3k"},
                  "chain_identity_deployed": {f"deployed - {m}": {mt: ci("chain", "deployed", m, mt) for mt in mets} for m in CHU if m not in ("acand3k", "deployed")}}
res["ensemble_size"] = {f"{a} - {bb}": {mt: ci("ens", a, bb, mt) for mt in mets} for a, bb in (("acand3k_c1-6", "acand3k_c1-3"), ("acand3k", "acand3k_c1-6"), ("acand3k", "acand3k_c1-3")) if a in UNITS and bb in UNITS and a != bb}
# 항등: 원 BOOT(배포 기준)
orig = json.load(open(f"results/cppch/baseline_compare/BOOT_E2PP_vs_BL_{ORG}.json")); dmax = 0.0
for m, v in orig["compare"].items():
    mm = m.split(" - ")[1]; k = f"deployed - {mm}"
    if k in res["compare"]["chain_identity_deployed"]:
        for a, bb in (("C_ext", "ext"), ("TCGA", "held")):
            for f in ("diff", "lo", "hi"): dmax = max(dmax, abs(v[a][f] - res["compare"]["chain_identity_deployed"][k][bb][f]))
res["identity_max_abs_vs_BOOT_E2PP_vs_BL"] = dmax
json.dump(res, open(f"{OUTD}/PROG2_{ORG}.json", "w"), ensure_ascii=False, indent=1); print(ORG, "identity", dmax)
