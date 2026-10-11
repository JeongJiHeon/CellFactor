"""KM(위험 상위 20% vs 전체, 로그순위) — code/report/km_v6.py 를 그대로 실행하되 CP-PCH 위험만 A_cand3k(c1–c6 체인 위험의 코호트 내 z 평균)로 바꾼다.
env IDCHECK=1 이면 원 경로 그대로 실행해 KM_5MODELS_LOGRANK_v6.json 을 재현하는지 본다(출력은 results/report_acand3k/_idcheck_*).
출력 results/report_acand3k/{ORG}_ACAND3K_RISKS.json · fig_km_5models_acand3k.png · KM_5MODELS_LOGRANK_acand3k.json(+ 배포 E2″·벌크 직접·조성만 p 참조)."""
import os, json, numpy as np, torch
OUT = "results/report_acand3k/"; os.makedirs(OUT, exist_ok=True); ID = os.environ.get("IDCHECK") == "1"
def build(org, fname):
    ch = []
    for c in range(1, 7):
        p = torch.load(f"results/hvg_design/run1a/{org}/A_cand3k/c{c}/pred_bulk.pt", map_location="cpu", weights_only=False)
        rc = {k: np.asarray(v, float) for k, v in p["risk_cohorts"].items()}; rc["TCGA-heldout"] = np.asarray(p["risk_all"], float)[np.asarray(p["heldout_idx"], int)]; ch.append((rc, p))
    z = lambda a: (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12); p0 = ch[0][1]; ho = np.asarray(p0["heldout_idx"], int); out = {}
    for co in ch[0][0]:
        r = np.mean([z(rc[co]) for rc, _ in ch], 0); eps = {}
        if co == "TCGA-heldout":
            eps["OS"] = {"time": np.asarray(p0["time_all"], float)[ho].tolist(), "event": np.asarray(p0["event_all"], float)[ho].tolist(), "indices": list(range(len(ho)))}
        else:
            for ep, d in p0["endpoints"][co].items(): eps[ep] = {k: np.asarray(d[k]).tolist() for k in ("time", "event", "indices")}
        out[co] = {"risk": r.tolist(), "endpoints": eps}
    json.dump(out, open(OUT + fname, "w")); return OUT + fname
src = open("analysis/report/km_v6.py").read()
if not ID:
    bp = build("breast", "BREAST_ACAND3K_RISKS.json"); lp = build("liver", "LIVER_ACAND3K_RISKS.json"); pp = build("pancreas", "PANC_ACAND3K_RISKS.json")
    os.environ["KM_BR_CP"] = bp
    src = src.replace('J("results/report/LIVER_V5_RISKS.json")', f'J("{lp}")').replace('J("results/report/PANC_V5_RISKS.json")', f'J("{pp}")')
    src = src.replace("fig.suptitle(\"High-risk (top 20%) vs All — 5-model comparison", "fig.suptitle(\"High-risk (top 20%) vs All — CP-PCH = A_cand3k (6-chain ensemble); 5-model comparison")
    src = src.replace('"results/report/fig_km_5models_v6.png"', f'"{OUT}fig_km_5models_acand3k.png"').replace('"results/report/KM_5MODELS_LOGRANK_v6.json"', f'"{OUT}KM_5MODELS_LOGRANK_acand3k.json"')
else:
    src = src.replace('"results/report/fig_km_5models_v6.png"', f'"{OUT}_idcheck_km.png"').replace('"results/report/KM_5MODELS_LOGRANK_v6.json"', f'"{OUT}_idcheck_KM.json"')
assert "results/report/fig_km_5models_v6.png" not in src and "results/report/KM_5MODELS_LOGRANK_v6.json" not in src
exec(compile(src, "km_v6(adapted)", "exec"), {"__name__": "__main__"})
if not ID:
    K = json.load(open(OUT + "KM_5MODELS_LOGRANK_acand3k.json")); V6 = json.load(open("results/report/KM_5MODELS_LOGRANK_v6.json")); X = json.load(open("results/report/KM_EXTRA_v7.json"))
    K["_reference"] = {"note": "같은 규칙·같은 코호트. 배포 E2″ = KM_5MODELS_LOGRANK_v6 의 CP-PCH, 벌크 직접·조성만 = KM_EXTRA_v7(모델 무관이라 재계산 안 함)",
                       "deployed_E2pp": {k.split("|")[0]: v for k, v in V6.items() if k.endswith("|CP-PCH")}, "extra": X["p"]}
    json.dump(K, open(OUT + "KM_5MODELS_LOGRANK_acand3k.json", "w"), ensure_ascii=False, indent=1)
