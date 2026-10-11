"""A_cand3k(전체 체인 앙상블, prog2 판) KM 곡선에 필요한 값을 코호트별로 추출.
입력: results_json/report_acand3k/prog2/{BREAST,LIVER,PANC}_ACAND3K_RISKS.json (risk + endpoints.time/event/indices).
규칙은 code/report/km_v6.py 의 km()/logrank() 그대로 — 위험 상위 20% vs 전체, 로그순위(Mantel-Haenszel).
각 코호트의 주 종점(OS 또는 OS*)만 사용(간 GSE54236 은 OS* 가 유일 종점).
출력 results_json/report_acand3k/km_data/
  {COHORT}_patients.csv   — time_years,event,risk,high_risk(top20%) · 환자별, KM을 직접 그릴 수 있는 원값
  {COHORT}_curve.csv      — group,time_years,survival · km() 으로 만든 계단형 포인트(두 그룹: all/high20)
  SUMMARY.json            — 코호트별 n/events/n_high20/logrank_p
"""
import os, json
import numpy as np

SRC = "results_json/report_acand3k/prog2/"
OUT = "results_json/report_acand3k/km_data/"
os.makedirs(OUT, exist_ok=True)

FILES = {
    "Breast": SRC + "BREAST_ACAND3K_RISKS.json",
    "Liver": SRC + "LIVER_ACAND3K_RISKS.json",
    "Pancreas": SRC + "PANC_ACAND3K_RISKS.json",
}
# 코호트별 주 종점(여러 종점이 있을 때 report/PROG2_ACAND3K.md 의 "주 외부 정의"·km_v6.py 규칙과 동일하게 OS 우선, 없으면 유일 종점)
PRIMARY_EP = {"GSE54236": "OS*"}


def km(t, e):
    """results-identical to code/report/km_v6.py:km — 계단형 KM, (xs_days, ys)."""
    o = np.argsort(t)
    t, e = t[o], e[o]
    n = len(t)
    S = 1.0
    xs = [0.0]
    ys = [1.0]
    for i, (ti, ei) in enumerate(zip(t, e)):
        if ei > 0:
            S *= 1 - 1 / (n - i)
            xs.append(ti)
            ys.append(S)
    xs.append(t[-1] if len(t) else 0)
    ys.append(S)
    return np.array(xs), np.array(ys)


def logrank(t, e, g):
    """results-identical to code/report/km_v6.py:logrank — Mantel-Haenszel 로그순위 2-군 p값."""
    ts = np.unique(t[e > 0])
    O1 = E1 = V = 0.0
    for x in ts:
        at = t >= x
        n = at.sum()
        n1 = (at & (g == 1)).sum()
        d = ((t == x) & (e > 0)).sum()
        d1 = ((t == x) & (e > 0) & (g == 1)).sum()
        if n < 2:
            continue
        O1 += d1
        E1 += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    from math import erfc, sqrt
    return float(erfc(abs((O1 - E1) / np.sqrt(max(V, 1e-12))) / sqrt(2))) if V > 0 else np.nan


summary = {}
for organ, path in FILES.items():
    R = json.load(open(path))
    for coh, c in R.items():
        ep_name = PRIMARY_EP.get(coh) or ("OS" if "OS" in c["endpoints"] else next(iter(c["endpoints"])))
        v = c["endpoints"][ep_name]
        r = np.asarray(c["risk"], float)
        t = np.asarray(v["time"], float)
        e = np.asarray(v["event"], float)
        idx = np.asarray(v.get("indices", list(range(len(t)))), int)
        ri = r[idx] if len(idx) == len(t) else r
        m = np.isfinite(ri) & np.isfinite(t) & np.isfinite(e) & (t > 0)
        ri, t, e = ri[m], t[m], e[m]
        if len(t) < 20:
            continue
        hi = (ri >= np.quantile(ri, 0.8)).astype(int)
        p = logrank(t, e, hi)
        tag = f"{organ}_{coh}_{ep_name}".replace(" ", "").replace("*", "star")

        # 환자별 원값(직접 KM 을 그릴 수 있는 최소 단위)
        with open(OUT + f"{tag}_patients.csv", "w") as f:
            f.write("time_years,event,risk,high_risk_top20\n")
            for ti, ei, ri_, hi_ in zip(t, e, ri, hi):
                f.write(f"{ti/365.25:.6f},{int(ei)},{ri_:.6f},{hi_}\n")

        # km() 으로 만든 계단형 포인트 (두 그룹)
        with open(OUT + f"{tag}_curve.csv", "w") as f:
            f.write("group,time_years,survival\n")
            xs, ys = km(t, e)
            for x, y in zip(xs, ys):
                f.write(f"all,{x/365.25:.6f},{y:.6f}\n")
            xs, ys = km(t[hi == 1], e[hi == 1])
            for x, y in zip(xs, ys):
                f.write(f"high20,{x/365.25:.6f},{y:.6f}\n")

        summary[tag] = {
            "organ": organ, "cohort": coh, "endpoint": ep_name,
            "n": int(len(t)), "events": int(e.sum()), "n_high20": int(hi.sum()),
            "logrank_p_top20_vs_all": p,
        }
        print(f"{tag}: n={len(t)} events={int(e.sum())} n_high20={int(hi.sum())} p={p:.3e}")

json.dump(summary, open(OUT + "SUMMARY.json", "w"), ensure_ascii=False, indent=1)
print("[saved]", OUT + "SUMMARY.json")
