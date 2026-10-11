#!/usr/bin/env python3
"""PLAN_SIMULATION 요약: results/mech/simulation/{SCEN}_r{r}.json → SUMMARY.json · SIMULATION.md (+ results_json/mech, final_results/07_mechanism 미러).
반복 10회 중앙값과 범위. 판정 문구는 계획서 기술 기준(유형 기여 Spearman 중앙값 ≥ 0.5 → 복원, 부호 ≥ 9/10 → 방향 복원)만 쓴다."""
import json, shutil
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; D = ROOT / "results/mech/simulation"
SCENS = ["A", "B", "C0", "C5", "C9", "D"]; METHODS = ["CellFactor", "oracle_comp", "oracle_state", "oracle_both"]
LABEL = {"A": "A independent", "B": "B comp.–state ρ 0.6", "C0": "C shared hypoxia ρ 0", "C5": "C shared hypoxia ρ 0.5", "C9": "C shared hypoxia ρ 0.9", "D": "D domain shift"}
R = {s: [json.load(open(p)) for p in sorted([q for q in D.glob(f"{s}_r*.json") if not q.stem.endswith("_a0")], key=lambda p: int(p.stem.split("_r")[1]))] for s in SCENS}
R = {s: v for s, v in R.items() if v}


def st(x):
    x = [v for v in x if v is not None]
    return {"median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x)), "n": len(x)} if x else None


S = {}
for s, rs in R.items():
    o = {"n_reps": len(rs), "c_true_eta": st([r["c_true_eta"] for r in rs]), "bulk_cox_c": st([r["bulk_cox"]["c_index"] for r in rs]),
         "bulk_cox_pearson_eta": st([r["bulk_cox"]["pearson_eta"] for r in rs]), "composition_mae": st([r["composition"]["mae_mean"] for r in rs]),
         "composition_spearman": {t: st([r["composition"]["spearman"][t] for r in rs]) for t in rs[0]["composition"]["spearman"]},
         "state_spearman": {k: {"M_hat": st([r["state"][k]["M_hat"] for r in rs]), "M_star": st([r["state"][k]["M_star"] for r in rs])} for k in rs[0]["state"]}, "methods": {}}
    for m in METHODS:
        ms = [r["methods"][m] for r in rs]; a = ms[0]
        o["methods"][m] = {"c_index": st([x["c_index"] for x in ms]), "pearson_eta": st([x["pearson_eta"] for x in ms]),
                           "attribution_share": st([x["attribution_share"] for x in ms]), "topK_precision": st([x["topK_precision"] for x in ms]),
                           "P3_endo_share_contrib": st([x["P3_endo_share_contrib"] for x in ms]), "P3_endo_share_W": st([x["P3_endo_share_W"] for x in ms]),
                           "type_spearman": {t: st([x["type_spearman"][t] for x in ms]) for t in a["type_spearman"]},
                           "terms": {k: {"spearman": st([x["terms"][k]["spearman"] for x in ms]), "sign_ok": int(sum(x["terms"][k]["sign_ok"] for x in ms)),
                                         "contrib_sd": st([x["terms"][k]["contrib_sd"] for x in ms])} for k in a["terms"]},
                           "type_var_share": {t: st([x["type_var_share"][t] for x in ms]) for t in a["type_var_share"]}}
    S[s] = o
json.dump(S, open(D / "SUMMARY.json", "w"), ensure_ascii=False, indent=1)


def f(x, d=2):
    return "–" if x is None else f"{x['median']:.{d}f} ({x['min']:.{d}f}–{x['max']:.{d}f})"


def verdict_type(x):
    return "" if x is None else (" 복원" if x["median"] >= 0.5 else "")


L = ["# Ground-truth simulation 결과 (PLAN_SIMULATION)", "",
     "GSE161529 세포로 만든 가상 유방암 환자(학습 500 · 시험 500, 생성 도너 26/26 분리), 반복별 값의 중앙값(최소–최대). CellFactor = 논문 c1–c6(PACE·CCE·CCSE 고정), CSRF 만 가상 생존으로 재적합. "
     "oracle_comp = 정답 조성 + M̂, oracle_state = π̂ + M*(뽑힌 세포 z′ 유형 평균), oracle_both = 둘 다. 판정 문구는 사전 기술 기준만 사용(유형 기여 Spearman 중앙값 ≥ 0.5 → 복원, 부호 ≥ 9/10 → 방향 복원).", ""]
for s, o in S.items():
    L += [f"## {LABEL[s]} (반복 {o['n_reps']})", "",
          f"- C-index: 정답 η {f(o['c_true_eta'], 3)} · Bulk Cox {f(o['bulk_cox_c'], 3)} · " + " · ".join(f"{m} {f(o['methods'][m]['c_index'], 3)}" for m in METHODS),
          f"- 조성 MAE {f(o['composition_mae'], 3)}; 조성 Spearman " + ", ".join(f"{t} {f(v)}" for t, v in o["composition_spearman"].items()), "",
          "| 지표 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for t in o["methods"]["CellFactor"]["type_spearman"]:
        L.append(f"| P1 유형 기여 Spearman: {t} | " + " | ".join(f(o["methods"][m]["type_spearman"][t]) + verdict_type(o["methods"][m]["type_spearman"][t]) for m in METHODS) + " |")
    L.append("| P1 귀속 비율(효과 유형) | " + " | ".join(f(o["methods"][m]["attribution_share"]) for m in METHODS) + " |")
    for k in o["methods"]["CellFactor"]["terms"]:
        L.append(f"| P2 {k}: Spearman · 부호 | " + " | ".join(f"{f(o['methods'][m]['terms'][k]['spearman'])} · {o['methods'][m]['terms'][k]['sign_ok']}/{o['n_reps']}" + (" 방향 복원" if o['methods'][m]['terms'][k]['sign_ok'] >= 0.9 * o['n_reps'] else "") for m in METHODS) + " |")
    L.append("| 상위 K 정밀도 | " + " | ".join(f(o["methods"][m]["topK_precision"]) for m in METHODS) + " |")
    L.append("| P3 내피 저산소 귀속 비율(기여 SD) | " + " | ".join(f(o["methods"][m]["P3_endo_share_contrib"]) for m in METHODS) + " |")
    L.append("| P3 보조(W·SD(M̂)) | " + " | ".join(f(o["methods"][m]["P3_endo_share_W"]) for m in METHODS) + " |")
    L.append("| Pearson(η̂, η) | " + " | ".join(f(o["methods"][m]["pearson_eta"]) for m in METHODS) + " |")
    L += ["", "상태 복원 Spearman(정답 s 대비, 체인 평균): " + "; ".join(f"{k} M̂ {f(v['M_hat'])} / M* {f(v['M_star'])}" for k, v in o["state_spearman"].items()), ""]
open(D / "SIMULATION.md", "w").write("\n".join(L))
for dst in (ROOT / "results_json/mech/simulation", ROOT / "final_results/07_mechanism/simulation"):
    dst.mkdir(parents=True, exist_ok=True)
    for p in [D / "SUMMARY.json", D / "SIMULATION.md"] + sorted(D.glob("*_r*.json")): shutil.copy(p, dst / p.name)
print("\n".join(L[:40]))
