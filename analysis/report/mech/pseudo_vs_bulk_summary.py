#!/usr/bin/env python3
"""PLAN_PSEUDO_BULK 요약표 → results/mech/pseudo_vs_bulk/PSEUDO_BULK.md (+ results_json 미러, final_results/07_mechanism 사본)."""
import json, shutil
from pathlib import Path
import numpy as np
F = Path(__file__).resolve().parents[3]; D = F / "results/mech/pseudo_vs_bulk"
L = ["# 유사벌크 대 실제 벌크 — 프로그램 앵커 feature M̂ (PLAN_PSEUDO_BULK, CellFactor c1–c6)", "",
     "값은 6체인 중앙값(괄호 = 최소–최대). SD 비·|d| 는 이름 붙은 축 중앙값, 기준 = 학습 도너 유사벌크(P_tr). P_ho = 보류 도너 유사벌크(도메인 안 도너 이동 기준선), R = TCGA 실제 벌크.", "",
     "| 장기 | 세포유형 | SD 비 R | SD 비 P_ho | J1 수축 | \\|d\\| R | \\|d\\| P_ho | J2 이동 | 유효 랭크 R / P_tr / P_ho / 정답 | J3 랭크↓ |", "|---|---|---|---|---|---|---|---|---|---|"]
f3 = lambda v: f"{v[0]:.2f} ({v[1]:.2f}–{v[2]:.2f})"
yn = lambda b: "예" if b else "아니오"
J4 = []
for org in ("breast", "liver", "pancreas"):
    r = json.load(open(D / f"{org}.json"))
    for t, j in r["judgement_c1_6"].items():
        tr = f"{j['eff_rank_truth']:.1f}" if j["eff_rank_truth"] is not None else "—"
        L.append(f"| {org} | {t} | {f3(j['sd_ratio_R'])} | {f3(j['sd_ratio_Pho'])} | {yn(j['J1_shrink'])} | {f3(j['d_abs_R'])} | {f3(j['d_abs_Pho'])} | {yn(j['J2_shift'])} | "
                 f"{j['eff_rank_R']:.1f} / {j['eff_rank_Ptr']:.1f} / {j['eff_rank_Pho']:.1f} / {tr} | {yn(j['J3_rank_down'])} |")
    J4.append((org, r["J4"]))
L += ["", "## J4 앵커–발현 정합 (체인 평균 Σπ̂M̂ 와 프로그램 유전자 발현 점수의 Spearman, 이름 붙은 축 중앙값)", "",
      "| 장기 | P_tr | R (TCGA) | R − P_tr (95% CI) | 판정 | P_ho | R held-out | 외부 코호트 |", "|---|---|---|---|---|---|---|---|"]
for org, j in J4:
    o = j["other_domains_rho_med"]; ext = ", ".join(f"{k[4:]} {v:.2f}" for k, v in o.items() if k.startswith("ext:"))
    L.append(f"| {org} | {j['rho_med_Ptr']:.3f} | {j['rho_med_R']:.3f} | {j['diff']:.3f} ({j['ci95'][0]:.3f} to {j['ci95'][1]:.3f}) | {j['verdict']} | {o['P_ho']:.3f} | {o['R_heldout']:.3f} | {ext} |")
L += ["", "## 민감도(판정에 쓰지 않음)", "", "| 장기 | 세포유형 | SD 비 R 겹침 | \\|d\\| R 겹침 | 축 구조 유사도 R / P_ho | SD 비 자유 축 R | P_tr 예측/정답 SD 비 | c7–c12 J1/J2/J3 |", "|---|---|---|---|---|---|---|---|"]
for org in ("breast", "liver", "pancreas"):
    r = json.load(open(D / f"{org}.json")); ch = [str(c) for c in range(1, 7)]
    for t in r["types"]:
        g = lambda dom, k: np.median([r["chains"][c]["per_type"][t][dom][k] for c in ch])
        pt = [r["chains"][c]["per_type"][t]["P_tr_sd_pred_over_truth_med_named"] for c in ch]; pt = [x for x in pt if x is not None]
        s = r.get("sensitivity_c7_12", {}).get(t); ss = "/".join(yn(s[k]) for k in ("J1_shrink", "J2_shift", "J3_rank_down")) if s else "—"
        L.append(f"| {org} | {t} | {g('R_overlap', 'sd_ratio_med_named'):.2f} | {g('R_overlap', 'd_abs_med_named'):.2f} | {g('R', 'struct_sim'):.2f} / {g('P_ho', 'struct_sim'):.2f} | "
                 f"{g('R', 'sd_ratio_med_free'):.2f} | {(np.median(pt) if pt else float('nan')):.2f} | {ss} |")
L += ["", f"조성 겹침 표본 수(R 중 q 가 P_tr 5–95 백분위 안): " + ", ".join(f"{o} {json.load(open(D / f'{o}.json'))['chains']['1']['n_overlap']}/{json.load(open(D / f'{o}.json'))['chains']['1']['n']['R']}" for o in ("breast", "liver", "pancreas")),
      "유방 H 는 26도너 전체로 학습되어 유방 P_ho 도 표본 내다(도너 이동 기준선 과소 가능)."]
txt = "\n".join(L) + "\n"
for d in (D, F / "results_json/mech/pseudo_vs_bulk"): (d / "PSEUDO_BULK.md").write_text(txt)
fr = F / "final_results/07_mechanism/pseudo_vs_bulk"; fr.mkdir(exist_ok=True); (fr / "PSEUDO_BULK.md").write_text(txt)
for o in ("breast", "liver", "pancreas"): shutil.copy(D / f"{o}.json", fr / f"{o}.json")
print(txt)
