#!/usr/bin/env python3
"""PLAN_DOMAIN_GAP 요약 → results/mech/domain_gap/DOMAIN_GAP.md (+ results_json 미러, final_results/07_mechanism/domain_gap/)."""
import json, shutil
from pathlib import Path
import numpy as np
F = Path(__file__).resolve().parents[3]; D = F / "results/mech/domain_gap"; O = ("breast", "liver", "pancreas")
R = {o: json.load(open(D / f"{o}.json")) for o in O}; med = lambda xs: float(np.median(xs))
L = ["# 유사벌크–실제 벌크 도메인 간극과 conditional CORAL (PLAN_DOMAIN_GAP)", "", "## A. 입력 공간 (유사벌크 학습 도너 가방 300 vs TCGA 전체)", "",
     "| 장기 | 시점 | PC1/PC2 설명분산 | PC1 도메인 AUROC | 분류 AUROC 로지스틱 / RF | 조성 겹침 표본 로지스틱 / RF | 유전자 평균 r | 유전자 SD 비 R/P | 0 비율 P / R |", "|---|---|---|---|---|---|---|---|---|"]
for o in O:
    for v, vn in (("raw_log2cpm", "공통 단위 log2 CPM"), ("model_input", "모델 입력")):
        a = R[o]["A"]["views"][v]; z = f"{a['zero_fraction_P']:.3f} / {a['zero_fraction_R']:.3f}" if "zero_fraction_P" in a else "—"
        L.append(f"| {o} | {vn} | {a['pca_var_explained_pc1_2'][0]:.2f} / {a['pca_var_explained_pc1_2'][1]:.2f} | {a['pc_domain_auc'][0]:.2f} | {a['auroc']['logistic']:.2f} / {a['auroc']['random_forest']:.2f} | "
                 f"{a['auroc_overlap']['logistic']:.2f} / {a['auroc_overlap']['random_forest']:.2f} | {a['gene_mean_pearson']:.2f} | {a['gene_sd_ratio_R_over_P_median']:.2f} | {z} |")
L += ["", "모델 입력 시점의 로지스틱 AUROC 가 0.5 미만인 것은 도메인별 표준화로 평균이 0 에 맞춰져 선형 분리 신호가 없을 때 교차검증에서 생기는 현상(우연 수준 이하)이다. 비선형(RF)은 두 시점 모두 완전 분리.", "",
      "## B. CORAL 끔 → 켬 (체인 c1–c6 중앙값, 주 비교 = 유사벌크 학습 도너 vs TCGA held-out)", "",
      "| 장기 | CORAL 목적값 | 평균 차(공동 잔차) | 공분산 상대차 | 분류 AUROC 로지스틱 | 분류 AUROC RF | 보류 도너 상태 MAE/상수 | 외부 평균 C | 켬−끔 외부 C (95% CI) | held-out C 켬−끔 (95% CI) |", "|---|---|---|---|---|---|---|---|---|---|"]
for o in O:
    B = R[o]["B"]; g = lambda arm, k: med([B[arm][str(c)]["P_vs_Runseen"][k] for c in range(1, 7)]); ga = lambda arm, k: med([B[arm][str(c)]["P_vs_Runseen"]["auroc"][k] for c in range(1, 7)])
    st = lambda arm: med([B[arm][str(c)]["state"]["mae_ratio_mean"] for c in range(1, 7)]); b6 = R[o]["B6"]
    L.append(f"| {o} | {g('CORAL_off','coral_objective'):.3f} → {g('CORAL_on','coral_objective'):.3f} | {g('CORAL_off','mean_diff'):.2f} → {g('CORAL_on','mean_diff'):.2f} | {g('CORAL_off','cov_rel_diff'):.2f} → {g('CORAL_on','cov_rel_diff'):.2f} | "
             f"{ga('CORAL_off','logistic'):.3f} → {ga('CORAL_on','logistic'):.3f} | {ga('CORAL_off','random_forest'):.3f} → {ga('CORAL_on','random_forest'):.3f} | {st('CORAL_off'):.3f} → {st('CORAL_on'):.3f} | "
             f"{b6['point']['CORAL_off']['external_mean']:.3f} → {b6['point']['CORAL_on']['external_mean']:.3f} | {b6['external_mean']['diff_on_minus_off']:+.3f} ({b6['external_mean']['ci95'][0]:+.3f} to {b6['external_mean']['ci95'][1]:+.3f}) | "
             f"{b6['heldout']['diff_on_minus_off']:+.3f} ({b6['heldout']['ci95'][0]:+.3f} to {b6['heldout']['ci95'][1]:+.3f}) |")
L += ["", "## 사전 판정", "", "| 장기 | J-B 정렬 (목적값↓ 체인 / AUROC↓ 체인) | J-S 상태 | J-P 예후 |", "|---|---|---|---|"]
for o in O:
    j = R[o]["judgement"]; L.append(f"| {o} | {j['J_B']} ({j['coral_obj_lower_chains']}/6, {j['auroc_lower_chains']}/6) | {j['J_S']} (체인 중앙 상대 변화 {j['state_rel_change_median']:+.3f}) | {j['J_P']} |")
L += ["", "보조 비교(P vs TCGA fit, 보류 도너 가방 vs TCGA held-out)는 json 의 B.*.P_vs_Rfit · Pho_vs_Runseen. 유방 보류 도너 상태 지표는 표본 내(26도너 학습)."]
txt = "\n".join(L) + "\n"
fr = F / "final_results/07_mechanism/domain_gap"; fr.mkdir(exist_ok=True)
for d in (D, F / "results_json/mech/domain_gap", fr): (d / "DOMAIN_GAP.md").write_text(txt)
for o in O: shutil.copy(D / f"{o}.json", fr / f"{o}.json")
print(txt)
