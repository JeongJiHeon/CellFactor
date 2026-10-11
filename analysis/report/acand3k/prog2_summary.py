"""PROG2 요약 — PROG2_{breast,liver,pancreas,lung}.json + KM → results/report_acand3k/prog2/PROG2_ACAND3K.md/json (+ results_json 미러).
모든 표에 C 정의를 적는다: 체인 평균 C(체인·시드별 C 평균) · 앙상블 C(z-평균 위험의 C). 간 외부 주 지표 = 확정 4종점(ext_primary4)."""
import json, shutil
from pathlib import Path
D = Path("results/report_acand3k/prog2"); MIR = Path("results_json/report_acand3k/prog2"); MIR.mkdir(parents=True, exist_ok=True)
P = {o: json.load(open(D / f"PROG2_{o}.json")) for o in ("breast", "liver", "pancreas", "lung")}
KO = {"breast": "유방", "liver": "간", "pancreas": "췌장", "lung": "폐(LUAD)"}
NAME = {"acand3k": "A_cand3k", "A_cand3k": "A_cand3k", "deployed": "배포 E2″", "DEPLOYED": "배포 E2″", "directbulk": "벌크 직접 ridge Cox", "direct-bulk": "벌크 직접 ridge Cox",
        "pi_only": "조성만(π)", "composition-only": "조성만(π)", "SIDISH_ens": "SIDISH", "SIDISH_NNLS": "SIDISH(NNLS)"}
nm = lambda m: NAME.get(m, m)
EXTK = {"breast": "ext", "liver": "ext_primary4", "pancreas": "ext", "lung": "ext"}
A = {"breast": "acand3k", "liver": "acand3k", "pancreas": "acand3k", "lung": "A_cand3k"}
f3 = lambda x: f"{x:.3f}"; fd = lambda d: f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
L = ["# 예후 — A_cand3k, C 정의 분리판 (사용자 검토 4번, 2026-09-30)", "",
     "**두 평가량을 섞지 않는다.** 체인 평균 C = 체인(시드)마다 C 를 구한 뒤 평균(v7 규약; 비교 모델은 시드별 평균, Scissor·벌크 직접·조성만은 1회). "
     "앙상블 C = 체인(시드) 위험을 코호트마다 z 점수로 바꿔 평균한 위험 하나의 C. 두 값은 다른 평가량이며 앙상블 C 가 보통 더 높다.",
     "주표 체인 수: 유방·간 c1–c12, 췌장·폐 c1–c6. 배포 E2″ 3시드. 외부 = 유방 SCAN-B 4 + METABRIC 2 · 간 확정 4종점(6종점은 부록) · 췌장 3코호트 · 폐 선암 5코호트 OS. held-out = TCGA OS.",
     "사후 선택된 후보이며 외부 코호트는 선택 과정에서 봤다 — 독립 검증 아님.", ""]
J = {"definitions": P["breast"]["definitions"], "organs": {}}
for o, R in P.items():
    ek = EXTK[o]; a = A[o]; pt = R["point"]; nmem = R["n_members"]
    base = [m for m in pt["chain"] if not m.startswith(a + "_") and m != a]
    L += [f"## {KO[o]}", "", f"### 점추정 (외부 = {'확정 4종점' if o == 'liver' else '주 외부 정의'})", "",
          "| 모델 | 구성원 수 | 체인 평균 C 외부 | 체인 평균 C held-out | 앙상블 C 외부 | 앙상블 C held-out |", "|---|---|---|---|---|---|"]
    for m in [a] + base:
        L.append(f"| {nm(m)} | {nmem.get(m, '')} | {f3(pt['chain'][m][ek])} | {f3(pt['chain'][m]['held'])} | {f3(pt['ens'][m][ek])} | {f3(pt['ens'][m]['held'])} |")
    L += ["", "### A_cand3k − 모델, 환자 짝 bootstrap (B=2000, 코호트별 환자 복원추출) [95% CI]", "",
          "| 비교 | 앙상블 C 외부 | 앙상블 C held-out | 체인 평균 C 외부 | 체인 평균 C held-out |", "|---|---|---|---|---|"]
    for m in base:
        k = f"{a} - {m}"; e = R["compare"]["ens"].get(k); c = R["compare"]["chain"].get(k)
        if e is None or c is None: continue
        L.append(f"| {nm(a)} − {nm(m)} | {fd(e[ek])} | {fd(e['held'])} | {fd(c[ek])} | {fd(c['held'])} |")
    L += ["", "### 체인 구성 점검(체인 평균 C) · 앙상블 크기 점검(앙상블 C)", "",
          "| 구성 | 구성원 | 체인 평균 C 외부 | 체인 평균 C held-out | 앙상블 C 외부 | 앙상블 C held-out |", "|---|---|---|---|---|---|"]
    for u in [x for x in (a + "_c1-3", a + "_c1-6", a, a + "_c7-12") if x in pt["chain"]]:
        lab = {a + "_c1-3": "c1–c3", a + "_c1-6": "c1–c6", a: "전체", a + "_c7-12": "c7–c12(새 체인)"}[u]
        L.append(f"| {lab} | {nmem.get(u, '')} | {f3(pt['chain'][u][ek])} | {f3(pt['chain'][u]['held'])} | {f3(pt['ens'][u][ek])} | {f3(pt['ens'][u]['held'])} |")
    es = R.get("ensemble_size", {})
    if es:
        L += ["", "앙상블 크기 차이(같은 체인 중첩, 앙상블 C, 환자 bootstrap): " + " · ".join(
            f"{k.replace(a + '_', '').replace(a, '전체').replace(' - ', ' − ')} 외부 {fd(v[ek])}, held-out {fd(v['held'])}" for k, v in es.items())]
    if "chain_config" in R:
        L += ["체인 구성 차이(체인 평균 C, 환자 bootstrap): " + " · ".join(f"c1–c6 − c1–c3 외부 {fd(v[ek])}" for k, v in R["chain_config"].items())]
    idk = [k for k in R if k.startswith("identity")]
    L += ["", f"항등 점검: {', '.join(f'{k} = {R[k]:.1e}' for k in idk)} (체인 평균 규약 배포 − 모델 CI 가 원 BOOT_E2PP_vs_BL 과 같음; 폐는 앙상블 A_cand3k − 모델이 REPRO_lung_unified_boot 와 같음)", ""]
    if o == "liver":
        L += [f"간 6종점 외부(부록): A_cand3k 체인 평균 {f3(pt['chain'][a]['ext'])} · 앙상블 {f3(pt['ens'][a]['ext'])}; 배포 체인 평균 {f3(pt['chain']['deployed']['ext'])}.", ""]
    J["organs"][o] = {"ext_key": ek, "point": pt, "compare": R["compare"], "ensemble_size": es, "chain_config": R.get("chain_config"), "identity": {k: R[k] for k in idk}}
# 요약 줄(간 새 체인 하락)
lv = P["liver"]["point"]["chain"]
_c712 = "acand3k_c7-12" in lv
L += ["## 요약 수치", ""] + ([
      f"- 간 새 체인 외부 하락(체인 평균 C, 확정 4종점): c1–c6 {f3(lv['acand3k_c1-6']['ext_primary4'])} → c7–c12 {f3(lv['acand3k_c7-12']['ext_primary4'])}; 전체 12체인 {f3(lv['acand3k']['ext_primary4'])}.",
      f"- 유방: c1–c6 {f3(P['breast']['point']['chain']['acand3k_c1-6']['ext'])} → c7–c12 {f3(P['breast']['point']['chain']['acand3k_c7-12']['ext'])}; 전체 {f3(P['breast']['point']['chain']['acand3k']['ext'])}."] if _c712 else
      ["- 2026-10-11: 모든 장기 c1–c6 만 사용(PLAN_ALPHA_FREE; 유방 CSRF α 자유). c7–c12 비교 줄 없음."]) + [
      "- 이전 보고서의 '앙상블 크기 점검'(c1–c3 vs c1–c6 체인 평균 C)은 체인 구성 점검이었다. 앙상블 크기는 위 앙상블 C 행으로 본다.", "",
      "## KM", "", "위험 = A_cand3k 체인 c1–c6(세 장기 공통)의 코호트 내 z-평균 앙상블 위험. 상위 20% vs 전체 로그순위(km_v6 규칙). 그림 fig_km_5models_acand3k.png.", "",
      "| 코호트 | A_cand3k 앙상블 p |", "|---|---|"]
K = json.load(open(D / "KM_5MODELS_LOGRANK_acand3k.json"))
for k, v in K.items():
    if k.endswith("|CP-PCH"): L.append(f"| {k.split('|')[0]} | {v:.1e} |")
J["km"] = {k: v for k, v in K.items() if k.endswith("|CP-PCH")}
open(D / "PROG2_ACAND3K.md", "w").write("\n".join(L) + "\n"); json.dump(J, open(D / "PROG2_ACAND3K.json", "w"), ensure_ascii=False, indent=1)
for f in D.glob("*"):
    if f.suffix in (".md", ".json") and not f.name.startswith("_shard"): shutil.copy2(f, MIR / f.name)
print("\n".join(L))
