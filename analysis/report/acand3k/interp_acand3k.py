"""A_cand3k 판 v7 보고서 5절(세포·프로그램 해석)·6절(강건성·재현) 입력. 학습 없음 · CPU.
원천: results/hvg_design/run1a/{organ}/{arm}/c{k}/{pred_bulk.pt, EVAL.json, REVAL.json, SPATIAL.json, SPATIAL_V2.json}
      · results_json/tcga_panel/repro/REPRO_{liver,breast}_c1-12.json · 00.data/tcga BRCA 생존·PAM50(유방 아형 축).
정의
 - 위험 = Σ_c π_c(α_c + W_c·M̂_c) (pred_bulk risk_all 과 일치 확인). 효과 = W_c ⊙ SD(M̂_c)(TCGA 학습 환자) — v7 eval_ext 'effects' 와 같은 정의.
 - 체인 간 안정성 = 유형별 효과 코사인·상위5 겹침을 유형 평균, 모든 체인 쌍 평균 ± SD. A 의 자유축 4개는 체인마다 정렬되지 않으므로 코사인은 명명 축만(주), 전체 축(참고).
 - 배포 E2″ 비교 = 공통 명명 축에서 체인 평균 효과의 유형별 코사인, 상위10(유형,프로그램) 겹침. 배포는 1a DEPLOYED c1–c3(배포 H·Stage 3 재현 체인).
 - 명명 축(REVAL named_axes_M2): 사전 기대 부호 항목에서 공선성 제외 뒤 (a) 예측 프로그램-위험 상관 부호 부합 (b) 자기 W 부호 부합. 순열 귀무 없음(기술).
 - PAM50(유방): 종양 구획 M̂(체인 평균)의 축별 AUROC, 기대 축 순위(분리도 순) — bio_subtype_axis_specificity.py 와 같은 규칙·같은 조인.
출력 results/report_acand3k/INTERP_ACAND3K.json/.md · fig_prog_effects_{organ}.png"""
import os
import json, glob, itertools, numpy as np, torch, pandas as pd
from collections import Counter
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
F = Path(os.environ.get("CELLFACTOR_PROJECT", ".")); RUN = F / "results/hvg_design/run1a"; OUT = F / "results/report_acand3k"; OUT.mkdir(parents=True, exist_ok=True)
ORGS = ("breast", "liver", "pancreas", "lung"); KO = {"breast": "유방", "liver": "간", "pancreas": "췌장", "lung": "폐(LUAD)"}
EXT = {"breast": ["SCAN-B|OS", "SCAN-B|DRFi", "SCAN-B|RFi", "SCAN-B|BCFi", "METABRIC|OS", "METABRIC|RFS"], "liver": ["GSE14520|OS", "GSE14520|RFS", "GSE76427|OS", "GSE76427|RFS"],
       "pancreas": ["GSE62452|OS", "GSE57495|OS", "GSE85916|OS"], "lung": [f"{c}|OS" for c in ("GSE37745", "GSE31210", "GSE50081", "GSE72094", "GSE68465")]}
def chains(a, maxc=6): return [1, 2, 3] if a == "DEPLOYED" else list(range(1, maxc + 1))
def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"mean": float(v.mean()) if len(v) else None, "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v))}
def cos(a, b): a, b = np.asarray(a, float), np.asarray(b, float); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def top(v, k=5): return set(np.argsort(-np.abs(np.asarray(v)))[:k].tolist())
def J(p): return json.load(open(p))
def pb(o, a, c): return torch.load(RUN / o / a / f"c{c}" / "pred_bulk.pt", map_location="cpu", weights_only=False)
def effects(p):
    W = p["W"].float().numpy(); M = p["M"].float().numpy(); tr = np.asarray(p["train_idx"], int)
    r = (p["pi"].float().numpy() * (p["alpha"].float().numpy()[None] + (M * W[None]).sum(-1))).sum(1)
    assert np.abs(r - np.asarray(p["risk_all"], float)).max() < 1e-4
    return W * M[tr].std(0), list(p["programs"]), list(p["cell_types"])
res = {"definitions": __doc__, "programs": {}, "state_named": {}, "pam50": {}, "s_c": {}, "robustness": {}, "not_regenerated": []}
MD = ["# A_cand3k — 세포·프로그램 해석 · 강건성 입력 (v7 5·6절 대응)", "", "원천·정의는 INTERP_ACAND3K.json definitions. 사후 기술 표이며 세포 수준 위험 귀속의 검증이 아니다.", ""]
# ───────── 1. 프로그램 효과
for o in ORGS:
    E = {c: effects(pb(o, "A_cand3k", c)) for c in chains("A_cand3k")}; D = {c: effects(pb(o, "DEPLOYED", c)) for c in chains("DEPLOYED")}
    names = E[1][1]; types = E[1][2]; named = [i for i, n in enumerate(names) if not n.startswith("free_")]; free = [i for i, n in enumerate(names) if n.startswith("free_")]
    assert all(E[c][1] == names for c in E)
    dn = D[1][1]; dnamed = [i for i, n in enumerate(dn) if not n.startswith("free_")]
    Emean = np.mean([E[c][0] for c in E], 0); Dmean = np.mean([D[c][0] for c in D], 0)
    # 체인 간 안정성
    pc_n, pc_a, pt5, sgn = [], [], [], []
    for a, b in itertools.combinations(sorted(E), 2):
        pc_n.append(np.mean([cos(E[a][0][t, named], E[b][0][t, named]) for t in range(len(types))]))
        pc_a.append(np.mean([cos(E[a][0][t], E[b][0][t]) for t in range(len(types))]))
        pt5.append(np.mean([len(top(E[a][0][t, named]) & top(E[b][0][t, named])) for t in range(len(types))]))
    dpc = [np.mean([cos(D[a][0][t, dnamed], D[b][0][t, dnamed]) for t in range(len(types))]) for a, b in itertools.combinations(sorted(D), 2)]
    # 상위 (유형, 프로그램) — 명명 축, 체인 평균 효과
    flat = [(float(Emean[t, k]), types[t], names[k], int(sum(np.sign(E[c][0][t, k]) == np.sign(Emean[t, k]) for c in E))) for t in range(len(types)) for k in named]
    adv = sorted(flat, key=lambda x: -x[0])[:10]; pro = sorted(flat, key=lambda x: x[0])[:10]
    # 배포 비교(공통 명명 축)
    common = [n for n in names if not n.startswith("free_") and n in dn]; ia = [names.index(n) for n in common]; idd = [dn.index(n) for n in common]
    cos_dep = {types[t]: cos(Emean[t, ia], Dmean[t, idd]) for t in range(len(types))}
    dflat = [(float(Dmean[t, k]), types[t], dn[k]) for t in range(len(types)) for k in dnamed]
    dtop = {(x[1], x[2]) for x in sorted(dflat, key=lambda x: -abs(x[0]))[:10]}; atop = {(x[1], x[2]) for x in sorted(flat, key=lambda x: -abs(x[0]))[:10]}
    # 자유축 몫
    fs = [float(np.abs(E[c][0][:, free]).sum() / np.abs(E[c][0]).sum()) for c in E]; dfs = [float(np.abs(D[c][0][:, [i for i, n in enumerate(dn) if n.startswith('free_')]]).sum() / np.abs(D[c][0]).sum()) for c in D]
    only_dep = sorted(n for n in dn if not n.startswith("free_") and n not in names)
    res["programs"][o] = {"n_named_axes": len(named), "n_free_axes": len(free), "deployed_named_axes": len(dnamed), "dropped_vs_deployed_dict": only_dep,
                          "chain_stability": {"effect_cos_named": ms(pc_n), "effect_cos_all_axes": ms(pc_a), "top5_overlap_named": ms(pt5), "n_pairs": len(pc_n)},
                          "deployed_chain_stability_named": ms(dpc),
                          "top_adverse": [{"effect": e, "cell_type": t, "program": n, "sign_agree_chains": s, "n_chains": len(E)} for e, t, n, s in adv],
                          "top_protective": [{"effect": e, "cell_type": t, "program": n, "sign_agree_chains": s, "n_chains": len(E)} for e, t, n, s in pro],
                          "vs_deployed": {"n_common_named": len(common), "effect_cos_by_type": cos_dep, "effect_cos_mean": float(np.mean(list(cos_dep.values()))),
                                          "top10_overlap": len(dtop & atop), "shared_top10": sorted(f"{a}·{b}" for a, b in dtop & atop)},
                          "free_share_absW": {"A_cand3k": ms(fs), "DEPLOYED": ms(dfs)}}
    # 그림: 상위 |효과| 명명 축 16개 × 유형
    kk = sorted(named, key=lambda k: -np.abs(Emean[:, k]).max())[:16]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(types), 0.9 + 0.36 * len(kk)))
    Mx = Emean[:, kk].T; v = float(np.abs(Mx).max()) or 1.0
    cm = LinearSegmentedColormap.from_list("d", ["#2166ac", "#f7f7f7", "#b2182b"])
    ax.imshow(Mx, cmap=cm, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    for i, k in enumerate(kk):
        for t in range(len(types)):
            s = sum(np.sign(E[c][0][t, k]) == np.sign(Emean[t, k]) for c in E)
            ax.text(t, i, f"{Emean[t, k]:+.2f}\n{s}/6", ha="center", va="center", fontsize=6.5, color="#111" if abs(Emean[t, k]) < 0.6 * v else "white")
    ax.set_yticks(range(len(kk))); ax.set_yticklabels([names[k] for k in kk], fontsize=7.5); ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=35, ha="right", fontsize=7.5)
    ax.set_title(f"{o} · A_cand3k effect W⊙SD(M̂)\n6-chain mean · k/6 = chains with same sign", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / f"fig_prog_effects_{o}.png", dpi=170); plt.close(fig)
    P = res["programs"][o]; st = P["chain_stability"]
    MD += [f"## 프로그램 효과 — {KO[o]}", "", f"명명 축 {len(named)}(배포 {len(dnamed)}) + 자유 {len(free)}. 배포 사전에만 있는 축: {', '.join(only_dep) or '없음'}.",
           f"체인 간 효과 코사인(명명) {st['effect_cos_named']['mean']:.3f} ± {st['effect_cos_named']['sd']:.3f} (배포 3체인 {P['deployed_chain_stability_named']['mean']:.3f}) · 상위5 겹침 {st['top5_overlap_named']['mean']:.2f}/5 · "
           f"배포와 공통 축 효과 코사인 {P['vs_deployed']['effect_cos_mean']:.3f} · 상위10 겹침 {P['vs_deployed']['top10_overlap']}/10 · 자유축 |효과| 몫 {P['free_share_absW']['A_cand3k']['mean']:.3f} (배포 {P['free_share_absW']['DEPLOYED']['mean']:.3f})", "",
           "| 방향 | 유형 · 프로그램 | 효과 | 같은 부호 체인 |", "|---|---|---|---|"]
    MD += [f"| 불리 | {x['cell_type']} · {x['program']} | {x['effect']:+.3f} | {x['sign_agree_chains']}/6 |" for x in P["top_adverse"][:6]]
    MD += [f"| 유리 | {x['cell_type']} · {x['program']} | {x['effect']:+.3f} | {x['sign_agree_chains']}/6 |" for x in P["top_protective"][:6]] + [""]
    if o == "breast": MD += ["주의: 배포 E2″ 에서 BASAL_PROGRAM·HM_Myc_Targets_V1 은 단변량(+)과 다변량(−) 부호가 뒤집혔고 π̂_tumor 와 상관했다(results/report/BREAST_PROLIF_SIGN_CHECK.json). A_cand3k 의 BASAL_PROGRAM 유리(−) 효과도 다변량 계수라 생물학적 보호 역할로 읽지 않는다. 효과는 위험 가중치의 방향이지 세포 수준 위험 귀속이 아니다.", ""]
# ───────── 2. 상태·판독·명명 축 (REVAL)
ARMS = ["DEPLOYED", "F", "A_cand3k"]
MD += ["## 상태 복원·판독·명명 축 (REVAL, 보류 도너 가방)", "", "| 장기 | 팔 | n | 상태 MAE/SD↓ | 상수 대비 MAE 비↓ | 도너 간 r 중앙 | 판독 r(명명) | 기대 항목 판독 A·B | 명명 축 부합(예측-위험) | 자기 W 부호 부합 |", "|---|---|---|---|---|---|---|---|---|---|"]
for o in ORGS:
    for a in ARMS:
        rows = [J(RUN / o / a / f"c{c}" / "REVAL.json") for c in chains(a) if (RUN / o / a / f"c{c}" / "REVAL.json").exists()]
        def g(f): return ms([f(r) for r in rows])
        agr, own, nn = [], [], []
        for r in rows:
            it = [x for x in r.get("named_axes_M2", []) if not x.get("excluded_collinear")]
            if it: agr.append(np.mean([x["sign_agrees"] for x in it])); own.append(np.mean([x["own_W_agrees"] for x in it])); nn.append(len(it))
        d = {"n": len(rows), "state_mae_over_sd": g(lambda r: r["state_common"]["mae_over_sd_mean"]), "mae_ratio_vs_const": g(lambda r: r["state_common"]["mae_ratio_vs_const_mean"]),
             "donor_pearson_median": g(lambda r: r["state_common"].get("donor_pearson_median_mean", np.nan)), "readout_named_r": g(lambda r: r["grades"]["named_nonHM_median_r"]),
             "readable_expect": g(lambda r: r["grades"].get("readable_expect", np.nan)), "n_expect": len(rows[0]["grades"].get("expect_items", [])) if rows else 0,
             "named_M2_sign_agree": ms(agr), "named_M2_ownW_agree": ms(own), "named_M2_n_items": ms(nn)}
        res["state_named"].setdefault(o, {})[a] = d
        f3 = lambda x: "–" if x["mean"] is None else f"{x['mean']:.3f}"
        MD.append(f"| {KO[o]} | {a} | {d['n']} | {f3(d['state_mae_over_sd'])} | {(f3(d['mae_ratio_vs_const']) if (d['mae_ratio_vs_const']['mean'] or 0) < 10 else '불안정*')} | {f3(d['donor_pearson_median'])} | {f3(d['readout_named_r'])} | "
                  f"{d['readable_expect']['mean']:.1f}/{d['n_expect']} | {f3(d['named_M2_sign_agree'])} ({d['named_M2_n_items']['mean']:.0f}항목) | {f3(d['named_M2_ownW_agree'])} |")
MD += ["", "*췌장 상수 대비 MAE 비는 상수 기준 MAE 가 0 에 가까운 (유형, 축) 칸 때문에 수백~천 배로 폭주한다(REVAL 원값 그대로 JSON 에 보존). 비교에 쓰지 않는다. 췌장·폐 도너 간 r 는 보류 도너 수 부족으로 REVAL 이 산출하지 않았다. 명명 축 항목 수는 공선성 제외(|corr(M̂, π_tumor)|>0.3)와 사전 차이로 팔마다 다르다.", "상태 MAE/SD·도너 간 r = REVAL state_common(보류 val+test 가방, 공통 K0DEDUP 판독 공간). 명명 축 = REVAL named_axes_M2(공선성 제외 뒤 부합 비율, 순열 귀무 없음).", ""]
# ───────── 3. PAM50 (유방)
s = pd.read_csv(os.path.join(os.environ.get("CELLFACTOR_DATA", "data"), "tcga/BRCA_survival.raw"), sep="\t").dropna(subset=["OS", "OS.time"]); s["key"] = list(zip(s["OS.time"].astype(float), s["OS"].astype(float)))
uniq = {k for k, v in Counter(s["key"]).items() if v == 1}; pid = s[s["key"].isin(uniq)].set_index("key")["_PATIENT"].to_dict()
pam = {r["patientId"]: r["value"] for r in J(os.path.join(os.environ.get("CELLFACTOR_DATA", "data"), "tcga/tcga_brca_pam50_cbioportal.json"))}
EXPECT = {"BRCA_Basal": ("BASAL_PROGRAM", +1), "BRCA_LumA": ("ER_SIGNALING", +1), "BRCA_LumB": ("LUMINAL_PROLIF", +1), "BRCA_Her2": ("HER2_PROGRAM", +1)}
MD += ["## 유방 PAM50 아형 축 (종양 구획 M̂, 학습·선택에 PAM50 미사용)", "", "| 팔 | 아형 | n | 기대 축 | AUROC | 순위/축 수 | 방향 |", "|---|---|---|---|---|---|---|"]
for a in ARMS:
    Ps = [pb("breast", a, c) for c in chains(a)]; prog = [n for n in Ps[0]["programs"]]; tum = list(Ps[0]["cell_types"]).index("Cancer Epithelial")
    Mt = np.mean([p["M"].float().numpy()[:, tum] for p in Ps], 0); t = np.asarray(Ps[0]["time_all"], float); e = np.asarray(Ps[0]["event_all"], float)
    lab = np.array([pam.get(pid.get((tt, ee)), "") for tt, ee in zip(t, e)], dtype=object); ok = lab != ""
    out = {"n_labelled": int(ok.sum()), "n_axes": len(prog), "per_subtype": {}}
    for sub, (ax_, sign) in EXPECT.items():
        y = (lab[ok] == sub).astype(float)
        if ax_ not in prog: out["per_subtype"][sub] = {"n": int(y.sum()), "expected_axis": ax_, "status": "축 없음(사전에 없음)"}; MD.append(f"| {a} | {sub} | {int(y.sum())} | {ax_} | – | 축 없음 | – |"); continue
        auc = np.array([roc_auc_score(y, Mt[ok, k]) for k in range(Mt.shape[1])]); order = np.argsort(-np.maximum(auc, 1 - auc)); k = prog.index(ax_); rank = int(np.where(order == k)[0][0]) + 1
        out["per_subtype"][sub] = {"n": int(y.sum()), "expected_axis": ax_, "auroc": float(auc[k]), "rank": rank, "direction_correct": bool((auc[k] > 0.5) == (sign > 0)), "top3": [(prog[j], float(auc[j])) for j in order[:3]]}
        MD.append(f"| {a} | {sub} | {int(y.sum())} | {ax_} | {auc[k]:.3f} | {rank}/{len(prog)} | {'맞음' if (auc[k] > 0.5) == (sign > 0) else '반대'} |")
    res["pam50"][a] = out
MD += ["", f"PAM50 조인 = (OS.time, OS) 유일 키(bio_subtype_axis_specificity.py 와 같음). 표지 {res['pam50']['A_cand3k']['n_labelled']}명.", ""]
# ───────── 4. 세포당 위험 s_c (유방 TNBC96, 1a SPATIAL.json)
SCK = [("s_c|T-cells", "T세포 → TIL·림프구", "<0.5"), ("s_c|B-cells", "B세포 → TIL·림프구", "<0.5"), ("s_c|Myeloid", "골수성 → TIL·림프구", "<0.5"), ("s_c|Cancer Epithelial", "종양상피 → Tumor", ">0.5"), ("s_c|CAFs", "CAF → Tumor", ">0.5")]
MD += ["## 세포당 위험 s_c → 라벨 영역 (유방 TNBC96, 절편 내 AUROC, P0)", "", "| s_c → 라벨 | 기존 가정 | 배포(1a 3체인) | F | A_cand3k (6체인 평균 ± SD) | A 판정 |", "|---|---|---|---|---|---|"]
for k, lab, exp in SCK:
    v = {a: ms([J(RUN / "breast" / a / f"c{c}" / "SPATIAL.json")["spatial"].get(k, {}).get("auroc", np.nan) for c in chains(a)]) for a in ARMS}
    res["s_c"][k] = {"label": lab, "expected": exp, **v}; m = v["A_cand3k"]["mean"]; okk = (m < 0.5) if exp == "<0.5" else (m > 0.5)
    MD.append(f"| {lab} | {exp.replace('<', '&lt;').replace('>', '&gt;')} | {v['DEPLOYED']['mean']:.3f} | {v['F']['mean']:.3f} | {m:.3f} ± {v['A_cand3k']['sd']:.3f} | {'기존 가정과 일치' if okk else '반대'} |")
MD += ["", "간 s_c(v7 eval_ext 확장 평가)는 1a 공간 평가에 없어 산출하지 않았다.", ""]
# ───────── 5. 강건성
LV = J(F / "results_json/tcga_panel/repro/REPRO_liver_c1-12.json"); BR = J(F / "results_json/tcga_panel/repro/REPRO_breast_c1-12.json")
rob = {}
for o in ORGS:
    cs = sorted(int(p.parent.name[1:]) for p in (RUN / o / "A_cand3k").glob("c*/EVAL.json"))
    cs = [c for c in cs if c <= 6]   # 2026-10-11: c1–c6 만(PLAN_ALPHA_FREE; 유방 c7–c12 는 α=0 판으로 남아 있음)
    ev = {c: J(RUN / o / "A_cand3k" / f"c{c}" / "EVAL.json")["prognosis"] for c in cs}
    ext = {c: float(np.mean([ev[c][k] for k in EXT[o]])) for c in cs}; held = {c: ev[c].get("TCGA-heldout|OS", np.nan) for c in cs}
    rob[o] = {"chains": cs, "ext_c1_6": ms([ext[c] for c in cs if c <= 6]), "ext_c7_12": ms([ext[c] for c in cs if c > 6]), "ext_all": ms(list(ext.values())),
              "held_c1_6": ms([held[c] for c in cs if c <= 6]), "held_c7_12": ms([held[c] for c in cs if c > 6]), "held_all": ms(list(held.values()))}
L0, L1 = LV["summary"]["p0"]["s1:A_cand3k"], LV["summary"]["p1"]["s1:A_cand3k"]
rob["liver"]["repro_p0"] = {"conf4": L0["conf4"], "held": L0["TCGA-heldout|OS"]}; rob["liver"]["repro_p1"] = {"conf4": L1["conf4"], "held": L1["TCGA-heldout|OS"]}
rob["liver"]["boot_p0"] = {k: LV["bootstrap"]["p0"]["compare"][k] for k in ("s1:A_cand3k − s1:F", "s1:A_cand3k − ref:DEPLOYED", "s1:A_cand3k − hfix:HFIX_cand3k")}
rob["liver"]["boot_p1"] = {k: LV["bootstrap"]["p1"]["compare"][k] for k in ("s1:A_cand3k − s1:F", "s1:A_cand3k − ref:DEPLOYED")}
rob["breast"]["repro_rows"] = BR["rows"]["A_cand3k"]; rob["breast"]["boot"] = {k: BR["boot"][k] for k in ("A_cand3k − F", "A_cand3k − DEPLOYED")}
res["robustness"] = rob
MD += ["## 강건성 — 체인 변동 · 새 체인 · 전처리 민감도", "", "| 장기 | 체인 | 외부 C c1–c6 (평균 ± SD) | c7–c12 | held-out c1–c6 | c7–c12 |", "|---|---|---|---|---|---|"]
f = lambda x: "–" if x["mean"] is None else f"{x['mean']:.3f} ± {x['sd']:.3f}" if x["sd"] is not None else f"{x['mean']:.3f}"
for o in ORGS:
    r = rob[o]; MD.append(f"| {KO[o]} | {len(r['chains'])} | {f(r['ext_c1_6'])} | {f(r['ext_c7_12'])} | {f(r['held_c1_6'])} | {f(r['held_c7_12'])} |")
bp = lambda d: f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] {d['verdict']}"
MD += ["", "간 c1–c12 (확정 4종점, 환자 짝 bootstrap 앙상블 위험 B=2000; REPRO_liver_c1-12):", ""]
for k, v in rob["liver"]["boot_p0"].items(): MD.append(f"- P0 {k}: 외부 {bp(v['conf4'])} · held-out {bp(v['TCGA-heldout|OS'])}")
for k, v in rob["liver"]["boot_p1"].items(): MD.append(f"- P1(중심화 민감도) {k}: 외부 {bp(v['conf4'])}")
MD.append(f"- 시드 짝 Δ(F 대비) 확정4 P0 {L0['conf4']['delta']['mean']:+.3f} ± {L0['conf4']['delta']['se']:.3f} ({L0['conf4']['delta']['n_improve']}/{L0['conf4']['delta']['n']}) · P1 {L1['conf4']['delta']['mean']:+.3f} ± {L1['conf4']['delta']['se']:.3f} ({L1['conf4']['delta']['n_improve']}/{L1['conf4']['delta']['n']})")
bb = lambda d: f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] {d['verdict']}"
MD += ["", "유방 c1–c12 (외부 = 규약 중심화 경로 SCAN-B 4 + METABRIC 2; REPRO_breast_c1-12):", ""]
for k, v in rob["breast"]["boot"].items(): MD.append(f"- {k}: 외부 {bb(v['C_ext'])} · held-out {bb(v['TCGA|OS'])}")
rr = rob["breast"]["repro_rows"]; MD.append(f"- 시드 짝 Δ(F 대비) 외부 {rr['외부C']['delta']:+.3f} ± {rr['외부C']['se']:.3f} ({rr['외부C']['n_improve']}/{rr['외부C']['n']}) · TNBC 종양 P0 {rr['TNBC종양 P0']['delta']:+.3f} ({rr['TNBC종양 P0']['n_improve']}/{rr['TNBC종양 P0']['n']}) · P1 {rr['TNBC종양 P1']['delta']:+.3f} ({rr['TNBC종양 P1']['n_improve']}/{rr['TNBC종양 P1']['n']})")
# ───────── 6. 재생성하지 않은 v7 항목
NR = [("5절 벌크→세포상태 직접 복원(유방 GSE161529 외부 도너)", "A_cand3k 는 교사·디코더·H 입력이 cand3k 목록이라 G161 세포의 cand3k 토큰 scGPT 재임베딩과 A_cand3k Stage 1′ 인코딩이 새로 필요하다(기존 state_recovery_arms.py 는 배포 z′·G176 레지스트리 전용). 대신 REVAL 보류 도너 가방 상태 지표(위 표)를 쓴다."),
      ("5절 벌크→세포상태 직접 복원(간 보류 도너)", "state_recovery_arms.py 는 배포 Stage 1′ z′·전체 패널 H 에 묶여 있다. 같은 목적의 1a 정의(REVAL state_common: 보류 val+test 도너 가방, 공통 판독 공간)로 대체."),
      ("5절 명명 축 문헌 부호 부합의 순열 귀무(R=200)", "named_axes_arms.py 가 배포·절제 H 체크포인트와 배포 사전에 묶여 있고 순열마다 Stage 3 재적합이 필요. REVAL named_axes_M2 의 관측 부합률만 기술로 보고한다."),
      ("5절 세포축 Leiden 군집 판독(SSTAR_LEIDEN·SSTAR_ALL8·G161 외부)", "군집 스코어러·기준선 고정 세포 점수·Leiden 분할이 배포 z′ 파이프라인 전용이고, 유방 G161 은 cand3k 재임베딩이 필요하다. 새 파이프라인이 필요해 이번 범위에서 제외."),
      ("5절 간 세포당 위험 s_c(MIF·H&E 라벨)", "v7 값은 절제 확장 평가(eval_ext) 산출물. 1a 공간 평가(spatial_1a.py)는 간 s_c 를 계산하지 않는다."),
      ("5절 유방 TIL 저위험 분해·EXP15 W 순열", "배포 헤드 전용 w_exp 스크립트 산출물이며 A_cand3k 로 다시 돌리지 않았다."),
      ("6절 CORAL/paired 4설정 절제(A–E 블록, CORAL X 상세)", "절제 H 는 배포 E2″ 구조로만 학습돼 있다. A_cand3k 에 같은 절제를 하려면 4설정×6시드 재학습이 필요(범위 밖)."),
      ("6절 v6 재현·조인 감사 행", "배포 산출물 재현 기록이라 모델과 무관. 원 v7 값을 그대로 인용 가능.")]
res["not_regenerated"] = [{"item": a, "reason": b} for a, b in NR]
MD += ["", "## v7 항목 중 A_cand3k 로 다시 만들지 않은 것", ""] + [f"- **{a}** — {b}" for a, b in NR]
json.dump(res, open(OUT / "INTERP_ACAND3K.json", "w"), ensure_ascii=False, indent=1, default=float); open(OUT / "INTERP_ACAND3K.md", "w").write("\n".join(MD) + "\n")
print("\n".join(MD))
