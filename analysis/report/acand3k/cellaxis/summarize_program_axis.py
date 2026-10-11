#!/usr/bin/env python3
"""PLAN_PROGRAM_AXIS 집계(E1·E2·E3) — 판정 기준은 계획 문서(results_json/report_acand3k/cellaxis/PLAN_PROGRAM_AXIS.md)에 고정된 값.
입력(results/report_acand3k/cellaxis/):
  leiden/SSTAR_LEIDEN_{org}_A_cand3k_s{k}.json            full (기존)
  program_axis/SSTAR_LEIDEN_{org}_A_cand3k__MASK_{m}[_r{r}]_s{k}.json   E2 마스크
  program_axis/W_{org}_A_cand3k_s{k}.pt + signature_{org}_s{k}.json    E1
  program_axis/SSTAR_LEIDEN_{org}_{arm}_s{k}.json (arm ∈ B0, B1_cand3k, F) + rawgene 행(full 실행에 RAW_GENE=1)   E3
유방(2026-10-06 추가): full = results/report_acand3k/g161/leiden/SSTAR_G161_LEIDEN_A_cand3k_s{k}.json, E2·E3 은 내부(int, GSE176078)·외부(ext, GSE161529) 각각.
판정(pass)은 계획 범위(10시드 · 무작위 MASK_REPS 회)를 다 채운 장기·검정에서만 낸다 — 미완이면 null + coverage 기록(예비 판정은 provisional_criterion_met).
출력 program_axis/SUMMARY.json · SUMMARY.md"""
import json, glob, re, os
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

F = Path(os.environ.get("CELLFACTOR_PROJECT", ".")); CA = F / "results/report_acand3k/cellaxis"; PA = CA / "program_axis"
ORGS = ("liver", "pancreas", "breast"); SEEDS = range(10); ARMS3 = ("B0", "B1_cand3k", "F"); RNG = np.random.default_rng(0)
MREPS = int(os.environ.get("MASK_REPS", "50"))
SIDES = {"liver": ("int",), "pancreas": ("int",), "breast": ("int", "ext")}
J = lambda p: json.load(open(p))


def full_path(o, s): return F / f"results/report_acand3k/g161/leiden/SSTAR_G161_LEIDEN_A_cand3k_s{s}.json" if o == "breast" else CA / f"leiden/SSTAR_LEIDEN_{o}_A_cand3k_s{s}.json"


def seed_mean(p, side="int"): return J(p)[f"CP-PCH_{side}_mean"][0]


def boot_ci(x, B=2000):
    x = np.asarray(x, float); bs = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(B)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {"plan": "results_json/report_acand3k/cellaxis/PLAN_PROGRAM_AXIS.md", "E1": {}, "E2": {}, "E3": {}}
for o in ORGS:
    # ---------------- E2 ---------------- (마스크 결과가 있는 시드만 센다)
    for side in SIDES[o]:
        rows = []
        for s in SEEDS:
            pf = full_path(o, s)
            if not pf.exists(): continue
            r = {"seed": s, "full": seed_mean(pf, side)}
            for m in ("named", "free"):
                pm = PA / f"SSTAR_LEIDEN_{o}_A_cand3k__MASK_{m}_s{s}.json"
                if pm.exists(): r[m] = seed_mean(pm, side)
            rr = [seed_mean(p, side) for p in sorted(glob.glob(str(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__MASK_named_rand4_r*_s{s}.json")))]
            if rr: r["named_rand4_median"] = float(np.median(rr)); r["named_rand4_n"] = len(rr)
            if not any(k in r for k in ("named", "free", "named_rand4_median")): continue
            for k in ("named", "free", "named_rand4_median"):
                if k in r: r[f"retain_{k}"] = (r[k] - 0.5) / (r["full"] - 0.5) if r["full"] != 0.5 else None
            rows.append(r)
        if rows:
            def mean_of(k): v = [r[k] for r in rows if r.get(k) is not None]; return float(np.mean(v)) if v else None
            n_rand_gt_free = sum(1 for r in rows if "named_rand4_median" in r and "free" in r and r["named_rand4_median"] > r["free"])
            complete = len(rows) == len(SEEDS) and all(r.get("named_rand4_n", 0) >= MREPS and "named" in r and "free" in r for r in rows)
            e2 = {"per_seed": rows, "retain_named_mean": mean_of("retain_named"), "retain_free_mean": mean_of("retain_free"),
                  "n_seeds_rand4_gt_free": n_rand_gt_free, "n_seeds": len(rows), "complete": complete,
                  "coverage": {"seeds": [r["seed"] for r in rows], "rand4_reps_min": min(r.get("named_rand4_n", 0) for r in rows), "rand4_reps_planned": MREPS}}
            crit = bool(e2["retain_named_mean"] is not None and e2["retain_named_mean"] >= 0.8 and n_rand_gt_free >= 8)
            e2["provisional_criterion_met"] = crit; e2["pass"] = crit if complete else None
            out["E2"][o if side == "int" else f"{o}_ext"] = e2
    # ---------------- E1 ----------------
    rhos, cross, perm_p, tops = [], [], [], []
    for s in SEEDS:
        pw, ps = PA / f"W_{o}_A_cand3k_s{s}.pt", PA / f"signature_{o}_s{s}.json"
        if not (pw.exists() and ps.exists()): continue
        W = torch.load(pw, map_location="cpu", weights_only=False); sig = J(ps)
        named_common = set.intersection(*[{n for n in nm if not n.startswith(("free", "dim"))} for nm in W["axis_names"]])
        axes = sorted(named_common & set(sig["programs"]))
        contrib = {}
        for c, v in W["clusters"].items():
            per_chain = []
            for ch, fits in enumerate(v["fits"]):
                nm = W["axis_names"][ch]; sd = np.asarray(W["z_sd_train_focus"][ch]); w = np.mean([f["W_focus"] for f in fits], 0) * sd
                per_chain.append(np.array([w[nm.index(a)] for a in axes]))
            contrib[c] = (np.mean(per_chain, 0), np.mean(np.sign(per_chain) == np.sign(np.mean(per_chain, 0)), 0))
        cl = [c for c in contrib if c in sig["clusters"]]
        for c in cl:
            dv = np.array([sig["clusters"][c]["delta"][a] for a in axes]); wv = contrib[c][0]
            rho = spearmanr(wv, dv).correlation; rhos.append(rho)
            others = [spearmanr(wv, np.array([sig["clusters"][c2]["delta"][a] for a in axes])).correlation for c2 in cl if c2 != c]
            if others: cross.append(rho - float(np.mean(others)))
            null = [spearmanr(RNG.permutation(wv), dv).correlation for _ in range(1000)]
            perm_p.append(float((np.sum(np.array(null) >= rho) + 1) / 1001))
            top3 = [axes[i] for i in np.argsort(-wv)[:3]]; topd = [axes[i] for i in np.argsort(-dv)[:5]]
            tops.append({"seed": s, "cluster": c, "rho": float(rho), "top3_readout": top3, "top5_delta": topd, "hit": len(set(top3) & set(topd)),
                         "sign_agree_top3": [float(contrib[c][1][axes.index(a)]) for a in top3], "markers_top10": sig["clusters"][c]["markers"][:10]})
    if rhos:
        n_axes = len(axes); exp_hit = 3 * 5 / max(n_axes, 1)
        cross_p = float(np.mean([RNG.choice([-1, 1], len(cross)) @ np.abs(cross) >= np.sum(cross) for _ in range(5000)])) if cross else None   # 부호 뒤집기 순열(교차 귀무 대비 차이 > 0)
        e1 = {"n_clusters": len(rhos), "rho_median": float(np.median(rhos)), "rho_iqr": [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))],
              "rho_minus_crosscluster_mean": float(np.mean(cross)) if cross else None, "p_vs_crosscluster": cross_p,
              "frac_label_perm_p_lt_0.05": float(np.mean(np.array(perm_p) < 0.05)), "top3_hit_mean": float(np.mean([t["hit"] for t in tops])),
              "top3_hit_expected_by_chance": exp_hit, "n_named_axes": n_axes, "examples": sorted(tops, key=lambda t: -t["rho"])[:8]}
        n_seed_e1 = len({t["seed"] for t in tops}); crit = bool(e1["rho_median"] > 0 and cross_p is not None and cross_p < 0.05)
        e1.update({"n_seeds": n_seed_e1, "complete": n_seed_e1 == len(SEEDS), "provisional_criterion_met": crit, "pass": crit if n_seed_e1 == len(SEEDS) else None})
        out["E1"][o] = e1
    # ---------------- E3 ----------------
    for side in SIDES[o]:
        e3 = {}
        for arm in ARMS3:
            d = []
            for s in SEEDS:
                pa_, pb_ = full_path(o, s), PA / f"SSTAR_LEIDEN_{o}_{arm}_s{s}.json"
                if pa_.exists() and pb_.exists(): d.append(seed_mean(pa_, side) - seed_mean(pb_, side))
            if d:
                ci = boot_ci(d); ni = bool(ci[0] > -0.02)
                e3[arm] = {"diff_mean": float(np.mean(d)), "ci95": ci, "n_seeds": len(d), "complete": len(d) == len(SEEDS),
                           "noninferior_margin_-0.02": ni if len(d) == len(SEEDS) else None, "provisional_noninferior": ni}
        if side == "int":   # 항등 점검: W 저장·원시 유전자 실행(마스크 없음)의 CellFactor 값 = 기존 full JSON
            ident = []
            for p in sorted(glob.glob(str(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__RAWGENE_s*.json"))):
                s_ = int(re.search(r"_s(\d+)\.json$", p).group(1)); pf = full_path(o, s_)
                if pf.exists(): ident.append(max(abs(seed_mean(p, sd) - seed_mean(pf, sd)) for sd in SIDES[o]))
            if ident: out.setdefault("identity_full_vs_dumprun_max_abs", {})[o] = float(max(ident))
        rg = [J(p).get(f"rawgene_{side}_mean", [None])[0] for p in sorted(glob.glob(str(PA / f"SSTAR_LEIDEN_{o}_A_cand3k__RAWGENE_s*.json")))]
        rg = [v for v in rg if v is not None]
        if rg: e3["rawgene_reference_mean"] = float(np.mean(rg)); e3["rawgene_n_seeds"] = len(rg)
        if e3: e3["pass_B0"] = e3.get("B0", {}).get("noninferior_margin_-0.02"); out["E3"][o if side == "int" else f"{o}_ext"] = e3
    out.setdefault("coverage", {})[o] = {"full_seeds": [s for s in SEEDS if full_path(o, s).exists()],
        "signature_seeds": [s for s in SEEDS if (PA / f"signature_{o}_s{s}.json").exists()], "W_seeds": [s for s in SEEDS if (PA / f"W_{o}_A_cand3k_s{s}.pt").exists()],
        "E3_arms": {a: len(glob.glob(str(PA / f"SSTAR_LEIDEN_{o}_{a}_s*.json"))) for a in ARMS3}}

json.dump(out, open(PA / "SUMMARY.json", "w"), ensure_ascii=False, indent=1)
f2 = lambda v: "NA" if v is None else f"{v:.2f}"
verdict = lambda e: {True: "통과", False: "미통과", None: "판정 보류(범위 미완)"}[e.get("pass")]
L = ["# PLAN_PROGRAM_AXIS 결과 요약", "", f"판정은 10시드 · 무작위 {MREPS}회를 다 채운 경우에만 낸다. 미완이면 '판정 보류' + 예비 기준 충족 여부.", ""]
for o in ORGS:
    L.append(f"## {o}  (coverage {out.get('coverage', {}).get(o)})")
    for side in SIDES[o]:
        k = o if side == "int" else f"{o}_ext"; tag = "" if o != "breast" else (" [내부]" if side == "int" else " [외부]")
        if k in out["E2"]:
            e = out["E2"][k]
            L.append(f"- E2{tag} 명명 축 보존율 {f2(e['retain_named_mean'])} · 자유 축 {f2(e['retain_free_mean'])} · 무작위4 명명 > 자유 {e['n_seeds_rand4_gt_free']}/{e['n_seeds']} 시드 "
                     f"(무작위 최소 {e['coverage']['rand4_reps_min']}/{MREPS}회) → {verdict(e)} · 예비 기준 {e['provisional_criterion_met']}")
        if k in out["E3"]:
            for arm, v in out["E3"][k].items():
                if isinstance(v, dict): L.append(f"- E3{tag} A_cand3k − {arm}: {v['diff_mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] ({v['n_seeds']} 시드) 비열등 {v['noninferior_margin_-0.02']} (예비 {v['provisional_noninferior']})")
            if "rawgene_reference_mean" in out["E3"][k]: L.append(f"- 원시 유전자 판독{tag}(참고) {out['E3'][k]['rawgene_reference_mean']:.3f} ({out['E3'][k]['rawgene_n_seeds']} 시드)")
    if o in out["E1"]:
        e = out["E1"][o]
        L.append(f"- E1 ρ 중앙값 {f2(e['rho_median'])} ({e['n_clusters']} 군집 · {e['n_seeds']} 시드) · 교차 군집 대비 p {e['p_vs_crosscluster']} · 상위3 적중 {f2(e['top3_hit_mean'])} (우연 {f2(e['top3_hit_expected_by_chance'])}) → {verdict(e)} · 예비 기준 {e['provisional_criterion_met']}")
    L.append("")
(PA / "SUMMARY.md").write_text("\n".join(L)); print("\n".join(L))
