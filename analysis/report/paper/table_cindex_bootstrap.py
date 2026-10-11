#!/usr/bin/env python3
"""cindex_bootstrap2000.py 산출 → 표. 출력 final_results/02_cindex/bootstrap_ci_B2000/{CINDEX_CI_B2000.csv, .md, PAIRED_B2000.csv}"""
import json
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parents[3] / "final_results/02_cindex/bootstrap_ci_B2000"
ORG = [("breast", "Breast"), ("liver", "Liver"), ("pancreas", "Pancreas")]
rows, prs = [], []
for o, on in ORG:
    R = json.load(open(D / f"{o}_BOOT.json"))
    for m, d in R["models"].items():
        for k, v in d.items():
            sc = "Held-out (TCGA OS)" if k == "heldout" else ("External mean" if k == "external_mean" else ("Held-out endpoint" if k.startswith("TCGA") else ("External endpoint" if k in R["external_endpoints"] else "Appendix endpoint")))
            if sc == "Held-out endpoint": continue
            rows.append({"organ": on, "scope": sc, "endpoint": k, "model": m, "C": v["point"], "boot_mean": v["boot_mean"], "se": v["se"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]})
    for m, d in R["paired_vs_CellFactor"].items():
        for k, v in d.items():
            prs.append({"organ": on, "scope": "held-out" if k == "heldout" else "external mean", "comparison": f"CellFactor − {m}", "diff": v["diff"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1], "p_two_sided": v["p_two_sided"], "verdict": v["verdict"]})
T = pd.DataFrame(rows); P = pd.DataFrame(prs)
T.to_csv(D / "CINDEX_CI_B2000.csv", index=False, float_format="%.4f"); P.to_csv(D / "PAIRED_B2000.csv", index=False, float_format="%.4f")
f = lambda r: f"{r.C:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})"
MOD = ["CellFactor", "SIDISH", "scSurv", "Scissor", "DEGAS", "Bulk Cox", "Composition-only Cox", "CellFactor (c1–c6)"]
md = ["# C-index with patient bootstrap 95% CI (B = 2,000)", "",
      "Patients resampled with replacement within each cohort (same draw for all endpoints of a cohort and for all models, seed 0). Percentile 95% CI. "
      "CellFactor = c1–c6 ensemble in all cancers (same as every other model's six seeds/chains). "
      "External mean = primary external endpoints (liver: GSE14520 and GSE76427 OS/RFS).", "",
      "## Summary", "", "| Organ | Set | " + " | ".join(MOD) + " |", "|---|---|" + "---|" * len(MOD)]
for o, on in ORG:
    for sc in ("Held-out (TCGA OS)", "External mean"):
        q = T[(T.organ == on) & (T.scope == sc)].set_index("model")
        md.append(f"| {on} | {sc} | " + " | ".join(f(q.loc[m]) if m in q.index else "—" for m in MOD) + " |")
md += ["", "## Per endpoint", "", "| Organ | Endpoint | " + " | ".join(MOD[:5]) + " |", "|---|---|" + "---|" * 5]
for o, on in ORG:
    q = T[(T.organ == on) & T.scope.isin(["External endpoint", "Appendix endpoint"])]
    for ep in dict.fromkeys(q.endpoint):
        qq = q[q.endpoint == ep].set_index("model"); tag = " (appendix)" if qq.scope.iloc[0] == "Appendix endpoint" else ""
        md.append(f"| {on} | {ep.replace('|', ' ')}{tag} | " + " | ".join(f(qq.loc[m]) if m in qq.index else "—" for m in MOD[:5]) + " |")
md += ["", "## Paired difference, CellFactor − model", "", "| Organ | Set | Comparison | Difference (95% CI) | p | Verdict |", "|---|---|---|---|---|---|"]
for _, r in P.iterrows():
    md.append(f"| {r.organ} | {r.scope} | {r.comparison} | {r['diff']:.3f} ({r.ci_lo:.3f} to {r.ci_hi:.3f}) | {r.p_two_sided:.3g} | {r.verdict} |")
open(D / "CINDEX_CI_B2000.md", "w").write("\n".join(md) + "\n"); print("\n".join(md[:20]))
