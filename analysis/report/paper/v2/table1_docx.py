#!/usr/bin/env python3
"""Table 1(사용자 양식) — 코호트·종점별 환자 C-index, 10개 모델(발표된 방법 5 + 단순화 모델 5). 점추정(CI 는 Table S1).
행 굵게 = 행 최고값(부록 행 제외), 외부 평균 행 음영·기울임, 부록(GSE54236) 회색. 원천 final_results/02_cindex/bootstrap_ci_B2000/{organ}_BOOT.json, cindex_per_cohort.csv
출력 final_results/figures/manuscript_v2/Table1_full.{docx,png,pdf,csv}"""
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[4]; D = ROOT / "final_results/02_cindex"; OUT = ROOT / "final_results/figures/manuscript_v2"
R = {o: json.load(open(D / f"bootstrap_ci_B2000/{o}_BOOT.json")) for o in ("breast", "liver", "pancreas")}
cpc = pd.read_csv(D / "cindex_per_cohort.csv")
MOD = [("CellFactor", "CellFactor"), ("SIDISH", "SIDISH"), ("scSurv", "scSurv"), ("Scissor", "Scissor"), ("DEGAS", "DEGAS"),
       ("Composition-only Cox", "Comp. Cox"), ("Program-score Cox", "Program-score Cox"),
       ("Composition + program-score Cox", "Comp. + program-score Cox")]
ORD = {"Breast": ("breast", [("TCGA (held-out)", "OS", "TCGA-heldout|OS"), ("METABRIC", "OS", "METABRIC|OS"), ("METABRIC", "RFS", "METABRIC|RFS"), ("SCAN-B", "BCFi", "SCAN-B|BCFi"),
                             ("SCAN-B", "DRFi", "SCAN-B|DRFi"), ("SCAN-B", "OS", "SCAN-B|OS"), ("SCAN-B", "RFi", "SCAN-B|RFi"), ("Mean of external endpoints", "", "external_mean")]),
       "Liver": ("liver", [("TCGA (held-out)", "OS", "TCGA-heldout|OS"), ("GSE14520", "OS", "GSE14520|OS"), ("GSE14520", "RFS", "GSE14520|RFS"), ("GSE76427", "OS", "GSE76427|OS"),
                           ("GSE76427", "RFS", "GSE76427|RFS"), ("Mean of external endpoints", "", "external_mean"), ("GSE54236 †", "OS*", "GSE54236|OS*")]),
       "Pancreas": ("pancreas", [("TCGA (held-out)", "OS", "TCGA-heldout|OS"), ("GSE57495", "OS", "GSE57495|OS"), ("GSE62452", "OS", "GSE62452|OS"), ("GSE85916", "OS", "GSE85916|OS"),
                                 ("Mean of external endpoints", "", "external_mean")])}


def ne(o, k):
    if "|" not in k: return "", ""
    coh, end = k.split("|"); q = cpc[(cpc.organ == o) & (cpc.cohort == coh) & (cpc.endpoint == end)]
    return (f"{int(q.n.iloc[0]):,}", f"{int(q.events.iloc[0]):,}") if len(q) else ("", "")


rows = []
for on, (o, rr) in ORD.items():
    rows.append({"kind": "organ", "label": on})
    for c, e, k in rr:
        v = [R[o]["models"][m][k]["point"] for m, _ in MOD]; n, ev = ne(o, k)
        rows.append({"kind": "mean" if k == "external_mean" else ("appendix" if "†" in c else "row"), "label": c, "end": e, "n": n, "ev": ev, "v": v, "best": max(v)})
flat = [{"Cancer_or_cohort": r["label"], "Endpoint": r.get("end", ""), "n": r.get("n", ""), "Events": r.get("ev", ""), **({h: f"{x:.3f}" for (_, h), x in zip(MOD, r["v"])} if "v" in r else {})} for r in rows]
pd.DataFrame(flat).to_csv(OUT / "Table1_full.csv", index=False)

# ---------- docx ----------
doc = Document(); sec = doc.sections[0]; sec.orientation = WD_ORIENT.LANDSCAPE; sec.page_width, sec.page_height = sec.page_height, sec.page_width
for m in ("left_margin", "right_margin"): setattr(sec, m, Inches(0.5))
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(9)
p = doc.add_paragraph(); r = p.add_run("Table 1. Patient-level C-index of CellFactor, published methods and simplified bulk models for each cohort and survival endpoint."); r.bold = True
H1 = ["Cohort", "Endpoint", "n", "Events"] + [h for _, h in MOD]
t = doc.add_table(rows=2, cols=len(H1)); t.alignment = 1
g = t.rows[0].cells; g[4].merge(g[8]).text = "Published single-cell–bulk methods"; g[9].merge(g[11]).text = "Simplified bulk models"
for cc in (g[4], g[9]):
    for pp in cc.paragraphs:
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for rr_ in pp.runs: rr_.italic = True
BLUE = RGBColor(0x2A, 0x78, 0xD6); GREY = RGBColor(0x80, 0x80, 0x80)


def shade(cell, hexc):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), hexc); tcPr.append(s)


for i, h in enumerate(H1):
    c = t.rows[1].cells[i]; c.text = ""; run = c.paragraphs[0].add_run(h); run.bold = True
    if h == "CellFactor": run.font.color.rgb = BLUE
    if i >= 2: c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in rows:
    cells = t.add_row().cells
    if r["kind"] == "organ":
        run = cells[0].paragraphs[0].add_run(r["label"]); run.bold = True; continue
    vals = [r["label"], r["end"], r["n"], r["ev"]] + [f"{x:.3f}" for x in r["v"]]
    for i, v in enumerate(vals):
        run = cells[i].paragraphs[0].add_run(("   " if i == 0 else "") + v)
        if i >= 2: cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if i >= 4 and r["kind"] != "appendix" and abs(r["v"][i - 4] - r["best"]) < 1e-12: run.bold = True
        if r["kind"] == "mean": run.italic = True
        if r["kind"] == "appendix": run.font.color.rgb = GREY
    if r["kind"] == "mean":
        for c in cells: shade(c, "F2F1EE")
note = ("Harrell's C-index (point estimates; 95% CIs from 2,000 patient bootstrap resamples are given in Supplementary Table S1). Bold, highest value in each row. "
        "External mean = mean over the primary external endpoints of each cancer. † Appendix cohort not used as a primary endpoint; OS* treats all patients as events because event status is unavailable. "
        "CellFactor = ensemble of chains c1–c6. Comp. Cox, Cox on estimated cell-type fractions; "
        "Program-score Cox, ridge Cox on bulk program scores without deconvolution; Comp. + program-score Cox, additive Cox on fractions and program scores without cell-type × state interaction. "
        "Simplified models use the same composition estimates and patients as CellFactor, do not use the CellFactor state estimator, and were fitted on TCGA training patients only.")
doc.add_paragraph(note).runs[0].font.size = Pt(8)
doc.save(OUT / "Table1_full.docx")

# ---------- 미리보기 렌더 ----------
fig, ax = plt.subplots(figsize=(13.5, 0.27 * (len(rows) + 4))); ax.axis("off")
xs = [0.0, 0.135, 0.205, 0.245] + [0.29 + 0.087 * i for i in range(8)]; y = 1.0; dy = 1.0 / (len(rows) + 3)
ax.text((xs[4] + xs[8]) / 2 + 0.03, y, "Published single-cell–bulk methods", ha="center", fontsize=8.5, style="italic", transform=ax.transAxes)
ax.text((xs[9] + xs[11]) / 2 + 0.03, y, "Simplified bulk models", ha="center", fontsize=8.5, style="italic", transform=ax.transAxes)
y -= dy
for x, h in zip(xs, H1):
    ax.text(x + (0.03 if x >= 0.29 else 0), y, {"Comp. + program-score Cox": "Comp. +\nprogram-\nscore Cox", "Program-score Cox": "Program-\nscore Cox", "Comp. Cox": "Comp.\nCox", "Bulk Cox": "Bulk\nCox", "State-only": "State-\nonly"}.get(h, h), ha="center" if x >= 0.29 else "left", va="center", fontsize=8.5, fontweight="bold",
            color="#2a78d6" if h == "CellFactor" else "black", transform=ax.transAxes, family="serif")
ax.plot([0, 1], [y - dy * 1.0] * 2, color="black", lw=0.8, transform=ax.transAxes); y -= dy * 0.4
for r in rows:
    y -= dy
    if r["kind"] == "organ": ax.text(0, y, r["label"], fontweight="bold", fontsize=9, va="center", transform=ax.transAxes, family="serif"); continue
    if r["kind"] == "mean": ax.add_patch(plt.Rectangle((0, y - dy / 2), 1, dy, color="#f2f1ee", transform=ax.transAxes, zorder=0))
    col = "#808080" if r["kind"] == "appendix" else "black"; sty = "italic" if r["kind"] == "mean" else "normal"
    for i, (x, v) in enumerate(zip(xs, [r["label"], r["end"], r["n"], r["ev"]] + [f"{z:.3f}" for z in r["v"]])):
        bold = i >= 4 and r["kind"] != "appendix" and abs(r["v"][i - 4] - r["best"]) < 1e-12
        ax.text(x + (0.03 if x >= 0.29 else (0.012 if i == 0 else 0)), y, v, ha="center" if x >= 0.29 else "left", va="center", fontsize=8.5, color=col, style=sty,
                fontweight="bold" if bold else "normal", transform=ax.transAxes, family="serif")
ax.plot([0, 1], [y - dy * 0.6] * 2, color="black", lw=0.8, transform=ax.transAxes)
fig.savefig(OUT / "Table1_full.png", dpi=250, bbox_inches="tight"); fig.savefig(OUT / "Table1_full.pdf", bbox_inches="tight"); print("ok")
