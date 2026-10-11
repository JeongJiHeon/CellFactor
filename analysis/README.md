# Analysis scripts

Scripts used to compute the results and draw the figures and tables of the paper, as run in the development
project. They are provided as a record of how each reported number was obtained; most of them read
intermediate outputs of that project (model predictions, saved chain outputs and summary files under
`results/`, `results_json/` and `final_results/`) and some import development modules under `code/lib/` and
`code/eval/hvg_run1a/`, which are not part of this repository. They therefore do not run stand-alone on a fresh
clone. The model itself is defined in [`../model.py`](../model.py) and trained with [`../train.py`](../train.py).

The folder structure mirrors the development project (`code/report/...` there, `analysis/report/...` here),
so relative paths inside the scripts resolve against the project root. Absolute server paths were replaced by
two environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CELLFACTOR_PROJECT` | root of the development project | `.` |
| `CELLFACTOR_DATA` | public raw data (TCGA clinical/phenotype tables, GEO and cBioPortal downloads) | `data` |

Some comments and docstrings are in Korean.

## Figures and tables

| Item | Scripts (under `analysis/`) |
|---|---|
| Fig. 2a–c (UMAP, high-risk cells, composition) | `report/paper/v2/umap_rescore.py` (cell-level risk and top-10% selection), `report/paper/fig2_main.py` (panels) |
| Fig. 2d,e; Additional file 1: Figs. S2, S3 (within-cell-type subpopulations) | `report/acand3k/cellaxis/probe_leiden_arm.py` (Leiden targets and readout), `report/acand3k/cellaxis/cluster_program_signature.py` (model-independent cluster signatures), `report/acand3k/cellaxis/summarize_program_axis.py`, `report/paper/program_axis_hepexcl.py` (post hoc exclusion of hepatocyte-marker clusters), `report/paper/fig_program_axis_full.py` (panels) |
| Fig. 2f; Fig. S1 (Kaplan–Meier, log-rank) | `report/km_v6.py`, `report/acand3k/km_acand3k.py`, `report/acand3k/km_values_acand3k.py`, `report/acand3k/km_prog2.py`, `report/paper/v2/km_c1_6.py`, `report/paper/dump_km_5models.py` |
| Figure 2 assembly; Figs. S1–S3 layout | `report/paper/v2/fig2_v2.py`, `report/paper/v2/supp_v2.py` |
| Table 1; Table S1 (C-index, bootstrap CIs, paired differences) | `report/acand3k/prog2_acand3k.py`, `report/acand3k/prog2_summary.py`, `report/paper/cindex_bootstrap2000.py`, `report/paper/table_cindex_bootstrap.py`, `report/paper/v2/tables_v3.py`, `report/paper/v2/table1_docx.py` |
| Simplified models (Table 1); Table S9 (risk decomposition) | `report/mech/ablation_risks.py`, `report/paper/table_simple_baselines.py`, `report/mech/risk_decomposition.py` |
| Fig. 3 (spatial transcriptomics) | `report/paper/fig_spatial_paper3.py`, `report/paper/path_colors.py` |
| Fig. 4a–c (program effects, chain stability, PAM50) | `report/acand3k/interp_acand3k.py` |
| Fig. 4d (bulk-to-state recovery in GSE161529) | `report/acand3k/sr/sr_chain.py` |
| Fig. 4e; Tables S6, S7 (TCGA phenotypes, TP53) | `report/mech/exp2_tcga_pheno.py`, `report/mech/exp2b_tp53_state.py` |
| Figure 4 assembly | `report/paper/v2/fig4_v3.py` |
| Table 2; Table S5 (clinical covariate adjustment) | `report/mech/exp3_covariate_cox.py`, `report/paper/v2/tables_v3.py` |
| Fig. S4 (literature-expected directions, permutation null) | `eval/bio_literature_concordance.py` (expected directions), `report/acand3k/named_null/named_null_arm.py`, `report/paper/v2/supp_s4_s5.py` |
| Fig. S5; Table S8 (program deletion) | `report/mech/exp1_deletion.py`, `report/paper/v2/supp_s4_s5.py` |
| Fig. S6 (pseudo-bulk–bulk domain gap, conditional CORAL) | `report/mech/domain_gap.py`, `report/mech/domain_gap_diag.py`, `report/mech/domain_gap_summary.py`, `report/mech/pseudo_vs_bulk.py`, `report/mech/pseudo_vs_bulk_summary.py`, `report/paper/v2/supp_s6.py` |
| Fig. S7 (semi-synthetic cohorts) | `report/mech/simulation.py`, `report/mech/simulation_summary.py`, `report/paper/v2/supp_s7.py` |
| Additional files 1 and 2 | `report/paper/v2/supp_document.py`, `report/paper/v2/supp_tables.py` (with `S3_PROGRAM_REFERENCES.json`) |

Not included: the computation of the UMAP coordinates, the preparation of spatial spots and the inference of
spot-level risk, the runs of the comparator methods (SIDISH, scSurv, Scissor and DEGAS) and the
cell-type label transfer for GSE161529.
