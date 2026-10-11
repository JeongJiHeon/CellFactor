# CellFactor

CellFactor factorizes bulk-derived survival risk into cell-type composition and within-cell-type state.
From a bulk tumour transcriptome `x`, it estimates the cell-type composition `π(x)` and the mean program
state `M_c(x)` of each cell type, and computes the patient risk (Cox linear predictor) as

```
r(x) = Σ_c π_c(x) · ( α_c + W_c · M_c(x) )
```

so that every patient's risk decomposes exactly into additive cell-type- and program-level contributions.

> Jeong J, Kim I-H, Nam Y, Cho BH. *CellFactor: factorizing bulk-derived survival risk into cell-type
> composition and within-cell-type state.* (manuscript in preparation)

## Repository contents

| File | Contents |
|---|---|
| `model.py` | Model definitions: `ProgramAnchoredNBVAE` (PACE, Stage 1), `CompositionEstimator` (CCE, Stage 2A), `StateEstimator` (CCSE, Stage 2B), `CellFactorRisk` (CSRF, Stage 3), the losses (masked state, paired-delta, conditional CORAL, Cox), axis calibration, and checkpoint loaders. `CellFactor` wraps the frozen modules for inference. |
| `train.py` | Training of one model chain, stage by stage, with the settings used in the paper. |

## Model stages

| Stage | Module | Trained on | Output |
|---|---|---|---|
| 1 — PACE | `ProgramAnchoredNBVAE` | reference scRNA-seq (training donors) | program-anchored latent state `z′` of every cell (named program axes + 4 free axes) |
| 2A — CCE | `CompositionEstimator` | TCGA bulk with tumour purity, leukocyte fraction and teacher immune fractions | coarse (tumour / immune / other) and immune-relative composition |
| — | calibration + bags | paired pseudo-bulk bags from training donors | axis scales and reliabilities; fixed validation bags |
| 2B — CCSE | `StateEstimator` | online paired pseudo-bulk bags + TCGA bulk (conditional CORAL) | cell-type-specific mean state `M` |
| 3 — CSRF | `CellFactorRisk` | TCGA training patients, overall survival (ridge Cox) | `α` and `W` |

Survival labels are used only in Stage 3; PACE, CCE and CCSE are frozen before it. The reported model is the
cohort-wise z-score mean of the risks of six independently trained chains (c1–c6) per cancer type.

## Requirements

- Python 3.10
- PyTorch (CUDA GPU recommended)
- NumPy
- lifelines (only for the held-out C-index printed after Stage 3)

## Usage

`train.py` reads pre-processed inputs from a data root, set with `--root` or the `CELLFACTOR_ROOT` environment
variable (default: the current directory). Each cancer type (`breast`, `liver`, `pancreas`) has its own folder
under the data root (see [Inputs](#inputs)); the file names are defined in the `ORGANS` dictionary at the top of
`train.py`.

```bash
# one full chain (Stage 1 -> bags -> Stage 2B -> Stage 3), reusing the three Stage 2A checkpoints
python train.py --organ liver --chain 1 --stage all --out runs/liver_c1 --root /path/to/data --device cuda:0

# optional: train your own Stage 2A (one seed per call; the paper uses seeds 20260821-20260823)
python train.py --organ liver --stage 2a --seed 20260821 --out runs/liver_2a --root /path/to/data
python train.py --organ liver --chain 1 --stage all --out runs/liver_c1 --a_dir runs/liver_2a --root /path/to/data

# single stages
python train.py --organ liver --chain 1 --stage stage1 --out runs/liver_c1 --root /path/to/data   # also: bags, h, stage3
```

Seeds follow the paper: chain `c` uses Stage 1 seed `20260825 + c − 1` and the Stage 2B seed in `H_SEEDS`.
`--s1_epochs`, `--a2_steps` and `--h_steps` shorten the corresponding stages for smoke tests.

### Inputs

```
<root>/
├── breast/
│   ├── stage1_data.pt
│   ├── teacher.pt
│   ├── programs.json
│   ├── panel.txt
│   ├── registry.pt
│   ├── registry_2a.pt
│   ├── labels_2a.pt
│   ├── survival.pt
│   └── stage2a/            cell_fraction_measurement_aware_seed{20260821,20260822,20260823}.pt
├── liver/
│   ├── stage1_data.pt
│   ├── teacher.pt
│   ├── programs.json
│   ├── panel.txt
│   ├── registry.pt
│   └── stage2a/            cell_fraction_measurement_aware_seed{…}.pt (liver models)
└── pancreas/
    ├── stage1_data.pt
    ├── teacher.pt
    ├── programs.json
    ├── panel.txt
    ├── registry.pt
    └── stage2a/            cell_fraction_measurement_aware_seed{…}.pt (pancreas models)
```

| File | Key in `ORGANS` | Contents |
|---|---|---|
| `stage1_data.pt` | `stage1_data` | single-cell counts, cell types, donors, donor split and gene names (liver and pancreas: also TCGA survival) |
| `teacher.pt` | `teacher` | frozen scGPT embedding of every reference cell, computed on the selected 3,000-gene panel |
| `programs.json` | `programs` | program dictionary `{program: [genes]}` restricted to the selected panel |
| `panel.txt` | `panel` | the selected 3,000-gene panel, one gene per line (PACE decoder genes and CCSE input genes) |
| `registry.pt` | `registry` | TCGA bulk expression (standardised; breast log2(1 + CPM), liver and pancreas log2(count + 1)), patient split, Stage 2A labels and pseudo-bulk standardisation statistics |
| `registry_2a.pt` | `registry_2a` | breast only: the registry used to train Stage 2A (liver and pancreas use `registry.pt`) |
| `labels_2a.pt` | `labels_2a` | breast only: Stage 2A labels re-matched to the bulk rows by exact TCGA patient ID |
| `survival.pt` | `survival` | breast only: TCGA endpoints (liver and pancreas survival is stored in `stage1_data.pt`) |
| `stage2a/` | `a_dir` | the three organ-specific Stage 2A checkpoints (seeds 20260821–20260823) reused by default, or produced with `--stage 2a` |

These inputs are not included in this repository.

### Outputs (in `--out`)

| File | Stage |
|---|---|
| `stage1.pt`, `zprime.pt` | PACE weights and the latent state of every reference cell |
| `cell_fraction_measurement_aware_seed{seed}.pt` | CCE checkpoint (`--stage 2a`) |
| `calibration.json`, `bags_{trainsel,val,test}.pt` | axis calibration and fixed pseudo-bulk bags |
| `h.pt` | CCSE checkpoint |
| `stage3.pt` | `α`, `W`, composition `π`, states `M` and risk of every TCGA patient |

## Data

The single-cell references (GEO GSE176078, GSE151530 and GSE202051), TCGA bulk RNA-seq (TCGA-BRCA,
TCGA-LIHC and TCGA-PAAD) and the external cohorts used in the paper are publicly available; accession
numbers are listed in the manuscript.
