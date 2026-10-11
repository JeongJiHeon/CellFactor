"""CellFactor training.

One chain = Stage 1' -> (Stage 2A, shared by all chains) -> calibration + bags -> H -> Stage 3.

  python train.py --organ liver --chain 1 --stage all --out runs/liver_c1 --device cuda:0
  python train.py --organ liver --stage 2a --seed 20260821 --out runs/liver_2a      # optional; default reuses the paper's 2A

stages
  stage1  ProgramAnchoredNBVAE on the single-cell reference -> z' for every cell        out/stage1.pt, out/zprime.pt
  2a      CompositionEstimator on TCGA bulk (one seed per call; CellFactor uses 3)       out/cell_fraction_measurement_aware_seed{seed}.pt
          (pass that directory as --a_dir to use your own 2A in the later stages)
  bags    axis calibration (s, rho, w) + fixed pseudobulk bags (trainsel / val / test)  out/calibration.json, out/bags_{role}.pt
  h       StateEstimator on fresh paired pseudobulk bags each step                      out/h.pt
  stage3  ridge Cox readout W and alpha (all organs) on TCGA training patients           out/stage3.pt
  all     stage1 -> bags -> h -> stage3   (2A is read from --a_dir, default = the paper's 2A checkpoints)

Inputs (prepared upstream, paths in ORGANS, relative to ROOT):
  stage1_data  single-cell counts / cell type / donor / donor split / gene names, TCGA survival (liver, pancreas)
  teacher      scGPT embedding of every reference cell computed on the selected 3,000-gene panel
  programs     program dictionary {program: [genes]} restricted to the selected panel
  panel        selected 3,000-gene panel (Stage 1' decoder genes and H input genes)
  registry     TCGA bulk (bulk_x, standardised log2 CPM), patient split, 2A labels, pseudobulk standardisation
               (breast 2A labels: labels_2a, the same TCGA labels re-matched to bulk rows by exact patient ID)

Survival labels are used only in Stage 3. Seeds follow the paper: chain c uses Stage 1' seed 20260825+c-1 and
the H seed in H_SEEDS. The paper's CellFactor is the cohort-wise z-score mean of chain risks
(breast and liver c1-c12, pancreas c1-c6).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import (CellFactorRisk, CompositionEnsemble, CompositionEstimator, ProgramAnchoredNBVAE, StateEstimator,
                   calibrate_state_axes, coarse_composition_loss, conditional_coral, cox_nll, fine_composition,
                   load_composition_estimator, log_normalize, weighted_delta_loss, weighted_state_loss)

ROOT = Path("/workspace/jeongjiheon/CP_PCH_final")
_LIVER = "results/stage2_redesign/run_20260823_liver_online_domain_adaptive_h_v1"
_PANC = "sandbox6/results/stage2_redesign/run_20260823_pancreas_online_domain_adaptive_h_v1"
ORGANS = {
    "breast": dict(
        stage1_data="results/panel16k_g176/G176_STAGE1_DATA.pt",
        teacher="weights/hvg_run1a/breast/teacher/G176_cand3k_TEACHER.pt",
        programs="results/tcga_panel/design/dicts/breast_cand3k_PROGRAM_SETS_MIN.json",
        panel="results/tcga_panel/breast/cand3k.txt",
        registry="results/stage2_redesign/run_20260830_g176v2/00_registry/prepared_v2_all26.pt",
        registry_2a="results/stage2_redesign/run_20260828_p16k/00_registry/prepared_v2.pt",
        labels_2a="results/stage2_redesign/run_20260828_p16k_stage2a_100k/BREAST_2A_SUPERVISION.pt",   # labels re-matched by exact TCGA IDs
        a_dir="results/stage2_redesign/run_20260828_p16k_stage2a_100k/stage2a_composition/breast/checkpoints",
        survival="results/w_exp/cache/TCGA_endpoints.pt",
        tumor="Cancer Epithelial", immune=["B-cells", "Myeloid", "Plasmablasts", "T-cells"],
        s1_epochs=90, h_steps=4000,
        bag_seeds={"trainsel": 20279555, "val": 20271701, "test": 20271702}, bag_n={"trainsel": 300, "val": 1200, "test": 400},
        calib_from="trainsel",      # s, rho, w from the trainsel bags
        z_clip=12.0,                # z' = clamp(z / s, +-12)
        h_monitor="val", h_keep="last",   # H checkpoint = state after the last step
        real_rows="train",          # CORAL real bulk = all TCGA training patients
        alpha_free=True,            # Stage 3: alpha free (as liver/pancreas; the paper model since 2026-10-11)
        a2_val_dropout=False,
        infer_batch=512),           # bulk inference batch size before Stage 3 (as in the original runs; Stage 3 amplifies ULP differences)
    "liver": dict(
        stage1_data="results/liver/LIVER_STAGE1_DATA.pt",
        teacher="weights/hvg_run1a/liver/teacher/LIVER_cand3k_TEACHER.pt",
        programs="results/tcga_panel/design/dicts/liver_cand3k_PROGRAM_SETS_MIN.json",
        panel="results/tcga_panel/liver/cand3k.txt",
        registry=f"{_LIVER}/00_registry/prepared_liver_preprocess_v2_seed20260821.pt",
        registry_2a=f"{_LIVER}/00_registry/prepared_liver_preprocess_v2_seed20260821.pt",
        a_dir=f"{_LIVER}/stage2a_composition/liver/checkpoints",
        survival=None,              # TCGA survival stored in stage1_data
        tumor="Malignant cells", immune=["B cells", "T cells", "TAMs"],
        s1_epochs=30, h_steps=4000,
        bag_seeds={"trainsel": 20279555, "val": 20270932, "test": 20279101}, bag_n={"trainsel": 300, "val": 300, "test": 400},
        calib_from=("train", 20279777, 300),   # a separate 300-bag draw from training donors
        z_clip=None, h_monitor="trainsel", h_keep="best", real_rows="fit", alpha_free=True, a2_val_dropout=True, infer_batch=128),
    "pancreas": dict(
        stage1_data="sandbox6/results/pancreas/PANC_STAGE1_DATA.pt",
        teacher="weights/hvg_run1a/pancreas/teacher/PANC_cand3k_TEACHER.pt",
        programs="results/tcga_panel/design/dicts/pancreas_cand3k_PROGRAM_SETS_MIN.json",
        panel="results/tcga_panel/pancreas/cand3k.txt",
        registry=f"{_PANC}/00_registry/prepared_pancreas_preprocess_v2_seed20260821.pt",
        registry_2a=f"{_PANC}/00_registry/prepared_pancreas_preprocess_v2_seed20260821.pt",
        a_dir=f"{_PANC}/stage2a_composition/pancreas/checkpoints",
        survival=None,
        tumor="Epithelial", immune=["Immune"], n_immune_head=3,
        s1_epochs=30, h_steps=10000,
        bag_seeds={"trainsel": 20279555, "val": 20270932, "test": 20279101}, bag_n={"trainsel": 300, "val": 300, "test": 400},
        calib_from=("train", 20279777, 300),
        z_clip=None, h_monitor="trainsel", h_keep="best", real_rows="fit", alpha_free=True, a2_val_dropout=True, infer_batch=128),
}
A_SEEDS = [20260821, 20260822, 20260823]
S1_SEED = {c: 20260825 + c - 1 for c in range(1, 13)}
H_SEEDS = {"breast": {c: 20271700 + c for c in range(1, 13)}, "liver": {c: 20271700 + c for c in range(1, 13)},
           "pancreas": dict(zip(range(1, 7), [20271701, 20271702, 20271705, 20271706, 20271707, 20271708]))}

S1 = dict(free_dims=4, donor_rank=2, batch_size=256, lr=3e-4, beta_kl=0.1, free_bits=0.1, anchor_w=0.1, l1_free=1e-2, l2_free=0.0, l2_prog=1e-5)
A2 = dict(steps=100_000, batch_size=64, width=256, bottleneck=96, blocks=2, dropout=0.05, lr=3e-4, weight_decay=1e-4,
          gradient_clip=5.0, validation_every=50, teacher_weight=0.5)
H = dict(batch_size=32, real_batch_size=32, lr=3e-4, weight_decay=1e-4, gradient_clip=5.0, validation_every=100,
         n_cells=1024, pair_w=0.25, coral_w=0.05, coral_warmup=(500, 1500))
S3 = dict(steps=600, lr=1e-2, weight_decay=1e-4, ridge=0.05)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load(p, mmap=False):
    p = ROOT / p
    try: return torch.load(p, map_location="cpu", weights_only=False, mmap=mmap)
    except Exception: return torch.load(p, map_location="cpu", weights_only=False)


def panel_genes(cfg):
    return [g.strip() for g in open(ROOT / cfg["panel"]) if g.strip()]


def index_of(genes, names):
    gi = {g: i for i, g in enumerate(names)}
    idx = [gi[g] for g in genes if g in gi]
    assert len(idx) == len(genes), f"{len(genes) - len(idx)} panel genes missing"
    return torch.as_tensor(idx, dtype=torch.long)


def tcga_fit_val_split(bundle):
    """2A internal split of TCGA training patients: 85% fit / 15% validation (fixed seed); held-out never used."""
    allowed = np.asarray(bundle["bulk_split"]["train"], int); heldout = np.asarray(bundle["bulk_split"]["heldout"], int)
    shuffled = np.random.default_rng(20260821 + 101).permutation(allowed)
    n_val = max(1, int(round(.15 * len(shuffled))))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val]), heldout


def type_mapping(cfg, bundle):
    names = list(bundle["cell_type_names"])
    tumor = names.index(cfg["tumor"]); immune = [names.index(x) for x in cfg["immune"]]
    other = [i for i in range(len(names)) if i != tumor and i not in immune]
    prior = bundle["bags"]["train"]["composition"][:, other].mean(0)
    return {"tumor_index": tumor, "immune_indices": immune, "other_indices": other, "other_prior": prior / prior.sum().clamp_min(1e-12),
            "coarse_groups": ((tumor,), tuple(immune), tuple(other))}


def load_a(cfg, a_dir, device):
    a_dir = Path(a_dir) if a_dir else ROOT / cfg["a_dir"]
    return CompositionEnsemble([load_composition_estimator(a_dir / f"cell_fraction_measurement_aware_seed{s}.pt", device) for s in A_SEEDS]).to(device).eval()


# ============================================================================= Stage 1'
def train_stage1(cfg, chain, out, device):
    seed = S1_SEED[chain]; set_seed(seed); t0 = time.time()
    D = load(cfg["stage1_data"])
    counts, types, donors = D["sc_counts"], D["sc_cell_type"], D["sc_donor"]
    if counts.dtype == torch.uint16: counts = counts.view(torch.int16)
    genes = [str(g) for g in D["gene_names"]]; gi = {g: i for i, g in enumerate(genes)}
    dec_idx = index_of(panel_genes(cfg), genes); dec_genes = [genes[i] for i in dec_idx.tolist()]; dgi = {g: i for i, g in enumerate(dec_genes)}
    teacher = torch.load(ROOT / cfg["teacher"], map_location="cpu", weights_only=False)["teacher_emb"].float(); assert len(teacher) == len(counts)
    progs = json.load(open(ROOT / cfg["programs"])); names = sorted(progs); K = len(names)
    mask = torch.zeros(len(dec_genes), K)
    for k, nm in enumerate(names):
        for g in progs[nm]: mask[dgi[g], k] = 1.0
    epochs = cfg["s1_epochs"]
    model = ProgramAnchoredNBVAE(len(dec_genes), len(D["cell_type_names"]), len(D["donor_names"]), mask, S1["free_dims"], S1["donor_rank"], teacher.shape[1]).to(device)
    # anchor target: mean log1p-CP10k (full-panel library) of each program's genes, standardised on training cells
    pix = [torch.as_tensor([gi[g] for g in progs[nm]]) for nm in names]
    score = torch.empty(len(counts), K)
    for s in range(0, len(counts), 20000):
        ln = log_normalize(counts[s:s + 20000].float())
        for k, ix in enumerate(pix): score[s:s + len(ln), k] = ln[:, ix].mean(1)
    tr_d = torch.tensor(D["donor_split"]["train"]); tr_i = torch.where(torch.isin(donors, tr_d))[0]
    score = ((score - score[tr_i].mean(0)) / score[tr_i].std(0).clamp_min(1e-6)).clamp(-8., 8.)
    model.set_teacher_center(teacher[tr_i].mean(0))
    dcounts = counts[:, dec_idx].contiguous(); del counts
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=S1["lr"], weight_decay=1e-4); last_good = None
    for ep in range(epochs):
        model.train(); beta = S1["beta_kl"] * min(1.0, (ep + 1) / max(2, epochs // 4)); tot, n = 0.0, 0
        for local in torch.randperm(len(tr_i), generator=gen).split(S1["batch_size"]):
            idx = tr_i[local]; opt.zero_grad(set_to_none=True)
            loss, _ = model.loss(teacher[idx].to(device), dcounts[idx].float().to(device), types[idx].to(device), donors[idx].to(device),
                                 anchor=score[idx].to(device), beta=beta, free_bits=S1["free_bits"], anchor_w=S1["anchor_w"],
                                 l1_free=S1["l1_free"], l2_free=S1["l2_free"], l2_prog=S1["l2_prog"])
            if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
            loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                if last_good is not None: model.load_state_dict(last_good)
                continue
            opt.step(); tot += float(loss.detach()) * len(idx); n += len(idx)
        if n and np.isfinite(tot / n): last_good = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[stage1 c{chain}] epoch {ep + 1}/{epochs} loss {tot / max(n, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval(); zs = []
    with torch.no_grad():
        for s in range(0, len(dcounts), 512):
            zs.append(model.encode(teacher[s:s + 512].to(device), dcounts[s:s + 512].float().to(device))[0].cpu())
    z = torch.cat(zs)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": {**S1, "epochs": epochs, "seed": seed},
                "program_names": names, "decoder_gene_idx": dec_idx}, out / "stage1.pt")
    torch.save({"z": z, "program_names": names, "free_dims": S1["free_dims"], "cell_type_names": D["cell_type_names"]}, out / "zprime.pt")
    print(f"[stage1 c{chain}] z' {tuple(z.shape)} -> {out / 'zprime.pt'}", flush=True)


# ============================================================================= Stage 2A
def train_stage2a(cfg, seed, out, device):
    bundle = load(cfg["registry_2a"]); names = list(bundle["cell_type_names"])
    sup = load(cfg["labels_2a"]) if cfg.get("labels_2a") else bundle["supervision"]
    fit, val, heldout = tcga_fit_val_split(bundle)
    t_, i_ = np.asarray(sup["cellfrac"]), np.asarray(sup["leukocyte"])
    conflict = np.isfinite(t_) & np.isfinite(i_) & (t_ + i_ > 1 + 1e-6)          # incompatible paired labels are excluded
    experimental = (np.isfinite(t_) | np.isfinite(i_)) & ~conflict
    teacher = np.asarray(sup["teacher_relative"], float)[:, [names.index(x) for x in cfg["immune"]]]
    fit_exp = fit[experimental[fit]]; fit_teacher = fit[np.isfinite(teacher).all(1)[fit]]
    set_seed(seed)
    model = CompositionEstimator(bundle["bulk_x"].shape[1], cfg.get("n_immune_head", len(cfg["immune"])),
                                 A2["width"], A2["bottleneck"], A2["blocks"], A2["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=A2["lr"], weight_decay=A2["weight_decay"])
    rng = np.random.default_rng(seed + 1001); x = bundle["bulk_x"]; best, best_step, best_state = float("inf"), -1, None
    for step in range(1, A2["steps"] + 1):
        model.train(); idx = rng.choice(fit_exp, size=min(A2["batch_size"], len(fit_exp)), replace=True)
        pred, _ = model(x[idx].to(device))
        loss = coarse_composition_loss(pred, torch.as_tensor(t_[idx], device=device), torch.as_tensor(i_[idx], device=device))
        if len(fit_teacher):
            ti = rng.choice(fit_teacher, size=min(A2["batch_size"], len(fit_teacher)), replace=True)
            _, rel = model(x[ti].to(device)); target = torch.as_tensor(teacher[ti], device=device)
            loss = loss + A2["teacher_weight"] * (-(target * rel.clamp_min(1e-8).log()).sum(1).mean())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), A2["gradient_clip"]); opt.step()
        if step == 1 or step % A2["validation_every"] == 0 or step == A2["steps"]:
            # selection: mean validation MAE of the experimental tumour and immune fractions.
            # As in the original runs, liver/pancreas validate without model.eval() (dropout on); breast validates in eval mode.
            if not cfg["a2_val_dropout"]: model.eval()
            with torch.no_grad(): coarse = torch.cat([model(x[val][s:s + 256].to(device))[0].cpu() for s in range(0, len(val), 256)]).numpy()
            maes = []
            for col, y in ((0, t_[val]), (1, i_[val])):
                ok = np.isfinite(y) & ~conflict[val]
                if ok.any(): maes.append(np.abs(coarse[ok, col] - y[ok]).mean())
            crit = float(np.mean(maes))
            if crit < best: best, best_step, best_state = crit, step, copy.deepcopy(model.state_dict())
            if step == 1 or step % 5000 == 0: print(f"[2A {seed}] {step}/{A2['steps']} val MAE {crit:.5f} best {best:.5f}@{best_step}", flush=True)
    torch.save({"architecture": "CompositionEstimator", "state_dict": {k: v.cpu() for k, v in best_state.items()},
                "model_config": {k: A2[k] for k in ("width", "bottleneck", "blocks", "dropout")}, "seed": seed,
                "steps_completed": A2["steps"], "best_step": best_step, "best_validation_experimental_mae": best},
               out / f"cell_fraction_measurement_aware_seed{seed}.pt")


# ============================================================================= pseudobulk bags
class BagSampler:
    """Donor-pure pseudobulk bags of n_cells cells with the donor's own type composition.
    Cells are drawn per type to match the composition exactly; draw B is a second, independent draw with the same
    donor and allocation. Expression = log1p CPM of summed counts, then the registry's pseudobulk standardisation."""

    def __init__(self, data, z, donor_pool, n_types, bundle, n_cells=1024, seed=0, device="cpu"):
        counts = data["sc_counts"]
        if counts.dtype == torch.uint16: counts = counts.view(torch.int16)
        self.dev = torch.device(device); self.counts = counts.to(self.dev)
        self.cell_type, self.z, self.n_types, self.n_cells = data["sc_cell_type"], z.float(), n_types, n_cells
        donor = data["sc_donor"]
        pool = [int(d) for d in donor_pool]
        cells = {d: torch.where(donor == d)[0] for d in pool}
        self.pool = [d for d in pool if len(cells[d]) >= 8]
        self.type_cells = {d: [cells[d][self.cell_type[cells[d]] == c] for c in range(n_types)] for d in self.pool}
        self.comp = {}
        for d in self.pool:
            fc = torch.tensor([len(i) for i in self.type_cells[d]], dtype=torch.float32); self.comp[d] = fc / fc.sum().clamp_min(1)
        p = bundle["preprocessing"]; self.mean, self.sd, self.clamp = p["pseudo"]["mean"], p["pseudo"]["sd"], float(p["clamp"])
        self.gen = torch.Generator().manual_seed(seed)

    def _allocation(self, prob):
        raw = prob / prob.sum().clamp_min(1e-12) * self.n_cells; count = torch.floor(raw).long(); rem = self.n_cells - int(count.sum())
        if rem: count[torch.argsort(raw - count.float(), descending=True)[:rem]] += 1
        return count

    def _draw(self, d, alloc):
        sel = []
        for c, k in enumerate(alloc.tolist()):
            if k == 0: continue
            pool = self.type_cells[d][c]
            sel.append(pool[torch.randperm(len(pool), generator=self.gen)[:k] if k <= len(pool) else torch.randint(len(pool), (k,), generator=self.gen)])
        idx = torch.cat(sel); return idx[torch.randperm(len(idx), generator=self.gen)]

    def _expression(self, idx):
        s = self.counts[idx.to(self.dev)].sum(0, dtype=torch.float32)
        ln = torch.log1p(s / s.sum().clamp_min(1) * 1e6).cpu()
        return ((ln / math.log(2.0) - self.mean) / self.sd).clamp(-self.clamp, self.clamp)

    def _state(self, idx):
        t = self.cell_type[idx]; M = torch.zeros(self.n_types, self.z.shape[1]); present = torch.zeros(self.n_types)
        cnt = torch.bincount(t, minlength=self.n_types); M.index_add_(0, t.long(), self.z[idx]); present[cnt > 0] = 1
        return M / cnt.clamp(min=1).unsqueeze(1).to(M), present

    def sample(self, n):
        out = {k: [] for k in ("expression_a", "expression_b", "target_a", "target_b", "present_a", "present_b", "donor")}
        for _ in range(n):
            torch.rand((), generator=self.gen)                       # natural-vs-counterfactual draw of the original sampler (always natural)
            d = self.pool[int(torch.randint(len(self.pool), (1,), generator=self.gen))]
            alloc = self._allocation(self.comp[d].clone())
            ia, ib = self._draw(d, alloc), self._draw(d, alloc)
            for v, idx in (("a", ia), ("b", ib)):
                out[f"expression_{v}"].append(self._expression(idx)); M, p = self._state(idx); out[f"target_{v}"].append(M); out[f"present_{v}"].append(p)
            out["donor"].append(d)
        return {k: torch.tensor(v) if k == "donor" else torch.stack(v) for k, v in out.items()}


def make_bags(cfg, chain, out, device, a_dir):
    bundle = load(cfg["registry"]); D = load(cfg["stage1_data"], mmap=True); Cn = len(bundle["cell_type_names"])
    z_raw = torch.load(out / "zprime.pt", map_location="cpu", weights_only=False)["z"].float(); assert len(z_raw) == len(D["sc_counts"])
    sampler = lambda z, role, seed: BagSampler(D, z, bundle["donor_split"]["train" if role in ("train", "trainsel") else role], Cn, bundle, H["n_cells"], seed, device)
    # calibration on the unscaled z'
    if cfg["calib_from"] == "trainsel": b = sampler(z_raw, "trainsel", cfg["bag_seeds"]["trainsel"]).sample(cfg["bag_n"]["trainsel"])
    else: role, seed, n = cfg["calib_from"]; b = sampler(z_raw, role, seed).sample(n)
    s, rho, w = calibrate_state_axes(b["target_a"], b["present_a"], b["target_b"], b["present_b"])
    json.dump({"s": s.tolist(), "rho": rho.tolist(), "w": w.tolist()}, open(out / "calibration.json", "w"))
    zc = z_raw / s[None]
    if cfg["z_clip"]: zc = zc.clamp(-cfg["z_clip"], cfg["z_clip"])
    a = load_a(cfg, a_dir, device)
    for role in ("trainsel", "val", "test"):
        b = sampler(zc, role, cfg["bag_seeds"][role]).sample(cfg["bag_n"][role])
        with torch.no_grad(): b["coarse_a"] = a(b["expression_a"].to(device))[0].cpu(); b["coarse_b"] = a(b["expression_b"].to(device))[0].cpu()
        torch.save(b, out / f"bags_{role}.pt")
    print(f"[bags c{chain}] s {float(s.min()):.3f}-{float(s.max()):.3f} · rho median {float(rho.median()):.3f} -> {out}", flush=True)


# ============================================================================= Stage 2D (H)
def train_h(cfg, organ, chain, out, device, a_dir, steps=None):
    seed = H_SEEDS[organ][chain]; steps = steps or cfg["h_steps"]; t0 = time.time()
    bundle = load(cfg["registry"]); D = load(cfg["stage1_data"], mmap=True); Cn = len(bundle["cell_type_names"])
    G = index_of(panel_genes(cfg), [str(g) for g in bundle["gene_names"]])
    cal = json.load(open(out / "calibration.json")); s, w = torch.tensor(cal["s"]), torch.tensor(cal["w"]).to(device)
    zc = torch.load(out / "zprime.pt", map_location="cpu", weights_only=False)["z"].float() / s[None]
    if cfg["z_clip"]: zc = zc.clamp(-cfg["z_clip"], cfg["z_clip"])
    a = load_a(cfg, a_dir, device)
    source = BagSampler(D, zc, bundle["donor_split"]["train"], Cn, bundle, H["n_cells"], seed + 30000, device)
    set_seed(seed)
    m = StateEstimator(len(G), Cn, 3, zc.shape[1], 256, 128, 2).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=H["lr"], weight_decay=H["weight_decay"])
    mon = torch.load(out / f"bags_{cfg['h_monitor']}.pt", map_location="cpu", weights_only=False)
    rows = np.asarray(bundle["bulk_split"]["train"]) if cfg["real_rows"] == "train" else tcga_fit_val_split(bundle)[0]
    real = bundle["bulk_x"][rows].float(); rg = torch.Generator().manual_seed(seed + 40000)
    best, best_step, best_state = float("inf"), -1, None; good = copy.deepcopy(m.state_dict())
    for step in range(1, steps + 1):
        b = source.sample(H["batch_size"])
        xa, xb = b["expression_a"].to(device), b["expression_b"].to(device)
        ta, tb, ma, mb = (b[k].to(device) for k in ("target_a", "target_b", "present_a", "present_b"))
        with torch.no_grad(): qa, qb = a(xa)[0], a(xb)[0]
        xr = real[torch.randint(len(real), (H["real_batch_size"],), generator=rg)].to(device)
        with torch.no_grad(): qr = a(xr)[0]
        m.train(); fa, fb = m.encode(xa[:, G]), m.encode(xb[:, G]); pa, pb = m.decode_state(fa, qa), m.decode_state(fb, qb)
        sup = .5 * (weighted_state_loss(pa, ta, ma, w) + weighted_state_loss(pb, tb, mb, w))   # both draws supervised by their own targets
        pair = weighted_delta_loss(pa, pb, ta, tb, ma, mb, w)
        align = conditional_coral(fa, qa, m.encode(xr[:, G]), qr)
        s0, s1 = H["coral_warmup"]; warm = max(0., min(1., (step - s0) / (s1 - s0)))
        loss = sup + H["pair_w"] * pair + warm * H["coral_w"] * align
        if not torch.isfinite(loss):
            m.load_state_dict(good); opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), H["gradient_clip"])
        if not torch.isfinite(gn):
            m.load_state_dict(good); opt.zero_grad(set_to_none=True); continue
        opt.step()
        if step % 100 == 0 and all(torch.isfinite(p).all() for p in m.parameters()): good = copy.deepcopy(m.state_dict())
        if step == 1 or step % H["validation_every"] == 0:
            m.eval()
            with torch.no_grad():
                pred = torch.cat([m(mon["expression_a"][i:i + 128][:, G].to(device), mon["coarse_a"][i:i + 128].to(device)) for i in range(0, len(mon["donor"]), 128)])
                v = float(weighted_state_loss(pred, mon["target_a"].to(device), mon["present_a"].to(device), w))
            if v < best: best, best_step, best_state = v, step, copy.deepcopy(m.state_dict())
        if step == 1 or step % 500 == 0: print(f"[H {organ} c{chain}] {step}/{steps} monitor wMSE best {best:.5f}@{best_step} ({time.time() - t0:.0f}s)", flush=True)
    state = m.state_dict() if cfg["h_keep"] == "last" else best_state
    torch.save({"architecture": "StateEstimator", "state_dict": {k: v.cpu() for k, v in state.items()},
                "model_config": {"coarse_composition_dim": 3, "latent": zc.shape[1], "width": 256, "bottleneck": 128, "blocks": 2, "fine_cell_types": Cn},
                "seed": seed, "steps": steps, "kept": cfg["h_keep"], "best_step": best_step, "gene_idx": G, "s": s, "w": w.cpu()}, out / "h.pt")
    print(f"[H {organ} c{chain}] saved {out / 'h.pt'}", flush=True)


# ============================================================================= Stage 3
def fit_stage3(pi, M, time_, event, alpha_free, device):
    set_seed(0)
    m = CellFactorRisk(pi.shape[1], M.shape[2]).to(device)
    if not alpha_free: m.alpha.requires_grad_(False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=S3["lr"], weight_decay=S3["weight_decay"])
    pi, M, time_, event = pi.to(device), M.to(device), time_.to(device), event.to(device)
    for _ in range(S3["steps"]):
        loss = cox_nll(m.risk(pi, M), time_, event) + S3["ridge"] * m.W.square().sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m.eval()


def harrell_c(t, e, r):
    from lifelines.utils import concordance_index
    ok = np.isfinite(t) & (t > 0) & np.isfinite(e) & np.isfinite(r)
    return float(concordance_index(t[ok], -r[ok], e[ok]))


def train_stage3(cfg, organ, chain, out, device, a_dir):
    bundle = load(cfg["registry"]); Cn = len(bundle["cell_type_names"]); mapping = type_mapping(cfg, bundle)
    a = load_a(cfg, a_dir, device); ck = torch.load(out / "h.pt", map_location="cpu", weights_only=False)
    h = StateEstimator(len(ck["gene_idx"]), Cn, **{k: v for k, v in ck["model_config"].items() if k != "fine_cell_types"}).to(device)
    h.load_state_dict(ck["state_dict"]); h.eval(); G = ck["gene_idx"]
    x = bundle["bulk_x"].float(); P, Ms = [], []; bs = cfg["infer_batch"]
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs].to(device); q, rel = a(xb); P.append(fine_composition(q, rel, mapping, Cn).cpu()); Ms.append(h(xb[:, G], q).cpu())
    pi, M = torch.cat(P), torch.cat(Ms)
    train = np.asarray(bundle["bulk_split"]["train"]); held = np.asarray(bundle["bulk_split"]["heldout"])
    if cfg["survival"]:
        ep = load(cfg["survival"])["OS"]; t_all, e_all = np.asarray(ep["time"], float), np.asarray(ep["event"], float)
        train = train[np.isfinite(t_all[train]) & (t_all[train] > 0) & np.isfinite(e_all[train])]
    else:
        D = load(cfg["stage1_data"], mmap=True); t_all, e_all = np.asarray(D["bulk_survival_time"], float), np.asarray(D["bulk_survival_event"], float)
    ti = torch.as_tensor(train)
    m = fit_stage3(pi[ti], M[ti], torch.tensor(t_all[train], dtype=torch.float32), torch.tensor(e_all[train], dtype=torch.float32), cfg["alpha_free"], device)
    with torch.no_grad(): risk = m.risk(pi.to(device), M.to(device)).cpu().numpy()
    torch.save({"W": m.W.detach().cpu(), "alpha": m.alpha.detach().cpu(), "pi": pi, "M": M, "risk": risk, "train_idx": train, "heldout_idx": held,
                "cell_types": list(bundle["cell_type_names"]), "mapping": mapping}, out / "stage3.pt")
    print(f"[stage3 {organ} c{chain}] TCGA held-out OS C-index {harrell_c(t_all[held], e_all[held], risk[held]):.3f} -> {out / 'stage3.pt'}", flush=True)


# ============================================================================= main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--organ", required=True, choices=list(ORGANS))
    p.add_argument("--chain", type=int, default=1)
    p.add_argument("--stage", required=True, choices=["stage1", "2a", "bags", "h", "stage3", "all"])
    p.add_argument("--out", required=True, help="output directory for this chain (or for 2A checkpoints)")
    p.add_argument("--seed", type=int, help="2A seed (one of 20260821-20260823)")
    p.add_argument("--a_dir", help="directory of the three 2A checkpoints (default: the paper's)")
    p.add_argument("--h_steps", type=int, help="override H steps (smoke tests)")
    p.add_argument("--s1_epochs", type=int, help="override Stage 1' epochs (smoke tests)")
    p.add_argument("--a2_steps", type=int, help="override 2A steps (smoke tests)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg, out, dev = ORGANS[args.organ], Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    if args.s1_epochs: cfg = {**cfg, "s1_epochs": args.s1_epochs}
    if args.a2_steps: A2["steps"] = args.a2_steps
    if args.stage == "2a": train_stage2a(cfg, args.seed, out, dev); return
    if args.stage in ("stage1", "all"): train_stage1(cfg, args.chain, out, dev)
    if args.stage in ("bags", "all"): make_bags(cfg, args.chain, out, dev, args.a_dir)
    if args.stage in ("h", "all"): train_h(cfg, args.organ, args.chain, out, dev, args.a_dir, args.h_steps)
    if args.stage in ("stage3", "all"): train_stage3(cfg, args.organ, args.chain, out, dev, args.a_dir)


if __name__ == "__main__":
    main()
