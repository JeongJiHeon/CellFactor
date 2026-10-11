"""CellFactor model definitions.

CellFactor maps a bulk tumour expression profile x to a patient risk score

    r(x) = sum_c  pi_c(x) * ( alpha_c + W_c . M_c(x) )

  pi(x)   cell-type composition       Stage 2A  CompositionEstimator (+ fine_composition)
  M_c(x)  type-specific mean state    Stage 2D  StateEstimator, in the Stage 1' latent space
  W, alpha survival readout           Stage 3   CellFactorRisk (ridge Cox); alpha free in all organs

The Stage 1' latent space comes from ProgramAnchoredNBVAE: a negative-binomial VAE on
single cells whose first K latent axes are tied to gene programs (decoder mask + anchor
loss) and whose last F axes are free.

Parameter names match the original training scripts, so their checkpoints load directly
(see `load_composition_estimator`, `load_state_estimator`).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ----------------------------------------------------------------------------- building blocks
def nb_nll(x: Tensor, mu: Tensor, theta: Tensor) -> Tensor:
    """Negative-binomial NLL, excluding constants independent of parameters."""
    theta = theta.clamp(1e-4, 1e4)
    mu = mu.clamp(1e-6, 1e8)
    return -(torch.lgamma(x + theta) - torch.lgamma(theta)
             + theta * (torch.log(theta) - torch.log(theta + mu))
             + x * (torch.log(mu) - torch.log(theta + mu)))


def log_normalize(counts: Tensor, scale: float = 1e4) -> Tensor:
    return torch.log1p(counts / counts.sum(1, keepdim=True).clamp_min(1) * scale)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(width * 2, width))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


# ----------------------------------------------------------------------------- Stage 1'
class ProgramAnchoredNBVAE(nn.Module):
    """Single-cell NB-VAE with program-anchored latent axes.

    encoder: scGPT cell embedding (centred on training cells) -> q(z), z = [z_prog (K) | z_free (F)]
    decoder: softmax(gene_bias + type_intercept[c] + z_prog @ (D*mask)^T + z_free @ E^T + donor term) * library
    mask (genes x K) restricts program axis k to the genes of program k.
    """

    def __init__(self, genes: int, cell_types: int, donors: int, mask: Tensor, free_dims: int = 4,
                 donor_rank: int = 2, teacher_dim: int = 512):
        super().__init__()
        K = mask.shape[1]
        self.genes, self.K, self.F = genes, K, free_dims
        self.latent = K + free_dims
        self.register_buffer("mask", mask)
        self.register_buffer("teacher_center", torch.zeros(teacher_dim))
        self.posterior = nn.Sequential(nn.LayerNorm(teacher_dim), nn.Linear(teacher_dim, 512), nn.SiLU(), nn.Dropout(0.10),
                                       nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU())
        self.z_mean = nn.Linear(256, self.latent)
        self.z_logvar = nn.Linear(256, self.latent)
        self.D = nn.Parameter(torch.randn(genes, K) * 0.01)
        self.E = nn.Parameter(torch.randn(genes, free_dims) * 0.01)
        self.type_intercept = nn.Embedding(cell_types, genes)
        nn.init.zeros_(self.type_intercept.weight)
        self.gene_bias = nn.Parameter(torch.zeros(genes))
        self.log_theta = nn.Parameter(torch.zeros(genes))
        self.donor_loadings = nn.Parameter(torch.zeros(genes, donor_rank))
        self.donor_factors = nn.Embedding(donors, donor_rank)
        nn.init.normal_(self.donor_factors.weight, std=0.02)

    def set_teacher_center(self, c: Tensor) -> None:
        self.teacher_center.copy_(c.to(self.teacher_center))

    def encode(self, teacher: Tensor, counts: Tensor):
        library = counts.sum(1).clamp_min(1.0)
        h = self.posterior(teacher.float() - self.teacher_center)
        return self.z_mean(h), self.z_logvar(h).clamp(-8, 5), library

    def decode(self, z: Tensor, cell_type: Tensor, library: Tensor, donor: Tensor | None = None) -> Tensor:
        zp, u = z[:, :self.K], z[:, self.K:]
        logits = self.gene_bias[None] + self.type_intercept(cell_type) + zp @ (self.D * self.mask).T + u @ self.E.T
        if donor is not None:
            fac = self.donor_factors.weight - self.donor_factors.weight.mean(0, keepdim=True)
            logits = logits + fac[donor] @ self.donor_loadings.T
        return F.softmax(logits, dim=1) * library[:, None]

    def forward(self, teacher, counts, cell_type, donor=None, sample=True):
        mean, logvar, library = self.encode(teacher, counts)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        return {"mean": mean, "logvar": logvar, "z": z, "mu": self.decode(z, cell_type, library, donor), "library": library}

    def loss(self, teacher, counts, cell_type, donor, anchor=None, beta=0.1, free_bits=0.1,
             anchor_w=0.1, l1_free=1e-2, l2_free=0.0, l2_prog=1e-5):
        """NB reconstruction + beta*KL(free bits) + weight penalties + anchor_w * program-anchor MSE.
        anchor: per-cell program scores (mean log1p-CP10k of each program's genes), standardised on training cells."""
        out = self(teacher, counts, cell_type, donor)
        recon = nb_nll(counts, out["mu"], F.softplus(self.log_theta)[None]).mean()
        kd = (-0.5 * (1 + out["logvar"] - out["mean"].square() - out["logvar"].exp())).mean(0)
        kl = kd.clamp_min(free_bits).sum() / self.genes
        reg = (l1_free * self.E.abs().mean() + l2_free * self.E.square().mean()
               + (l2_prog * (self.D * self.mask).square().mean() if self.K > 0 else 0.0)
               + 1e-4 * (self.donor_loadings.square().mean() + self.donor_factors.weight.square().mean()))
        anch = torch.zeros((), device=counts.device)
        if anchor is not None and self.K > 0:
            zp = out["mean"][:, :self.K]
            zn = ((zp - zp.mean(0)) / zp.std(0).clamp_min(1e-2)).clamp(-10., 10.)
            anch = (zn - anchor).square().mean()
        loss = recon + beta * kl + reg + anchor_w * anch
        return loss, {"nb_nll": float(recon.detach()), "kl": float(kl.detach()), "anchor": float(anch.detach()),
                      "active_dims": int((kd.detach() > free_bits).sum())}


# ----------------------------------------------------------------------------- Stage 2A
class CompositionEstimator(nn.Module):
    """Bulk -> coarse composition (Tumor, Immune, Other) and the relative split of immune types.

    Trained on TCGA bulk with experimental labels (ABSOLUTE tumour fraction, methylation
    leukocyte fraction) and, at half weight, CIBERSORT relative immune composition.
    n_immune = 4 (breast), 3 (liver), 3 (pancreas; only the first column is used, one immune type).
    """

    def __init__(self, genes: int, n_immune: int, width: int = 256, bottleneck: int = 96, blocks: int = 2,
                 dropout: float = 0.05):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(genes, width), nn.LayerNorm(width), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(width, dropout=dropout) for _ in range(blocks)))
        self.bottleneck = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, bottleneck), nn.SiLU())
        self.coarse_head = nn.Linear(bottleneck, 3)
        self.immune_head = nn.Linear(bottleneck, n_immune)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.bottleneck(self.blocks(self.stem(x)))
        return F.softmax(self.coarse_head(h), -1), F.softmax(self.immune_head(h), -1)


class CompositionEnsemble(nn.Module):
    """Mean of several frozen CompositionEstimators (3 seeds in CellFactor)."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = [m(x) for m in self.models]
        return torch.stack([o[0] for o in out]).mean(0), torch.stack([o[1] for o in out]).mean(0)


def fine_composition(coarse: Tensor, immune_rel: Tensor, mapping: dict, n_types: int) -> Tensor:
    """(Tumor, Immune, Other) + immune split -> fine pi over all single-cell types.
    'Other' mass is split by a fixed prior (mean training-donor pseudobulk composition of the Other types)."""
    out = coarse.new_zeros(len(coarse), n_types)
    out[:, mapping["tumor_index"]] = coarse[:, 0]
    k = len(mapping["immune_indices"])
    rel = immune_rel
    if k < rel.shape[1]:   # pancreas: one immune type but a 3-column head -> keep the first column(s), renormalise
        rel = rel[:, :k]; rel = rel / rel.sum(1, keepdim=True).clamp_min(1e-12)
    out[:, mapping["immune_indices"]] = coarse[:, 1, None] * rel
    out[:, mapping["other_indices"]] = coarse[:, 2, None] * mapping["other_prior"].to(coarse)[None]
    return out / out.sum(1, keepdim=True).clamp_min(1e-12)


def coarse_composition_loss(pred: Tensor, tumor: Tensor, immune: Tensor) -> Tensor:
    """Joint CE when both experimental fractions exist, BCE on the single available one otherwise (NaN = missing)."""
    ht, hi = torch.isfinite(tumor), torch.isfinite(immune)
    pair = ht & hi
    per = pred.new_zeros(len(pred))
    valid = ht | hi
    if pair.any():
        target = torch.stack([tumor[pair], immune[pair], 1 - tumor[pair] - immune[pair]], 1)
        per[pair] = -(target * pred[pair].clamp_min(1e-8).log()).sum(1)
    only = ht & ~hi
    if only.any():
        p = pred[only, 0].clamp(1e-6, 1 - 1e-6)
        per[only] = -(tumor[only] * p.log() + (1 - tumor[only]) * torch.log1p(-p))
    only = hi & ~ht
    if only.any():
        p = pred[only, 1].clamp(1e-6, 1 - 1e-6)
        per[only] = -(immune[only] * p.log() + (1 - immune[only]) * torch.log1p(-p))
    if not valid.any():
        raise RuntimeError("composition batch has no labels")
    return per[valid].mean()


# ----------------------------------------------------------------------------- Stage 2D (H)
class StateEstimator(nn.Module):
    """Bulk x (panel genes) + stop-grad coarse composition q -> M (cell types x latent): per-type mean Stage 1' state."""

    def __init__(self, genes: int, fine_cell_types: int, coarse_composition_dim: int = 3, latent: int = 128,
                 width: int = 256, bottleneck: int = 128, blocks: int = 2):
        super().__init__()
        self.fine_cell_types, self.latent = fine_cell_types, latent
        self.stem = nn.Sequential(nn.Linear(genes, width), nn.LayerNorm(width), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.bottleneck = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, bottleneck), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(bottleneck + coarse_composition_dim, width), nn.SiLU(),
                                  nn.Linear(width, fine_cell_types * latent))

    def encode(self, x: Tensor) -> Tensor:
        return self.bottleneck(self.blocks(self.stem(x)))

    def decode_state(self, f: Tensor, q: Tensor) -> Tensor:
        return self.head(torch.cat([f, q.detach()], 1)).view(-1, self.fine_cell_types, self.latent)

    def forward(self, x: Tensor, q: Tensor) -> Tensor:
        return self.decode_state(self.encode(x), q)


def calibrate_state_axes(A: Tensor, pa: Tensor, B: Tensor, pb: Tensor, floor: float = 0.05):
    """From paired bags (two disjoint cell draws A/B of the same donor and composition):
    s   per-axis SD of the per-type mean state across bags (pooled over types) -> latent rescaled z' = z / s
    rho split-half reliability per (type, axis) = 1 - Var(A-B)/2 / Var((A+B)/2), floored
    w   = rho / mean(rho), the precision weight of the state and delta losses."""
    n_types, LAT = A.shape[1], A.shape[-1]
    both = pa.bool() & pb.bool()
    s2 = torch.zeros(LAT); n = 0
    noise = torch.zeros(n_types, LAT); total = torch.zeros(n_types, LAT)
    for c in range(n_types):
        m = both[:, c]
        if int(m.sum()) < 30:
            noise[c] = 1.0; total[c] = 1.0
            continue
        a_, b_ = A[m, c], B[m, c]
        mid = (a_ + b_) / 2
        noise[c] = ((a_ - b_).var(0) / 2).clamp_min(1e-12)
        total[c] = mid.var(0).clamp_min(1e-12)
        s2 += mid.var(0) * int(m.sum()); n += int(m.sum())
    s = (s2 / max(n, 1)).sqrt().clamp_min(1e-3)
    rho = (1 - noise / total).clamp(floor, 1.0)
    return s, rho, rho / rho.mean()


def weighted_state_loss(pred: Tensor, target: Tensor, present: Tensor, w: Tensor) -> Tensor:
    """Reliability-weighted MSE of per-type mean states over cell types present in the bag."""
    se = ((pred - target).square() * w).sum(-1) * present
    return se.sum() / (pred.shape[-1] * present.sum().clamp_min(1))


def weighted_delta_loss(pa: Tensor, pb: Tensor, ta: Tensor, tb: Tensor, ma: Tensor, mb: Tensor, w: Tensor) -> Tensor:
    """Paired term: the predicted A-B state difference must match the true A-B difference."""
    mask = (ma.bool() & mb.bool()).float()
    e = (((pa - pb) - (ta - tb)).square() * w).sum(2)
    return (e * mask).sum() / (pa.shape[2] * mask.sum().clamp_min(1))


def _residualize(f: Tensor, q: Tensor) -> Tensor:
    d = torch.cat([torch.ones(len(q), 1, device=q.device), q[:, :2]], 1)
    eye = torch.eye(d.shape[1], device=d.device) * 1e-3
    try:
        beta = torch.linalg.solve(d.T @ d + eye, d.T @ f)
    except torch._C._LinAlgError:
        beta = torch.linalg.lstsq(d.T @ d + eye, d.T @ f).solution
    return f - d @ beta


def conditional_coral(pseudo_f: Tensor, pseudo_q: Tensor, real_f: Tensor, real_q: Tensor) -> Tensor:
    """Align H bottleneck features of pseudobulk bags and real TCGA bulk after regressing out composition
    (mean + covariance matching, CORAL)."""
    p, r = _residualize(pseudo_f, pseudo_q), _residualize(real_f, real_q)
    scale = torch.cat([p.detach(), r.detach()]).std(0).clamp_min(0.1)
    p, r = p / scale, r / scale
    pc, rc = p - p.mean(0), r - r.mean(0)
    cp = pc.T @ pc / max(len(p) - 1, 1)
    cr = rc.T @ rc / max(len(r) - 1, 1)
    return (p.mean(0) - r.mean(0)).square().mean() + (cp - cr).square().mean()


# ----------------------------------------------------------------------------- Stage 3
class CellFactorRisk(nn.Module):
    """r = sum_c pi_c (alpha_c + W_c . M_c)."""

    def __init__(self, cell_types: int, latent: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(cell_types))
        self.W = nn.Parameter(torch.zeros(cell_types, latent))

    def risk(self, pi: Tensor, state: Tensor) -> Tensor:
        return (pi * (self.alpha[None] + torch.einsum("bck,ck->bc", state, self.W))).sum(1)

    def type_risk(self, state: Tensor) -> Tensor:
        """Per-type contribution before composition weighting: alpha_c + W_c . M_c."""
        return self.alpha[None] + torch.einsum("bck,ck->bc", state, self.W)


def cox_nll(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    """Breslow partial likelihood (ties ignored), averaged over events."""
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    return -((r - torch.logcumsumexp(r, 0)) * e).sum() / e.sum().clamp_min(1)


# ----------------------------------------------------------------------------- full model
class CellFactor(nn.Module):
    """Frozen CellFactor for inference: standardised bulk x (full panel) -> pi, M, risk.
    gene_idx selects the H input genes (the 3,000-gene selected panel) out of the full panel."""

    def __init__(self, composition: nn.Module, state: StateEstimator, risk: CellFactorRisk, mapping: dict,
                 n_types: int, gene_idx: Tensor):
        super().__init__()
        self.composition, self.state, self.risk_head = composition, state, risk
        self.mapping, self.n_types = mapping, n_types
        self.register_buffer("gene_idx", torch.as_tensor(gene_idx, dtype=torch.long))

    @torch.no_grad()
    def forward(self, x: Tensor):
        coarse, immune_rel = self.composition(x)
        pi = fine_composition(coarse, immune_rel, self.mapping, self.n_types)
        M = self.state(x[:, self.gene_idx], coarse)
        return {"pi": pi, "M": M, "coarse": coarse, "risk": self.risk_head.risk(pi, M)}


# ----------------------------------------------------------------------------- checkpoint loaders
def load_composition_estimator(path, device="cpu") -> CompositionEstimator:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("immune_relative_head", "immune_head"): v for k, v in ck["state_dict"].items()}   # breast name
    m = CompositionEstimator(sd["stem.0.weight"].shape[1], sd["immune_head.weight"].shape[0], **ck["model_config"])
    m.load_state_dict(sd)
    return freeze(m.to(device))


def load_state_estimator(path, device="cpu") -> StateEstimator:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, cfg = ck["state_dict"], ck["model_config"]
    m = StateEstimator(sd["stem.0.weight"].shape[1], cfg.get("fine_cell_types", sd["head.2.weight"].shape[0] // cfg["latent"]),
                       cfg["coarse_composition_dim"], cfg["latent"], cfg["width"], cfg["bottleneck"], cfg["blocks"])
    m.load_state_dict(sd)
    return freeze(m.to(device))
