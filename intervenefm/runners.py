"""Shared training / audit / statistics helpers for the InterveneFM exea jobs.

Everything here REUSES the vendored, validated pieces from `intervenefm/src/`
and `intervenefm/scripts/`:
  - src.data_norman / src.data_replogle / src.data_replogle_rpe1 for loading +
    HVG selection + train/test splits + the shared PerturbSeqDataset.
  - src.cpa_minimal.CPAMinimal / src.cpa_cpg.CPGModel for the models.
  - src.audit.predict_under_mode / get_top_deg_indices for the per-mode
    embedding-ablation audit (the validated statistical core).
  - scripts.train_audit_cpg.{load_replogle_no_hvg,restrict_to_hvg} and
    scripts.train_audit_cpg_norman.load_norman_full_panel for the full-gene-
    panel loading needed by the CPG gene-identity computation.

This module adds only what's genuinely new:
  - a dataset/model-agnostic manual audit loop (generalizes the per-script
    "doubles" (Norman) / "singles" (Replogle) audit loops into one function),
  - cluster-bootstrap + paired-Cohen's-d + Wilcoxon statistics (reusing the
    exact methodology of scripts/cluster_bootstrap.py and
    scripts/effect_sizes.py, refactored into reusable functions),
  - the CPG_linear ablation (a CPGPertEncoder subclass with the MLP replaced
    by a single nn.Linear -- src/cpa_cpg.py is untouched),
  - a checkpoint-aware generic training loop for exea's unattended, resumable
    execution model.
"""
from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon
from torch.utils.data import DataLoader

INTERVENEFM_ROOT = Path(__file__).resolve().parent
if str(INTERVENEFM_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERVENEFM_ROOT))

from exea.checkpoint import Checkpointer  # noqa: E402
from exea.jobs import UnitPaused  # noqa: E402

from src.audit import get_top_deg_indices, predict_under_mode  # noqa: E402
from src.cpa_cpg import CPGConfig, CPGModel, CPGPertEncoder, compute_gene_identity_table, \
    compute_gene_identity_table_go  # noqa: E402
from src.cpa_minimal import CPAConfig, CPAMinimal  # noqa: E402
from src.data_norman import PerturbSeqDataset  # noqa: E402

from . import staging  # noqa: E402
from .pertformer import PertFormer, PertFormerConfig, build_is_perturbed_mask  # noqa: E402

log = logging.getLogger("intervenefm.runners")


class TrainingInterrupted(UnitPaused):
    """Raised when ctx.should_stop() fires mid-training. Subclasses UnitPaused
    so the orchestrator rolls back the attempt (no penalty) and resumes next
    day from the checkpoint. A checkpoint is
    always saved immediately before this is raised, so the orchestrator's
    retry (same day, or the next day) resumes from the saved epoch instead
    of restarting -- see exea/checkpoint.py's docstring for the contract."""


# ============================================================================
# Config
# ============================================================================
VARIANTS = ("learned_CPA_baseline", "CPG_SVD", "CPG_GO", "CPG_linear")
PERTFORMER_VARIANTS = ("A", "B")
DATASETS = ("norman", "replogle_k562", "replogle_rpe1")

REAL_CFG: Dict[str, dict] = {
    "norman": dict(
        hvg=2000, max_cells=60000, epochs=20, batch_size=128, lr=3e-4,
        z_dim=64, pert_dim=32, hidden=256, pert_mlp_hidden=128, gene_id_dim=64,
        n_per_pair=200, deg_k=200, split_kind="0/2", test_frac_doubles=0.3,
        n_boot=2000,
        pf_d_model=64, pf_nhead=4, pf_layers=2, pf_ff=128, pf_epochs=10,
    ),
    "replogle_k562": dict(
        hvg=2000, max_cells=80000, epochs=40, batch_size=128, lr=3e-4,
        z_dim=64, pert_dim=32, hidden=256, pert_mlp_hidden=128, gene_id_dim=64,
        n_per_pair=200, deg_k=200, test_frac_genes=0.2, min_cells_per_gene=50,
        max_test_genes=80, n_boot=2000,
        pf_d_model=64, pf_nhead=4, pf_layers=2, pf_ff=128, pf_epochs=15,
    ),
    "replogle_rpe1": dict(
        hvg=2000, max_cells=80000, epochs=40, batch_size=128, lr=3e-4,
        z_dim=64, pert_dim=32, hidden=256, pert_mlp_hidden=128, gene_id_dim=64,
        n_per_pair=200, deg_k=200, test_frac_genes=0.2, min_cells_per_gene=50,
        max_test_genes=80, n_boot=2000,
        pf_d_model=64, pf_nhead=4, pf_layers=2, pf_ff=128, pf_epochs=15,
    ),
}

SMOKE_CFG_OVERRIDE = dict(
    hvg=40, max_cells=None, epochs=2, batch_size=16,
    z_dim=8, pert_dim=4, hidden=16, pert_mlp_hidden=8, gene_id_dim=8,
    n_per_pair=16, deg_k=16, split_kind="0/2", test_frac_doubles=0.3,
    test_frac_genes=0.34, min_cells_per_gene=5, max_test_genes=3, n_boot=50,
    pf_d_model=8, pf_nhead=2, pf_layers=1, pf_ff=16, pf_epochs=2,
)


def get_config(dataset: str, smoke: bool) -> dict:
    cfg = dict(REAL_CFG[dataset])
    if smoke:
        cfg.update(SMOKE_CFG_OVERRIDE)
    return cfg


# ============================================================================
# CPG_linear ablation (reviewer-demanded) -- subclass only, src/cpa_cpg.py
# is never modified.
# ============================================================================
class CPGPertEncoderLinear(CPGPertEncoder):
    """Same as CPGPertEncoder except the pert-embedding MLP is replaced by a
    single nn.Linear (no nonlinearity). Isolates: does the CPG's escape from
    gradient starvation come from the identity *feature* or from the MLP's
    nonlinear capacity to reshape it?"""

    def __init__(self, cfg: CPGConfig, gene_identity_table: torch.Tensor):
        super().__init__(cfg, gene_identity_table)
        self.mlp = nn.Linear(cfg.gene_id_dim, cfg.pert_dim)


# ============================================================================
# Dataset bundle loading (reuses src.data_* + scripts.train_audit_cpg* helpers)
# ============================================================================
def _cap_test_genes(adata, split: dict, max_test_genes: int, seed: int) -> dict:
    """Mirrors the test-gene capping logic in scripts/train_audit_replogle.py
    / train_audit_rpe1.py / train_audit_cpg.py (they all inline the same few
    lines rather than exporting a function)."""
    if len(split["test_genes"]) <= max_test_genes:
        return split
    rng = np.random.default_rng(seed)
    test_genes = list(rng.choice(np.array(split["test_genes"], dtype=object),
                                  size=max_test_genes, replace=False))
    keep = set(test_genes)
    n_pert_arr = adata.obs["n_pert"].values
    pg_arr = adata.obs["pert_genes"].values
    new_test, train_cells = [], []
    for ci, cell_id in enumerate(adata.obs.index):
        if n_pert_arr[ci] == 0:
            train_cells.append(cell_id)
        elif n_pert_arr[ci] == 1 and pg_arr[ci][0] in keep:
            new_test.append(cell_id)
        elif n_pert_arr[ci] == 1:
            train_cells.append(cell_id)
    split = dict(split)
    split["test_genes"] = test_genes
    split["test_cells"] = new_test
    split["train_cells"] = train_cells
    return split


def load_bundle(dataset: str, cfg: dict, seed: int, need_full_panel: bool) -> dict:
    """Returns {adata, adata_full(optional), vocab, split, kind, test_items}."""
    if dataset == "norman":
        from src import data_norman
        norman_path = staging.resolve_norman_path()
        if need_full_panel:
            from scripts import train_audit_cpg_norman as tacn
            tacn.NORMAN_DATA = norman_path  # override the module's captured path constant
            adata_full = tacn.load_norman_full_panel(max_cells=cfg["max_cells"], seed=seed)
            import scanpy as sc
            ctrl_mask = (adata_full.obs["n_pert"] == 0).values
            ctrl_only = adata_full[ctrl_mask].copy()
            sc.pp.highly_variable_genes(ctrl_only, n_top_genes=cfg["hvg"], subset=False,
                                         flavor="seurat", batch_key=None)
            hvg_mask = ctrl_only.var["highly_variable"].values
            adata = adata_full[:, hvg_mask].copy()
        else:
            data_norman.DATA = norman_path
            adata = data_norman.load_norman(n_top_hvg=cfg["hvg"], max_cells=cfg["max_cells"], seed=seed)
            adata_full = None
        vocab = data_norman.build_gene_vocab(adata)
        split = data_norman.build_split(adata, kind=cfg.get("split_kind", "0/2"), seed=seed,
                                         test_frac_doubles=cfg.get("test_frac_doubles", 0.3))
        return {"adata": adata, "adata_full": adata_full, "vocab": vocab, "split": split,
                "kind": "doubles", "test_items": split["test_pairs"]}

    which = "k562" if dataset == "replogle_k562" else "rpe1"
    from src import data_replogle, data_replogle_rpe1
    path = staging.resolve_replogle_path(which)
    if need_full_panel:
        from scripts.train_audit_cpg import load_replogle_no_hvg, restrict_to_hvg
        adata_full = load_replogle_no_hvg(path, max_cells=cfg["max_cells"], seed=seed)
        adata = restrict_to_hvg(adata_full, n_top_hvg=cfg["hvg"])
    else:
        mod = data_replogle if which == "k562" else data_replogle_rpe1
        mod.DATA = path
        if which == "k562":
            adata = data_replogle.load_replogle(n_top_hvg=cfg["hvg"], max_cells=cfg["max_cells"], seed=seed)
        else:
            adata = data_replogle_rpe1.load_replogle_rpe1(n_top_hvg=cfg["hvg"], max_cells=cfg["max_cells"], seed=seed)
        adata_full = None
    vocab = data_replogle.build_gene_vocab_replogle(adata)
    split = data_replogle.build_split_replogle(adata, test_frac_genes=cfg["test_frac_genes"], seed=seed,
                                                min_cells_per_gene=cfg["min_cells_per_gene"])
    split = _cap_test_genes(adata, split, cfg["max_test_genes"], seed)
    return {"adata": adata, "adata_full": adata_full, "vocab": vocab, "split": split,
            "kind": "singles", "test_items": split["test_genes"]}


# ============================================================================
# Model construction
# ============================================================================
def build_cpa_or_cpg_model(variant: str, bundle: dict, cfg: dict, seed: int):
    adata, vocab, split = bundle["adata"], bundle["vocab"], bundle["split"]
    if variant == "learned_CPA_baseline":
        mcfg = CPAConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab),
                          z_dim=cfg["z_dim"], pert_dim=cfg["pert_dim"], hidden=cfg["hidden"])
        return CPAMinimal(mcfg), None

    gene_id_dim = cfg["gene_id_dim"]
    if variant == "CPG_GO":
        gaf_path = staging.resolve_gaf_path(list(vocab.keys()))
        gene_id_table = compute_gene_identity_table_go(str(gaf_path), vocab, d_id=gene_id_dim, seed=seed)
    elif variant in ("CPG_SVD", "CPG_linear"):
        assert bundle.get("adata_full") is not None, f"{variant} requires the full gene panel"
        gene_id_table = compute_gene_identity_table(bundle["adata_full"], vocab, split["ctrl_cells"],
                                                     d_id=gene_id_dim, seed=seed)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    mcfg = CPGConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab), gene_id_dim=gene_id_dim,
                      z_dim=cfg["z_dim"], pert_dim=cfg["pert_dim"], hidden=cfg["hidden"],
                      pert_mlp_hidden=cfg["pert_mlp_hidden"])
    model = CPGModel(mcfg, gene_id_table)
    if variant == "CPG_linear":
        model.pert_encoder = CPGPertEncoderLinear(mcfg, gene_id_table)
    return model, gene_id_table


# ============================================================================
# Generic checkpoint-aware training loop (CPA / CPG models)
# ============================================================================
def train_model(model, loader: DataLoader, device, epochs: int, lr: float, weight_decay: float = 1e-5,
                 ckpt: Optional[Checkpointer] = None, ctx=None, logger: Optional[logging.Logger] = None,
                 log_every: int = 5) -> pd.DataFrame:
    model.to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lr, weight_decay=weight_decay)
    start_epoch, train_log = 0, []
    if ckpt is not None and ckpt.exists():
        state = ckpt.load()
        if state is not None:
            try:
                model.load_state_dict(state["state"]["model"])
                opt.load_state_dict(state["state"]["opt"])
                start_epoch = int(state["meta"].get("epoch", 0))
                train_log = list(state["meta"].get("train_log", []))
                if logger:
                    logger.info("[resume] restored checkpoint at epoch %d", start_epoch)
            except Exception:
                if logger:
                    logger.warning("[resume] checkpoint load failed, starting fresh")

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for basal, target, pidx in loader:
            basal, target, pidx = basal.to(device), target.to(device), pidx.to(device)
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * basal.shape[0]
            ep_n += basal.shape[0]
        avg = ep_loss / max(ep_n, 1)
        train_log.append({"epoch": epoch + 1, "mse": avg, "elapsed_s": time.time() - t0})
        if logger and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
            logger.info("[train] epoch %d/%d mse=%.4f (%.1fs)", epoch + 1, epochs, avg, time.time() - t0)
        meta = {"epoch": epoch + 1, "train_log": train_log}
        if ckpt is not None:
            ckpt.save({"model": model.state_dict(), "opt": opt.state_dict()}, meta)
        if ctx is not None and ctx.should_stop():
            if ckpt is not None:
                ckpt.save({"model": model.state_dict(), "opt": opt.state_dict()}, meta, force=True)
            raise TrainingInterrupted(f"time budget exhausted at epoch {epoch + 1}/{epochs}")
    if ckpt is not None:
        ckpt.clear()
    return pd.DataFrame(train_log)


# ============================================================================
# Generic manual audit (generalizes the doubles/singles loops in the
# vendored scripts into one dataset/model-agnostic function).
# ============================================================================
def _get_X_rows_fn(adata):
    X = adata.X
    is_sparse = hasattr(X, "toarray")

    def get(rows):
        if is_sparse:
            return np.asarray(X[rows].toarray()).astype(np.float32)
        return np.asarray(X[rows]).astype(np.float32)

    return get


def compute_train_embed_pool(model, train_pert_gene_indices: List[int], device) -> torch.Tensor:
    if len(train_pert_gene_indices) == 0:
        return torch.zeros((0, model.cfg.pert_dim), device=device)
    idx_t = torch.tensor(train_pert_gene_indices, dtype=torch.long, device=device)
    doubles = torch.stack([idx_t, torch.zeros_like(idx_t)], dim=1)
    with torch.no_grad():
        return model.pert_encoder(doubles)


def _train_pert_gene_indices_and_cells(adata, split: dict, vocab: dict) -> Tuple[List[int], List[str]]:
    obs = adata.obs
    train_set = set(split["train_cells"])
    pg = obs["pert_genes"].values
    n_pert_arr = obs["n_pert"].values
    train_pert_genes_set = set()
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set:
            for g in pg[ci]:
                if g in vocab:
                    train_pert_genes_set.add(g)
    train_pert_gene_indices = sorted(vocab[g] for g in train_pert_genes_set)
    train_perturbed_cell_ids = [cell_id for ci, cell_id in enumerate(obs.index)
                                 if cell_id in train_set and n_pert_arr[ci] >= 1]
    return train_pert_gene_indices, train_perturbed_cell_ids


def run_manual_audit(model, bundle: dict, vocab: dict, seed: int, n_per_pair: int, deg_k: int,
                      device) -> pd.DataFrame:
    """Audits a CPAMinimal- or CPGModel-family model across all 6 embedding
    modes (learned/mean/zero/random/identity/pop_mean), reusing
    src.audit.predict_under_mode + get_top_deg_indices verbatim. Generalizes
    over both "doubles" (Norman 0/2) and "singles" (Replogle 0/1) test items."""
    adata, split, kind = bundle["adata"], bundle["split"], bundle["kind"]
    train_pert_gene_indices, train_perturbed_cell_ids = _train_pert_gene_indices_and_cells(adata, split, vocab)

    rng = np.random.default_rng(seed)
    train_embed_pool = compute_train_embed_pool(model, train_pert_gene_indices, device)

    get = _get_X_rows_fn(adata)
    cell_to_row = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([cell_to_row[c] for c in split["ctrl_cells"]])
    ctrl_X = get(ctrl_rows)

    train_pert_mean = None
    if train_perturbed_cell_ids:
        train_pert_rows = np.array([cell_to_row[c] for c in train_perturbed_cell_ids])
        train_pert_mean = torch.from_numpy(get(train_pert_rows).mean(0)).to(device)

    pg = adata.obs["pert_genes"].values
    items = split["test_pairs"] if kind == "doubles" else split["test_genes"]
    modes = ["learned", "mean", "zero", "random", "identity"]
    if train_pert_mean is not None:
        modes.append("pop_mean")

    rows_out = []
    model.eval()
    for item in items:
        if kind == "doubles":
            g1, g2 = item
            mask = np.array([(set(g) == {g1, g2}) and (len(g) == 2) for g in pg])
            key, idx_g1, idx_g2 = f"{g1}_{g2}", vocab.get(g1, 0), vocab.get(g2, 0)
        else:
            tg = item
            mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg])
            key, idx_g1, idx_g2 = tg, vocab.get(tg, 0), 0
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5:
            continue
        pert_X = get(pert_rows)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=deg_k)
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)
        obs_delta_top = obs_delta_full[deg_idx]

        basal_rows = rng.choice(ctrl_rows, size=n_per_pair, replace=True)
        basal = torch.from_numpy(get(basal_rows)).to(device)
        pert_idx = torch.tensor([[idx_g1, idx_g2]] * n_per_pair, dtype=torch.long, device=device)

        with torch.no_grad():
            for mode in modes:
                x_hat = predict_under_mode(model, basal, pert_idx, mode, train_embed_pool, rng,
                                            train_pert_mean_x=train_pert_mean).cpu().numpy()
                pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
                pred_delta_top = pred_delta[deg_idx]
                rho = spearmanr(pred_delta_top, obs_delta_top).statistic
                num = np.dot(pred_delta - pred_delta.mean(), obs_delta_full - obs_delta_full.mean())
                denom = (np.linalg.norm(pred_delta - pred_delta.mean()) *
                         np.linalg.norm(obs_delta_full - obs_delta_full.mean()) + 1e-12)
                rows_out.append({
                    "item": key, "mode": mode,
                    "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                    "Pearson_delta_full": float(num / denom),
                    "n_obs_pert": int(len(pert_rows)),
                })
    return pd.DataFrame(rows_out)


# ============================================================================
# PertFormer (P20) training + audit
# ============================================================================
def build_pos_lookup(adata, vocab: dict) -> torch.Tensor:
    var_to_pos = {name: i for i, name in enumerate(adata.var_names)}
    n_perts = max(vocab.values()) if vocab else 0
    lookup = torch.full((n_perts + 1,), -1, dtype=torch.long)
    for gname, pidx in vocab.items():
        if gname in var_to_pos:
            lookup[pidx] = var_to_pos[gname]
    return lookup


def snapshot_init_qpert(model: PertFormer) -> dict:
    if model.cfg.variant == "A":
        return {"kind": "A", "weight": model.q_pert.weight.data.clone().cpu()}
    return {"kind": "B", "mlp": copy.deepcopy(model.q_pert_mlp).cpu().eval()}


def train_pertformer(model: PertFormer, loader: DataLoader, pos_lookup: torch.Tensor, device, epochs: int,
                      lr: float, weight_decay: float = 1e-5, ckpt: Optional[Checkpointer] = None,
                      ctx=None, logger: Optional[logging.Logger] = None, log_every: int = 5,
                      track_grad_indices: Optional[List[int]] = None):
    model.to(device)
    pos_lookup_dev = pos_lookup.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch, train_log = 0, []
    grad_norm_sum = None
    track_idx_t = None
    if track_grad_indices is not None and model.cfg.variant == "A":
        grad_norm_sum = torch.zeros(len(track_grad_indices))
        # Must live on the model's device: it indexes weight.grad (on device)
        # inside the training loop. CPU-only smoke never exposed this.
        track_idx_t = torch.tensor(track_grad_indices, dtype=torch.long,
                                   device=next(model.parameters()).device)

    if ckpt is not None and ckpt.exists():
        state = ckpt.load()
        if state is not None:
            try:
                model.load_state_dict(state["state"]["model"])
                opt.load_state_dict(state["state"]["opt"])
                start_epoch = int(state["meta"].get("epoch", 0))
                train_log = list(state["meta"].get("train_log", []))
                gns = state["meta"].get("grad_norm_sum")
                if gns is not None and grad_norm_sum is not None:
                    grad_norm_sum = torch.tensor(gns)
            except Exception:
                if logger:
                    logger.warning("[resume] pertformer checkpoint load failed, starting fresh")

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for basal, target, pidx in loader:
            basal, target, pidx = basal.to(device), target.to(device), pidx.to(device)
            n_genes = basal.shape[1]
            is_pert_mask = build_is_perturbed_mask(pidx, pos_lookup_dev, n_genes)
            delta_pred = model(basal, pidx, is_perturbed_mask=is_pert_mask)
            delta_target = target - basal
            loss = F.mse_loss(delta_pred, delta_target)
            opt.zero_grad()
            loss.backward()
            if grad_norm_sum is not None and model.q_pert.weight.grad is not None:
                with torch.no_grad():
                    gn = model.q_pert.weight.grad[track_idx_t].norm(dim=-1).cpu()
                    grad_norm_sum += gn
            opt.step()
            ep_loss += loss.item() * basal.shape[0]
            ep_n += basal.shape[0]
        avg = ep_loss / max(ep_n, 1)
        train_log.append({"epoch": epoch + 1, "mse": avg, "elapsed_s": time.time() - t0})
        if logger and ((epoch + 1) % log_every == 0 or epoch == epochs - 1):
            logger.info("[pertformer] epoch %d/%d mse=%.4f (%.1fs)", epoch + 1, epochs, avg, time.time() - t0)
        meta = {"epoch": epoch + 1, "train_log": train_log}
        if grad_norm_sum is not None:
            meta["grad_norm_sum"] = grad_norm_sum.tolist()
        if ckpt is not None:
            ckpt.save({"model": model.state_dict(), "opt": opt.state_dict()}, meta)
        if ctx is not None and ctx.should_stop():
            if ckpt is not None:
                ckpt.save({"model": model.state_dict(), "opt": opt.state_dict()}, meta, force=True)
            raise TrainingInterrupted(f"time budget exhausted at epoch {epoch + 1}/{epochs}")
    if ckpt is not None:
        ckpt.clear()
    return pd.DataFrame(train_log), grad_norm_sum


def compute_train_q_pool(model: PertFormer, train_pert_gene_indices: List[int], device) -> torch.Tensor:
    if len(train_pert_gene_indices) == 0:
        return torch.zeros((0, model.cfg.d_model), device=device)
    idx_t = torch.tensor(train_pert_gene_indices, dtype=torch.long, device=device)
    doubles = torch.stack([idx_t, torch.zeros_like(idx_t)], dim=1)
    with torch.no_grad():
        q = model.get_q_pert(doubles)  # (T, 1, d)
    return q.squeeze(1)


def pertformer_predict_under_mode(model: PertFormer, basal: torch.Tensor, pert_idx: torch.Tensor, mode: str,
                                   train_q_pool: torch.Tensor, rng: np.random.Generator,
                                   pos_lookup: torch.Tensor, train_pert_mean_x: Optional[torch.Tensor] = None
                                   ) -> torch.Tensor:
    """Mirrors src.audit.predict_under_mode's semantics for PertFormer."""
    B, G = basal.shape
    device = basal.device
    is_pert_mask = build_is_perturbed_mask(pert_idx, pos_lookup.to(device), G)

    if mode == "pop_mean":
        if train_pert_mean_x is None:
            raise ValueError("pop_mean requires train_pert_mean_x")
        return train_pert_mean_x.unsqueeze(0).expand(B, -1).clone()
    if mode == "identity":
        return basal.clone()
    if mode == "learned":
        delta = model(basal, pert_idx, is_perturbed_mask=is_pert_mask)
        return basal + delta

    d = train_q_pool.shape[-1] if train_q_pool.numel() > 0 else model.cfg.d_model
    if mode == "zero":
        q = torch.zeros(1, 1, d, device=device)
    elif mode == "mean":
        n_p = (pert_idx > 0).sum(dim=1).float().view(B, 1, 1)
        mean_q = train_q_pool.mean(dim=0).view(1, 1, d) if train_q_pool.numel() > 0 else torch.zeros(1, 1, d, device=device)
        q = mean_q.to(device) * n_p
    elif mode == "random":
        n_p = (pert_idx > 0).sum(dim=1).cpu().numpy()
        T = train_q_pool.shape[0]
        q = torch.zeros(B, 1, d, device=device)
        for i in range(B):
            k = int(n_p[i])
            if k == 0 or T == 0:
                continue
            picks = rng.choice(T, size=min(k, T), replace=False)
            q[i, 0] = train_q_pool[picks].sum(dim=0)
    else:
        raise ValueError(mode)

    delta = model(basal, pert_idx, is_perturbed_mask=is_pert_mask, q_override=q)
    return basal + delta


def run_pertformer_audit(model: PertFormer, bundle: dict, vocab: dict, seed: int, n_per_pair: int,
                          deg_k: int, device, pos_lookup: torch.Tensor) -> pd.DataFrame:
    adata, split, kind = bundle["adata"], bundle["split"], bundle["kind"]
    train_pert_gene_indices, train_perturbed_cell_ids = _train_pert_gene_indices_and_cells(adata, split, vocab)

    rng = np.random.default_rng(seed)
    train_q_pool = compute_train_q_pool(model, train_pert_gene_indices, device)

    get = _get_X_rows_fn(adata)
    cell_to_row = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([cell_to_row[c] for c in split["ctrl_cells"]])
    ctrl_X = get(ctrl_rows)

    train_pert_mean = None
    if train_perturbed_cell_ids:
        train_pert_rows = np.array([cell_to_row[c] for c in train_perturbed_cell_ids])
        train_pert_mean = torch.from_numpy(get(train_pert_rows).mean(0)).to(device)

    pg = adata.obs["pert_genes"].values
    items = split["test_pairs"] if kind == "doubles" else split["test_genes"]
    modes = ["learned", "mean", "zero", "random", "identity"]
    if train_pert_mean is not None:
        modes.append("pop_mean")

    rows_out = []
    model.eval()
    for item in items:
        if kind == "doubles":
            g1, g2 = item
            mask = np.array([(set(g) == {g1, g2}) and (len(g) == 2) for g in pg])
            key, idx_g1, idx_g2 = f"{g1}_{g2}", vocab.get(g1, 0), vocab.get(g2, 0)
        else:
            tg = item
            mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg])
            key, idx_g1, idx_g2 = tg, vocab.get(tg, 0), 0
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5:
            continue
        pert_X = get(pert_rows)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=deg_k)
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)
        obs_delta_top = obs_delta_full[deg_idx]

        basal_rows = rng.choice(ctrl_rows, size=n_per_pair, replace=True)
        basal = torch.from_numpy(get(basal_rows)).to(device)
        pert_idx = torch.tensor([[idx_g1, idx_g2]] * n_per_pair, dtype=torch.long, device=device)

        with torch.no_grad():
            for mode in modes:
                x_hat = pertformer_predict_under_mode(model, basal, pert_idx, mode, train_q_pool, rng,
                                                        pos_lookup, train_pert_mean_x=train_pert_mean).cpu().numpy()
                pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
                pred_delta_top = pred_delta[deg_idx]
                rho = spearmanr(pred_delta_top, obs_delta_top).statistic
                num = np.dot(pred_delta - pred_delta.mean(), obs_delta_full - obs_delta_full.mean())
                denom = (np.linalg.norm(pred_delta - pred_delta.mean()) *
                         np.linalg.norm(obs_delta_full - obs_delta_full.mean()) + 1e-12)
                rows_out.append({
                    "item": key, "mode": mode,
                    "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                    "Pearson_delta_full": float(num / denom),
                    "n_obs_pert": int(len(pert_rows)),
                })
    return pd.DataFrame(rows_out)


def compute_dormancy_diagnostics(model: PertFormer, test_gene_vocab_indices: List[int],
                                  init_snapshot: dict, grad_norm_sum: Optional[torch.Tensor],
                                  track_index_of: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """Empirically demonstrates (or refutes) dormancy of the held-out q_pert
    row: movement-from-init (both variants) and accumulated gradient norm
    during training (variant A only -- variant B has no per-gene parameter,
    only a shared MLP, so "gradient on the row" isn't a meaningful concept
    there; its escape mechanism is measured purely via movement-from-init)."""
    device = next(model.parameters()).device
    idx_t = torch.tensor(test_gene_vocab_indices, dtype=torch.long)
    rows = []
    if init_snapshot["kind"] == "A":
        init_w = init_snapshot["weight"][idx_t]                       # init snapshot is CPU
        cur_w = model.q_pert.weight.data[idx_t.to(device)].cpu()      # weight is on device
        movement = (cur_w - init_w).norm(dim=-1)
        for i, pidx in enumerate(test_gene_vocab_indices):
            gn = 0.0
            if grad_norm_sum is not None and track_index_of is not None and pidx in track_index_of:
                gn = float(grad_norm_sum[track_index_of[pidx]])
            rows.append({"pert_idx": int(pidx), "variant": "A",
                         "movement_from_init": float(movement[i]),
                         "grad_norm_sum_during_training": gn})
    else:
        ids = model.gene_identity_table[idx_t.to(device)].to(device)
        with torch.no_grad():
            cur_out = model.q_pert_mlp(ids).cpu()
            init_out = init_snapshot["mlp"](ids.cpu())
        movement = (cur_out - init_out).norm(dim=-1)
        for i, pidx in enumerate(test_gene_vocab_indices):
            rows.append({"pert_idx": int(pidx), "variant": "B",
                         "movement_from_init": float(movement[i]),
                         "grad_norm_sum_during_training": float("nan")})
    return pd.DataFrame(rows)


# ============================================================================
# Statistics: cluster-bootstrap CIs + paired Cohen's d + Wilcoxon.
# Reuses the exact methodology of scripts/cluster_bootstrap.py and
# scripts/effect_sizes.py (resample by ITEM -- pair or gene -- not by row),
# refactored into reusable, dataset-agnostic functions so every unit emits a
# full numeric table instead of a point estimate ("around 90%").
# ============================================================================
def cluster_bootstrap_table(df: pd.DataFrame, item_col: str = "item", mode_col: str = "mode",
                             value_col: str = "DE_Spearman", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for mode in df[mode_col].unique():
        sub = df[df[mode_col] == mode]
        per_item = sub.groupby(item_col)[value_col].mean().dropna()
        items = per_item.index.values
        if len(items) == 0:
            continue
        boots = np.array([per_item.loc[rng.choice(items, size=len(items), replace=True)].mean()
                           for _ in range(n_boot)])
        rows.append({
            "mode": mode, "n_items": int(len(items)),
            "mean": float(per_item.mean()),
            "ci_lower": float(np.percentile(boots, 2.5)),
            "ci_upper": float(np.percentile(boots, 97.5)),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def gap_stats_table(df: pd.DataFrame, item_col: str = "item", mode_col: str = "mode",
                     value_col: str = "DE_Spearman", baseline_mode: str = "learned",
                     n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Cluster-bootstrap CI + paired Cohen's d + Wilcoxon signed-rank on the
    gap (learned - ablation_mode), resampling by item."""
    rng = np.random.default_rng(seed)
    base = df[df[mode_col] == baseline_mode].groupby(item_col)[value_col].mean()
    rows = []
    for mode in df[mode_col].unique():
        if mode == baseline_mode:
            continue
        other = df[df[mode_col] == mode].groupby(item_col)[value_col].mean()
        common = base.index.intersection(other.index)
        if len(common) < 3:
            continue
        diff = (base.loc[common] - other.loc[common]).dropna()
        common = diff.index.values
        if len(common) < 3:
            continue
        boots = np.array([diff.loc[rng.choice(common, size=len(common), replace=True)].mean()
                           for _ in range(n_boot)])
        sd = diff.std(ddof=1)
        d = float(diff.mean() / sd) if sd > 0 else float("inf")
        try:
            stat, pval = wilcoxon(diff.values)
        except ValueError:
            stat, pval = float("nan"), float("nan")
        rows.append({
            "mode": mode, "n_items": int(len(common)),
            "gap_mean": float(diff.mean()),
            "gap_ci_lower": float(np.percentile(boots, 2.5)),
            "gap_ci_upper": float(np.percentile(boots, 97.5)),
            "cohens_d": d, "wilcoxon_p": float(pval),
        })
    return pd.DataFrame(rows)


# ============================================================================
# misc
# ============================================================================
def make_checkpointer(ctx, unit_id: str) -> Checkpointer:
    safe = unit_id.replace("::", "__").replace("/", "_")
    return Checkpointer(Path(ctx.state_dir) / "intervenefm" / "ckpt" / f"{safe}.pt", every_seconds=120.0)


def is_smoke() -> bool:
    return staging._is_smoke()
