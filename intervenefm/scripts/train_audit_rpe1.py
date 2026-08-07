"""Replogle RPE1 essential audit — same pipeline as K562, different cell line.

Single-perturbation 0/1 split. Identical 6-mode audit as K562 / Norman runs.
Goal: kill the "K562-only" reviewer critique. If RPE1 inversion replicates
the K562 inversion (pop_mean > learned), the audit is robustly cell-line-
generalizable.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_replogle_rpe1 import load_replogle_rpe1
from src.data_replogle import build_split_replogle, build_gene_vocab_replogle
from src.data_norman import PerturbSeqDataset
from src.cpa_minimal import CPAMinimal, CPAConfig
from src.audit import predict_under_mode, get_top_deg_indices
from scipy.stats import spearmanr


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_cells", type=int, default=80000)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_tag", default="rpe1")
    ap.add_argument("--n_per_pair", type=int, default=200)
    ap.add_argument("--max_test_genes", type=int, default=80)
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[run_rpe1] {vars(args)}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()
    adata = load_replogle_rpe1(n_top_hvg=args.hvg, max_cells=args.max_cells, seed=args.seed)
    print(f"[run_rpe1] adata {adata.shape} in {time.time()-t0:.1f}s")
    vocab = build_gene_vocab_replogle(adata)
    split = build_split_replogle(adata, test_frac_genes=0.2, seed=args.seed)

    if len(split['test_genes']) > args.max_test_genes:
        rng = np.random.default_rng(args.seed)
        split['test_genes'] = list(rng.choice(np.array(split['test_genes'], dtype=object),
                                              size=args.max_test_genes, replace=False))
        keep = set(split['test_genes'])
        new_test = []
        n_pert_arr = adata.obs['n_pert'].values
        pg_arr = adata.obs['pert_genes'].values
        for ci, cell_id in enumerate(adata.obs.index):
            if n_pert_arr[ci] == 1 and pg_arr[ci][0] in keep:
                new_test.append(cell_id)
        split['test_cells'] = new_test
        train_cells = []
        for ci, cell_id in enumerate(adata.obs.index):
            if n_pert_arr[ci] == 0:
                train_cells.append(cell_id)
            elif n_pert_arr[ci] == 1 and pg_arr[ci][0] not in keep:
                train_cells.append(cell_id)
        split['train_cells'] = train_cells
        print(f"[run_rpe1] capped test_genes={len(split['test_genes'])}, train={len(train_cells)}, test={len(new_test)}")

    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    cfg = CPAConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab),
                    z_dim=64, pert_dim=32, hidden=256)
    model = CPAMinimal(cfg)
    print(f"[run_rpe1] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_log = []
    for epoch in range(args.epochs):
        model.train(); ep_loss = 0.0; ep_n = 0
        for basal, target, pidx in train_loader:
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * basal.shape[0]; ep_n += basal.shape[0]
        avg = ep_loss / ep_n
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"[train] epoch {epoch+1}/{args.epochs} mse={avg:.4f} ({time.time()-t0:.1f}s)")
        train_log.append({"epoch": epoch+1, "mse": avg, "elapsed_s": time.time()-t0})

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(train_log).to_csv(out / f"training_log_{args.out_tag}_seed{args.seed}.csv", index=False)
    # Save the trained model for later analysis (Replogle mechanism diagnostic)
    torch.save({"state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "vocab": vocab,
                "split": split},
               out / f"model_{args.out_tag}_seed{args.seed}.pt")

    # === Audit ===
    obs = adata.obs
    train_set = set(split['train_cells'])
    pg = obs['pert_genes'].values
    train_pert_genes_set = set()
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set:
            for g in pg[ci]:
                if g in vocab:
                    train_pert_genes_set.add(g)
    train_pert_gene_indices = sorted(vocab[g] for g in train_pert_genes_set)
    print(f"[audit_rpe1] train pert genes: {len(train_pert_gene_indices)}")

    train_perturbed_cell_ids = []
    n_pert_arr = obs['n_pert'].values
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)
    print(f"[audit_rpe1] train perturbed cells: {len(train_perturbed_cell_ids)}")

    rng = np.random.default_rng(args.seed)
    train_embed_pool = model.pert_encoder.embed.weight.data[train_pert_gene_indices].clone()
    ctrl_to_row = {c: i for i, c in enumerate(obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in split['ctrl_cells']])
    X = adata.X
    is_sparse = hasattr(X, "toarray")
    def get_X_rows(rows):
        if is_sparse: return np.asarray(X[rows].toarray()).astype(np.float32)
        return np.asarray(X[rows]).astype(np.float32)
    ctrl_X = get_X_rows(ctrl_rows)
    train_pert_rows = np.array([ctrl_to_row[c] for c in train_perturbed_cell_ids])
    tpm = get_X_rows(train_pert_rows).mean(0)
    train_pert_mean = torch.from_numpy(tpm)

    rows_out = []
    for tg in split['test_genes']:
        mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg])
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5:
            continue
        pert_X = get_X_rows(pert_rows)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=200)
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)
        obs_delta_top = obs_delta_full[deg_idx]

        basal_rows = rng.choice(ctrl_rows, size=args.n_per_pair, replace=True)
        basal = torch.from_numpy(get_X_rows(basal_rows))
        idx = vocab.get(tg, 0)
        pert_idx = torch.tensor([[idx, 0]] * args.n_per_pair, dtype=torch.long)
        with torch.no_grad():
            for mode in ("learned", "mean", "zero", "random", "identity", "pop_mean"):
                x_hat = predict_under_mode(model, basal, pert_idx, mode, train_embed_pool, rng,
                                           train_pert_mean_x=train_pert_mean).numpy()
                pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
                pred_delta_top = pred_delta[deg_idx]
                rho = spearmanr(pred_delta_top, obs_delta_top).statistic
                num = np.dot(pred_delta - pred_delta.mean(), obs_delta_full - obs_delta_full.mean())
                denom = (np.linalg.norm(pred_delta - pred_delta.mean()) *
                         np.linalg.norm(obs_delta_full - obs_delta_full.mean()) + 1e-12)
                pearson = float(num / denom)
                mae = float(np.mean(np.abs(pred_delta_top - obs_delta_top)))
                mse = float(((x_hat - obs_pert_mean[None]) ** 2).mean())
                rows_out.append({
                    "test_gene": tg, "mode": mode,
                    "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                    "Pearson_delta_full": pearson,
                    "mae_delta_top200": mae, "mse_to_pert_mean": mse,
                    "n_obs_pert": int(len(pert_rows)),
                })
    df = pd.DataFrame(rows_out)
    out_csv = out / f"audit_{args.out_tag}_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    summary = df.groupby('mode').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        DE_Spearman_median=('DE_Spearman', 'median'),
        DE_Spearman_sem=('DE_Spearman', lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        Pearson_full_mean=('Pearson_delta_full', 'mean'),
        n_genes=('test_gene', 'count'),
    ).round(4)
    print("\n=== RPE1 audit summary ===")
    print(summary)
    summary.to_csv(out / f"audit_summary_{args.out_tag}_seed{args.seed}.csv")

if __name__ == "__main__":
    main()
