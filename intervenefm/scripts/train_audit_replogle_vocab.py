"""Replogle vocabulary-size sweep: train on N ∈ {100, 500, 2000} perturbations.

Tests the reviewer's critique that the Norman/Replogle inversion is hand-waved.
Mechanistic prediction: gap (learned − pop_mean) declines with N, because each
embedding gets fewer training cells. If the gap-vs-N curve crosses zero at the
right N, we have a unified one-parameter explanation for the Norman vs Replogle
divergence.

For each vocab cap N: keep only the top-N most frequent training perturbation
genes; relabel less-common perturbations as held-out (cells dropped from training,
but still available for the audit if N tracking is needed). Vocab is reduced
accordingly. 12 epochs to keep runtime manageable.
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

from src.data_replogle import load_replogle, build_split_replogle, build_gene_vocab_replogle
from src.data_norman import PerturbSeqDataset
from src.cpa_minimal import CPAMinimal, CPAConfig
from src.audit import predict_under_mode, get_top_deg_indices
from scipy.stats import spearmanr


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_cells", type=int, default=80000)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_train_perts", type=int, required=True,
                    help="Cap training pert vocab to top-N most frequent genes")
    ap.add_argument("--out_tag", default="replogle_vocab")
    ap.add_argument("--n_per_pair", type=int, default=200)
    ap.add_argument("--max_test_genes", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[vocab_sweep] {vars(args)}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    t0 = time.time()
    adata = load_replogle(n_top_hvg=args.hvg, max_cells=args.max_cells, seed=args.seed)
    print(f"[vocab_sweep] adata {adata.shape} in {time.time()-t0:.1f}s")
    full_vocab = build_gene_vocab_replogle(adata)

    # Count cells per pert gene; pick top-N as the trainable vocab
    obs = adata.obs
    pg = obs['pert_genes'].values
    counts = {}
    for genes in pg:
        for g in genes:
            counts[g] = counts.get(g, 0) + 1
    eligible = sorted(counts.items(), key=lambda kv: -kv[1])
    eligible_genes = [g for g, _ in eligible if counts[g] >= 50]
    print(f"[vocab_sweep] eligible (>=50 cells): {len(eligible_genes)}")

    rng = np.random.default_rng(args.seed)
    # Fix test genes deterministically across vocabulary sizes:
    # take a fixed last `max_test_genes` from eligible (least-frequent eligible),
    # to ensure they are NOT in any candidate train set.
    test_genes = list(eligible_genes[-args.max_test_genes:])
    train_candidates = eligible_genes[:-args.max_test_genes]
    if args.n_train_perts >= len(train_candidates):
        print(f"[vocab_sweep] capping n_train_perts to {len(train_candidates)}")
        args.n_train_perts = max(1, len(train_candidates))
    train_pert_genes = list(train_candidates[: args.n_train_perts])
    train_set_perts = set(train_pert_genes); test_set = set(test_genes)
    print(f"[vocab_sweep] train_perts={len(train_pert_genes)} test_genes={len(test_genes)}")

    # Build train/test split
    train_cells, test_cells, ctrl_cells = [], [], []
    n_pert_arr = obs['n_pert'].values
    for ci, cell_id in enumerate(obs.index):
        if n_pert_arr[ci] == 0:
            train_cells.append(cell_id); ctrl_cells.append(cell_id)
        else:
            g = pg[ci][0] if pg[ci] else None
            if g in test_set:
                test_cells.append(cell_id)
            elif g in train_set_perts:
                train_cells.append(cell_id)
            # else: drop (perturbation outside both sets)
    print(f"[vocab_sweep] train: {len(train_cells)}, test: {len(test_cells)}, ctrl: {len(ctrl_cells)}")

    # Reduce vocab to train_pert_genes only (test genes get index 0 = pad/control,
    # so the model has no per-gene id at test time — exactly matches the GEARS 0/1 setup
    # for these caps. We use full_vocab IDs for the embedding table size).
    # Use full_vocab for the model (so train_pert_gene_indices are valid)
    train_ds = PerturbSeqDataset(adata, train_cells, ctrl_cells, full_vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    cfg = CPAConfig(n_genes=adata.n_vars, n_pert_genes=len(full_vocab),
                    z_dim=64, pert_dim=32, hidden=256)
    model = CPAMinimal(cfg)
    print(f"[vocab_sweep] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
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
    pd.DataFrame(train_log).to_csv(
        out / f"training_log_{args.out_tag}_n{args.n_train_perts}_seed{args.seed}.csv", index=False)

    train_pert_gene_indices = sorted(full_vocab[g] for g in train_pert_genes if g in full_vocab)
    train_perturbed_cell_ids = []
    for ci, cell_id in enumerate(obs.index):
        if cell_id in set(train_cells) and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)
    print(f"[audit] train_pert genes={len(train_pert_gene_indices)} train_pert_cells={len(train_perturbed_cell_ids)}")

    rng = np.random.default_rng(args.seed)
    train_embed_pool = model.pert_encoder.embed.weight.data[train_pert_gene_indices].clone()
    ctrl_to_row = {c: i for i, c in enumerate(obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in ctrl_cells])
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
    for tg in test_genes:
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
        # Test gene NOT in trained vocab — index lookup gets 0 (pad). So 'learned' for
        # test genes uses the pad vector — which IS what GEARS 0/1 actually does at
        # eval (no embedding for unseen gene). This is honest about the training regime.
        idx = full_vocab.get(tg, 0)
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
                    "n_train_perts": args.n_train_perts,
                })
    df = pd.DataFrame(rows_out)
    out_csv = out / f"audit_{args.out_tag}_n{args.n_train_perts}_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    summary = df.groupby('mode').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        DE_Spearman_median=('DE_Spearman', 'median'),
        Pearson_full_mean=('Pearson_delta_full', 'mean'),
        n_genes=('test_gene', 'count'),
    ).round(4)
    print("\n=== Vocab-sweep audit summary ===")
    print(summary)
    summary.to_csv(out / f"audit_summary_{args.out_tag}_n{args.n_train_perts}_seed{args.seed}.csv")

if __name__ == "__main__":
    main()
