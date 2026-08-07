"""
Full pipeline: load Norman, build 0/2 split, train minimal CPA, run intervention-gap audit.

Outputs:
  results/audit_norman_02_seed{seed}.csv  (per-pair × per-mode DE-Spearman ρ)
  results/audit_summary_seed{seed}.csv    (aggregated)
  results/training_log_seed{seed}.csv     (epoch / loss curve)

Run:
  python3 scripts/train_and_audit.py --seed 0 --max_cells 60000 --hvg 2000 --epochs 20
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_norman import load_norman, build_split, build_gene_vocab, PerturbSeqDataset
from src.cpa_minimal import CPAMinimal, CPAConfig, gaussian_nll_loss
from src.audit import run_audit


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_cells", type=int, default=60000)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--pert_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--split", default="0/2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_tag", default="cpa_min")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[run] {vars(args)}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # === Data ===
    t0 = time.time()
    adata = load_norman(n_top_hvg=args.hvg, max_cells=args.max_cells, seed=args.seed)
    vocab = build_gene_vocab(adata)
    split = build_split(adata, kind=args.split, seed=args.seed)
    print(f"[run] data ready in {time.time()-t0:.1f}s. vocab={len(vocab)}, train={len(split['train_cells'])}, test={len(split['test_cells'])}")

    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # === Model ===
    cfg = CPAConfig(
        n_genes=adata.n_vars, n_pert_genes=len(vocab),
        z_dim=args.z_dim, pert_dim=args.pert_dim, hidden=args.hidden,
    )
    model = CPAMinimal(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] CPAMinimal params: {n_params/1e6:.2f}M  (cfg={cfg})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # === Train ===
    train_log = []
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0
        for basal, target, pidx in train_loader:
            basal = basal.to(args.device); target = target.to(args.device); pidx = pidx.to(args.device)
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * basal.shape[0]; ep_n += basal.shape[0]
        avg_loss = ep_loss / ep_n
        print(f"[train] epoch {epoch+1}/{args.epochs}  mse={avg_loss:.4f}  ({time.time()-t0:.1f}s)")
        train_log.append({"epoch": epoch+1, "mse": avg_loss, "elapsed_s": time.time()-t0})

    # Save training log
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"training_log_{args.out_tag}_seed{args.seed}.csv"
    pd.DataFrame(train_log).to_csv(log_path, index=False)
    print(f"[saved] {log_path}")
    # Save trained checkpoint for L2-norm diagnostic
    torch.save({"state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "vocab": vocab,
                "split": split},
               out_dir / f"model_{args.out_tag}_seed{args.seed}.pt")

    # === Audit ===
    train_pert_genes_set = set(split['train_single_genes'])
    # Add genes that appear in train doubles (which are in train_cells)
    obs = adata.obs
    pert_genes_arr = obs['pert_genes'].values
    train_cell_set = set(split['train_cells'])
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_cell_set and obs['n_pert'].iloc[ci] >= 1:
            for g in pert_genes_arr[ci]:
                if g in vocab:
                    train_pert_genes_set.add(g)
    train_pert_gene_indices = sorted(vocab[g] for g in train_pert_genes_set if g in vocab)
    print(f"[audit] {len(train_pert_gene_indices)} pert-genes seen during training")

    # Build training-perturbed cell list for the pop_mean baseline
    train_perturbed_cell_ids = []
    train_cell_set = set(split['train_cells'])
    n_pert_arr = adata.obs['n_pert'].values
    for ci, cell_id in enumerate(adata.obs.index):
        if cell_id in train_cell_set and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)
    print(f"[audit] {len(train_perturbed_cell_ids)} training perturbed cells available for pop_mean baseline")

    df = run_audit(
        model, adata,
        test_pairs=split['test_pairs'],
        ctrl_cell_ids=split['ctrl_cells'],
        gene_vocab=vocab,
        train_pert_gene_indices=train_pert_gene_indices,
        n_per_pair=200,
        seed=args.seed,
        train_perturbed_cell_ids=train_perturbed_cell_ids,
    )
    audit_path = out_dir / f"audit_{args.out_tag}_norman{args.split.replace('/', '')}_seed{args.seed}.csv"
    df.to_csv(audit_path, index=False)
    print(f"[saved] {audit_path}")

    # === Summary ===
    print("\n=== Audit summary ===")
    summary = df.groupby('mode').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        DE_Spearman_median=('DE_Spearman', 'median'),
        DE_Spearman_sem=('DE_Spearman', lambda s: s.std() / np.sqrt(len(s))),
        n_pairs=('pair', 'count'),
    ).round(4)
    print(summary)
    summary_path = out_dir / f"audit_summary_{args.out_tag}_norman{args.split.replace('/', '')}_seed{args.seed}.csv"
    summary.to_csv(summary_path)
    print(f"[saved] {summary_path}")

    # Compute key gaps
    means = summary['DE_Spearman_mean']
    print(f"\nLearned vs Mean ablation gap: {means.get('learned',np.nan) - means.get('mean',np.nan):+.4f}")
    print(f"Learned vs Zero ablation gap : {means.get('learned',np.nan) - means.get('zero',np.nan):+.4f}")
    print(f"Learned vs Random ablation gap: {means.get('learned',np.nan) - means.get('random',np.nan):+.4f}")

if __name__ == "__main__":
    main()
