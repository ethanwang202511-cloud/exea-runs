"""Sanity-check the data pipeline before training."""
import sys
sys.path.insert(0, "src")
from data_norman import load_norman, build_split, build_gene_vocab, PerturbSeqDataset
from torch.utils.data import DataLoader

print("[sanity] loading Norman...")
adata = load_norman(n_top_hvg=2000, max_cells=50000)  # small for sanity
print(f"  shape: {adata.shape}")
print(f"  X dtype after norm/log: {adata.X.dtype}")

vocab = build_gene_vocab(adata)
print(f"  pert-gene vocab size: {len(vocab)}")

split = build_split(adata, kind="0/2")
print(f"  train cells: {len(split['train_cells'])}, test cells: {len(split['test_cells'])}, "
      f"test pairs: {len(split['test_pairs'])}")

train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2)
test_ds = PerturbSeqDataset(adata, split['test_cells'], split['ctrl_cells'], vocab, max_perts=2)

print(f"  train ds: {len(train_ds)}, test ds: {len(test_ds)}")

# Pull a batch
loader = DataLoader(train_ds, batch_size=8, shuffle=True)
basal, tgt, pidx = next(iter(loader))
print(f"  batch basal: {basal.shape}, target: {tgt.shape}, pidx: {pidx.shape}")
print(f"  basal range: [{basal.min():.2f}, {basal.max():.2f}]")
print(f"  pidx examples: {pidx.tolist()}")
