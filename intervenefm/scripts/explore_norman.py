"""Quick exploration of the Norman 2019 AnnData file."""
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
adata = sc.read_h5ad(ROOT / "data" / "norman_2019.h5ad")

print("=== Shape ===")
print(adata.shape, "(cells x genes)")
print()
print("=== obs columns ===")
print(list(adata.obs.columns))
print()
print("=== obs head ===")
print(adata.obs.head())
print()

# Find the perturbation column
candidates = [c for c in adata.obs.columns if 'pert' in c.lower() or 'condition' in c.lower() or 'guide' in c.lower() or 'target' in c.lower() or 'gene' in c.lower()]
print(f"Pert-related columns: {candidates}")
print()

for c in candidates[:5]:
    vals = adata.obs[c].astype(str)
    n_unique = vals.nunique()
    print(f"  {c}: {n_unique} unique values, examples: {list(vals.unique()[:5])}")

print()
print("=== layers ===")
print(list(adata.layers.keys()) if hasattr(adata, 'layers') else "no layers")
print(f"X dtype: {adata.X.dtype}, X min/max/mean: {adata.X.min():.3f}/{adata.X.max():.3f}/{adata.X.mean():.3f}")
print()
print("=== var head ===")
print(adata.var.head())
