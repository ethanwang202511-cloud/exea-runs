"""Decoder-FT comparator: init decoder from a Stage-1 ckpt (trained on
5 source species), then full-fine-tune ALL decoder params on the
held-out species' train data. This is the "SuperAnimal-style full-FT"
H1 comparator (architecturally analogous: pretrain on N source species,
fine-tune on the target species).

In contrast to scripts/run_within_species_pilot.py with --no-id-token
(which trains the decoder from scratch), this script INITIALIZES from a
specified checkpoint, which gives a much fairer comparator for the H1
parameter-efficiency claim.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset, collate_cached
from iti.eval.pck import per_keypoint_distance_normalized, pck_metrics, bootstrap_ci
from iti.model.iti import ITIConfig, ITIModel
from iti.train.trainer import TrainConfig, evaluate, coord_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("decoder_ft_from_init")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init-ckpt", type=Path,
                   default=ROOT / "results/q2_within_species/iti_xattn_aux01_seed0/model.pt")
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_all")
    p.add_argument("--held-out-species", type=str, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-token", action="store_true",
                   help="zero+freeze the per-id token (so the model fine-tunes only the decoder)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state = torch.load(args.init_ckpt, map_location=args.device, weights_only=False)
    cfg = ITIConfig(**{k: v for k, v in state["cfg"].items()
                       if k in ITIConfig.__dataclass_fields__})
    model = ITIModel(cfg).to(args.device)
    model.load_state_dict(state["model_state_dict"])
    logger.info("Loaded ckpt %s", args.init_ckpt)

    train_full = CachedFeatureDataset(args.train_cache)
    val_full = CachedFeatureDataset(args.val_cache)
    name_to_gid = {r["species_name"]: r["identity_id"] for r in train_full.records}
    gid = name_to_gid[args.held_out_species]
    train_held = CachedFeatureDataset(args.train_cache, identity_subset={gid})
    val_held = CachedFeatureDataset(args.val_cache, identity_subset={gid})
    logger.info("Held-out %s: %d train, %d val", args.held_out_species,
                len(train_held), len(val_held))

    if args.zero_token:
        with torch.no_grad():
            model.tokens.emb.weight.zero_()
        model.tokens.emb.weight.requires_grad_(False)
        logger.info("zero+freeze token; effectively decoder-FT only")

    # Map the held-out species into the existing token bank's dim 0 (use it as a
    # "synthetic novel id" so the model still has its FiLM/attention path active).
    # We pick local id 0 (overrides the 'dog' token slot for evaluation purposes).
    REMAP_LOCAL = 0
    cfg_train = TrainConfig(
        n_steps=args.n_steps, batch_size=32, lr=args.lr, device=args.device,
        seed=args.seed, warmup_steps=20,
    )

    # Build train/val datasets that REMAP the held-out gid -> 0
    train_held_remap = CachedFeatureDataset(args.train_cache,
                                            identity_subset={gid},
                                            identity_remap={gid: REMAP_LOCAL})
    val_held_remap = CachedFeatureDataset(args.val_cache,
                                          identity_subset={gid},
                                          identity_remap={gid: REMAP_LOCAL})
    train_loader = DataLoader(train_held_remap, batch_size=cfg_train.batch_size,
                              shuffle=True, collate_fn=collate_cached, drop_last=True)
    val_loader = DataLoader(val_held_remap, batch_size=cfg_train.batch_size,
                            shuffle=False, collate_fn=collate_cached)

    pre = evaluate(model, val_loader, cfg_train)
    logger.info("PRE-FT (using local-id-0 token from source ckpt): %s", pre)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("trainable params: %d (out of %d)", n_trainable,
                sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(trainable, lr=cfg_train.lr, weight_decay=0.01)

    def lr_at(s):
        if s < cfg_train.warmup_steps:
            return cfg_train.lr * (s + 1) / cfg_train.warmup_steps
        prog = (s - cfg_train.warmup_steps) / max(cfg_train.n_steps - cfg_train.warmup_steps, 1)
        return cfg_train.lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * prog)))

    step = 0
    t0 = time.time()
    while step < cfg_train.n_steps:
        for batch in train_loader:
            if step >= cfg_train.n_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            feat = batch["feature"].to(args.device)
            ids = batch["identity_id"].to(args.device)
            kp = batch["keypoints_xy"].to(args.device)
            vis = batch["vis"].to(args.device)
            model.train()
            pred = model.forward_from_features(feat, ids)
            loss = coord_loss(pred, kp, vis, out_size=224)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if step % 50 == 0:
                logger.info("ft step %4d/%d loss=%.4e lr=%.2e (%.1fs)",
                            step, cfg_train.n_steps, loss.item(), lr_at(step), time.time() - t0)
            step += 1

    final = evaluate(model, val_loader, cfg_train)
    logger.info("POST-FT: %s", final)

    out = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pre_ft": pre, "post_ft": final, "n_trainable": n_trainable,
        "elapsed_s": time.time() - t0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=2))

    # Append a CSV row for comparison
    csv_path = ROOT / "results" / "decoder_ft_comparator.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a") as f:
        if write_header:
            f.write("species,init_ckpt,n_trainable,n_train_ex,n_val_ex,"
                    "pre_rmse_norm,post_rmse_norm,pre_pck_at_005,post_pck_at_005\n")
        f.write(f"{args.held_out_species},{args.init_ckpt.name},{n_trainable},"
                f"{len(train_held_remap)},{len(val_held_remap)},"
                f"{pre['rmse_norm']:.5f},{final['rmse_norm']:.5f},"
                f"{pre['pck@0.05']:.4f},{final['pck@0.05']:.4f}\n")


if __name__ == "__main__":
    main()
