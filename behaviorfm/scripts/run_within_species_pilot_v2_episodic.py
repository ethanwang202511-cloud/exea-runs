"""Within-species pilot WITH Stage-2 episodic meta-loss.

Training:
  Stage 1 (joint task): n_warmup_steps of standard pose loss to give the
    decoder a foothold (otherwise the meta-loss starts from random and
    collapses). We initialize from the saved Stage-1 checkpoint to keep
    things efficient.
  Stage 2 (episodic): n_episodic_steps of episodic-meta-loss; the
    loss-amplifier on the per-id token contribution.

Saves a new checkpoint and runs evaluation on val (training species).
The held-out species (giraffe) evaluation is still done by
run_h1_mini_pilot.py against the new checkpoint.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset, collate_cached
from iti.eval.pck import per_keypoint_distance_normalized, pck_metrics, bootstrap_ci
from iti.model.iti import ITIConfig, ITIModel
from iti.train.episodic import EpisodicTrainConfig, episodic_step, sample_episode_indices
from iti.train.trainer import TrainConfig, train_within_species, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("episodic_pilot")


def evaluate_using_other_species_token(
    model: ITIModel, val_loader: DataLoader, train_local_ids: list[int],
    cfg: TrainConfig, k_support_per_species: int = 5,
    train_features_per_local_id: dict[int, torch.Tensor] | None = None,
    train_kp_per_local_id: dict[int, torch.Tensor] | None = None,
    train_vis_per_local_id: dict[int, torch.Tensor] | None = None,
    n_inner: int = 5, inner_lr: float = 1.0,
) -> dict:
    """For each training species, simulate held-out adaptation: drop its trained
    token, init via interpolation over the OTHER species' tokens, do n_inner SGD
    steps on a few train-set support examples, evaluate on val. This gives us a
    lower-bound on H1 mechanism quality for species the model has actually seen
    (so it isolates the "is the adaptation pipeline working" question from the
    "did the decoder generalize" question).
    """
    model.eval()
    rng = np.random.default_rng(0)
    out = {}
    for local_id in train_local_ids:
        feats = train_features_per_local_id[local_id]
        kp = train_kp_per_local_id[local_id]
        vis = train_vis_per_local_id[local_id]
        idx = rng.choice(len(feats), size=min(k_support_per_species, len(feats)), replace=False)
        sup_f = feats[idx].to(cfg.device)
        sup_kp = kp[idx].to(cfg.device)
        sup_vis = vis[idx].to(cfg.device)

        # Build "other species" token bank.
        other_ids = [i for i in train_local_ids if i != local_id]
        with torch.no_grad():
            other_tokens = model.tokens.emb.weight[other_ids]
            ex_feat = sup_f.mean(dim=1, keepdim=False).unsqueeze(0)
            tok_init, _ = model.interpolation_head(ex_feat, other_tokens)
            tok = tok_init.squeeze(0).detach().clone()

        # Inner SGD (token-only)
        tok = torch.nn.Parameter(tok)
        opt = torch.optim.SGD([tok], lr=inner_lr, momentum=0.9)
        for _ in range(n_inner):
            B = sup_f.shape[0]
            pred = model.forward_with_token(sup_f, tok.unsqueeze(0).expand(B, -1))
            target = sup_kp / cfg.out_size
            per_kp = ((pred - target) ** 2).sum(dim=-1)
            loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        adapted = tok.detach()

        # Val on this species using the adapted token
        val_dists, val_diags = [], []
        for batch in val_loader:
            mask = (batch["identity_id"] == local_id)
            if not mask.any():
                continue
            sel_idx = mask.nonzero(as_tuple=True)[0]
            sel_feat = batch["feature"][sel_idx].to(cfg.device)
            B = sel_feat.shape[0]
            with torch.no_grad():
                pred_unit = model.forward_with_token(
                    sel_feat, adapted.unsqueeze(0).expand(B, -1)
                ).cpu()
            pred_xy = pred_unit * cfg.out_size
            sel_kp = batch["keypoints_xy"][sel_idx].numpy()
            sel_vis = batch["vis"][sel_idx].numpy()
            sel_side = batch["crop_side"][sel_idx].numpy()
            sel_diag = batch["bbox_diag_orig"][sel_idx].numpy()
            d = per_keypoint_distance_normalized(
                pred_xy.numpy(), sel_kp, sel_vis, sel_side, out_size=cfg.out_size,
            )
            val_dists.append(d); val_diags.append(sel_diag)
        if val_dists:
            d = np.concatenate(val_dists); diag = np.concatenate(val_diags)
            m = pck_metrics(d, diag)
            out[local_id] = m
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", type=Path,
                   default=ROOT / "results/within_species_pilot/iti_full_seed0/model.pt",
                   help="Stage-1 checkpoint to initialize from.")
    p.add_argument("--train-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_train_split1_pilot6")
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_pilot6")
    p.add_argument("--training-species", type=str,
                   default="dog,sheep,cat,antelope,zebra")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results/within_species_pilot_episodic")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodic-steps", type=int, default=300)
    p.add_argument("--k-support", type=int, default=5)
    p.add_argument("--q-query", type=int, default=8)
    p.add_argument("--n-inner", type=int, default=3)
    p.add_argument("--inner-lr", type=float, default=1.0)
    p.add_argument("--outer-lr", type=float, default=1e-4)
    p.add_argument("--interp-supervision-weight", type=float, default=0.0,
                   help="Weight on the explicit interp-head supervision aux loss")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_full = CachedFeatureDataset(args.train_cache)
    val_full = CachedFeatureDataset(args.val_cache)
    train_names = [s.strip() for s in args.training_species.split(",")]
    name_to_global = {r["species_name"]: r["identity_id"] for r in train_full.records}
    global_ids_training = [name_to_global[n] for n in train_names]
    remap = {gid: i for i, gid in enumerate(global_ids_training)}
    n_training_identities = len(remap)

    train_ds = CachedFeatureDataset(args.train_cache,
                                    identity_subset=set(global_ids_training),
                                    identity_remap=remap)
    val_ds = CachedFeatureDataset(args.val_cache,
                                  identity_subset=set(global_ids_training),
                                  identity_remap=remap)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_cached)

    # Load Stage-1 checkpoint
    state = torch.load(args.init_from, map_location=args.device, weights_only=False)
    cfg_dict = state["cfg"]
    cfg = ITIConfig(**{k: v for k, v in cfg_dict.items()
                       if k in ITIConfig.__dataclass_fields__})
    model = ITIModel(cfg).to(args.device)
    model.load_state_dict(state["model_state_dict"])
    logger.info("Loaded Stage-1 checkpoint from %s", args.init_from)

    train_cfg_obj = TrainConfig(device=args.device, out_size=224)
    pre_eval = evaluate(model, val_loader, train_cfg_obj)
    logger.info("Stage-1 val (within-species): %s", pre_eval)

    # Build per-species index lists (lightweight). Features are loaded
    # lazily per sampled episode to keep memory bounded.
    species_idx: dict[int, list[int]] = {}
    for i in range(len(train_ds)):
        item = train_ds[i]
        species_idx.setdefault(int(item["identity_id"]), []).append(i)
    species_idx_np = {k: np.asarray(v, dtype=np.int64) for k, v in species_idx.items()}
    logger.info("Per-species sizes: %s", {k: len(v) for k, v in species_idx_np.items()})

    def fetch(local_id: int, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds_idx = species_idx_np[local_id][indices]
        feats, kps, viss = [], [], []
        for di in ds_idx:
            it = train_ds[int(di)]
            feats.append(it["feature"]); kps.append(it["keypoints_xy"]); viss.append(it["vis"])
        return torch.stack(feats), torch.stack(kps), torch.stack(viss)

    # Stage-2 episodic training
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.outer_lr, weight_decay=0.01)
    history = []
    t0 = time.time()
    for step in range(args.n_episodic_steps):
        held = int(rng.integers(0, n_training_identities))
        # sample (k+q) examples from this species at random indices
        n_sp = len(species_idx_np[held])
        if n_sp >= args.k_support + args.q_query:
            sel = rng.choice(n_sp, size=args.k_support + args.q_query, replace=False)
        else:
            sel = rng.choice(n_sp, size=args.k_support + args.q_query, replace=True)
        s_local = sel[: args.k_support]; q_local = sel[args.k_support : args.k_support + args.q_query]
        sup_f, sup_kp, sup_vis = fetch(held, s_local)
        qry_f, qry_kp, qry_vis = fetch(held, q_local)
        sup_f = sup_f.to(args.device); sup_kp = sup_kp.to(args.device); sup_vis = sup_vis.to(args.device)
        qry_f = qry_f.to(args.device); qry_kp = qry_kp.to(args.device); qry_vis = qry_vis.to(args.device)

        model.train()
        loss_out, interp_aux = episodic_step(
            model, held_out_local_id=int(held),
            support_features=sup_f, support_kp=sup_kp, support_vis=sup_vis,
            query_features=qry_f, query_kp=qry_kp, query_vis=qry_vis,
            n_inner=args.n_inner, inner_lr=args.inner_lr, out_size=224,
            use_interpolation=True,
        )
        loss = loss_out
        if args.interp_supervision_weight > 0.0 and interp_aux is not None:
            loss = loss + args.interp_supervision_weight * interp_aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 25 == 0:
            aux_str = f"  aux={interp_aux.item():.4e}" if interp_aux is not None else ""
            logger.info("episodic step %4d/%d  loss=%.4e  out=%.4e%s  (%.1fs)",
                        step, args.n_episodic_steps, loss.item(), loss_out.item(),
                        aux_str, time.time() - t0)
        if step > 0 and step % 75 == 0:
            mid_eval = evaluate(model, val_loader, train_cfg_obj)
            logger.info("MID-EPISODIC VAL: rmse_norm=%.4f pck@0.05=%.3f",
                        mid_eval["rmse_norm"], mid_eval["pck@0.05"])
            history.append({"step": step, **mid_eval})

    final = evaluate(model, val_loader, train_cfg_obj)
    logger.info("Stage-2 final val (within-species): %s", final)
    history.append({"step": args.n_episodic_steps, **final})

    # H1 simulation: per-species adaptation (drop trained token, init via interp head, inner SGD)
    species_feats_lite, species_kp_lite, species_vis_lite = {}, {}, {}
    for lid, idxs in species_idx_np.items():
        # cap to first 16 examples per species to bound memory
        cap_idx = idxs[: min(16, len(idxs))]
        feats, kps, viss = [], [], []
        for di in cap_idx:
            it = train_ds[int(di)]
            feats.append(it["feature"]); kps.append(it["keypoints_xy"]); viss.append(it["vis"])
        species_feats_lite[lid] = torch.stack(feats)
        species_kp_lite[lid] = torch.stack(kps)
        species_vis_lite[lid] = torch.stack(viss)
    sim = evaluate_using_other_species_token(
        model, val_loader, list(range(n_training_identities)), train_cfg_obj,
        k_support_per_species=args.k_support,
        train_features_per_local_id=species_feats_lite,
        train_kp_per_local_id=species_kp_lite,
        train_vis_per_local_id=species_vis_lite,
        n_inner=10, inner_lr=args.inner_lr,
    )
    logger.info("Adapt-on-training-species (H1 simulation):")
    for lid, m in sim.items():
        logger.info("  local %d (%s):  rmse_norm=%.4f  pck@0.05=%.3f",
                    lid, train_names[lid], m["rmse_norm"], m["pck@0.05"])

    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "stage1_val": pre_eval,
        "stage2_history": history,
        "stage2_final": final,
        "h1_simulation_per_species": sim,
        "training_species": train_names,
        "elapsed_s": time.time() - t0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    ckpt = args.out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "train_species_names": train_names,
        "global_id_to_local": remap,
    }, ckpt)
    logger.info("Saved %s and %s", args.out_dir / "summary.json", ckpt)


if __name__ == "__main__":
    main()
