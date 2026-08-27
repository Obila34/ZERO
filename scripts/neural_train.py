#!/usr/bin/env python3
"""Train the GestureTCN on N2 shards (Phase E, N3).

    python scripts/neural_train.py data/gesture_shards --out models/gesture/tcn_v1.pt
    python scripts/neural_train.py --smoke        # synthetic end-to-end check

Loss = MSE on targets + PER-FRAME velocity L1 + PER-FRAME acceleration L1
+ a motion-energy match (closures/wrists separately). The per-frame terms
teach WHEN and HOW motion happens (v1/v2's scalar-only match couldn't:
0.63x human speed); the energy term forbids the degenerate answer of
stillness-when-unsure the per-frame terms alone reward (0.11x without
it). A per-sequence noise latent lets the model commit to one style
instead of averaging styles away. Eval prints the metrics the
plan commits to before any robot time:
  * velocity distribution ratio vs data (want ~1, mush -> ~0)
  * beat alignment: lag of the peak audio-energy/motion-speed
    cross-correlation (want |lag| <= 150 ms)
  * L/R asymmetry: left-right closure correlation (BEAT2 ground truth
    ~0.04 — near 1.0 means the model mirrors, humans don't)
  * wrist velocity ratio vs data (the wrists must move too)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from zero.expr.features import FEAT_DIM, FRAME_HZ  # noqa: E402
from zero.expr.model import build_model, save_checkpoint  # noqa: E402
from zero.expr.retarget import TARGET_DIM  # noqa: E402

WIN = 160          # 8 s training windows


def _load_shards(shard_dir: Path):
    xs, ys = [], []
    for f in sorted(shard_dir.glob("*.npz")):
        d = np.load(f)
        xs.append(d["feats"].astype(np.float32))
        ys.append(d["targets"].astype(np.float32))
    return xs, ys


def _synthetic_shards(n_clips=6, n_frames=400, seed=0):
    """Smoke-mode data with a REAL audio->motion correlation to learn:
    energy bursts drive closure with a small lag."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_clips):
        e = np.clip(rng.normal(0, 1, n_frames).cumsum() * 0.05, 0, None)
        e = np.abs(np.sin(np.linspace(0, 20, n_frames))) * \
            rng.uniform(0.5, 1.0)
        x = rng.normal(0, 0.1, (n_frames, FEAT_DIM)).astype(np.float32)
        x[:, 0] = e
        y = np.zeros((n_frames, TARGET_DIM), dtype=np.float32)
        lag = 2
        y[lag:, :10] = np.clip(e[:-lag, None] * 0.8
                               + rng.normal(0, 0.05, (n_frames - lag, 10)),
                               0, 1)
        xs.append(x)
        ys.append(y)
    return xs, ys


def _windows(xs, ys, rng):
    while True:
        i = rng.integers(len(xs))
        x, y = xs[i], ys[i]
        if len(x) <= WIN:
            yield x, y
            continue
        s = rng.integers(len(x) - WIN)
        yield x[s:s + WIN], y[s:s + WIN]


def evaluate(model, xs, ys, device) -> dict:
    import torch

    nd = getattr(model, "noise_dim", 0)
    vel_pred, vel_true, lags = [], [], []
    wvel_pred, wvel_true, asym = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(zip(xs, ys)):
            noise = None
            if nd:
                z = np.random.default_rng(i).normal(
                    size=(1, nd)).astype(np.float32)
                noise = torch.from_numpy(z).to(device)
            p = model(torch.from_numpy(x[None]).to(device),
                      noise)[0].cpu().numpy()
            vel_pred.append(np.abs(np.diff(p[:, :10], axis=0)).mean())
            vel_true.append(np.abs(np.diff(y[:, :10], axis=0)).mean())
            wvel_pred.append(np.abs(np.diff(p[:, 10:], axis=0)).mean())
            wvel_true.append(np.abs(np.diff(y[:, 10:], axis=0)).mean())
            pl, pr = p[:, :5].mean(1), p[:, 5:10].mean(1)
            if pl.std() > 1e-6 and pr.std() > 1e-6:
                asym.append(np.corrcoef(pl, pr)[0, 1])
            speed = np.abs(np.diff(p[:, :10], axis=0)).mean(axis=1)
            e = x[1:, 0] - x[1:, 0].mean()
            s = speed - speed.mean()
            if e.std() > 1e-6 and s.std() > 1e-6:
                xc = np.correlate(s, e, mode="full")
                lag = (np.argmax(xc) - (len(e) - 1)) / FRAME_HZ
                lags.append(lag)
    vr = (np.mean(vel_pred) / max(np.mean(vel_true), 1e-9))
    wvr = (np.mean(wvel_pred) / max(np.mean(wvel_true), 1e-9))
    return {"velocity_ratio": round(float(vr), 3),
            "wrist_velocity_ratio": round(float(wvr), 3),
            "lr_closure_corr": round(float(np.mean(asym)), 3)
            if asym else None,
            "beat_lag_ms": round(float(np.median(lags)) * 1000.0, 0)
            if lags else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="?", default=None)
    ap.add_argument("--out", default="models/gesture/tcn_v1.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vel-weight", type=float, default=1.0,
                    help="per-frame velocity L1 weight (v1/v2's scalar "
                         "mean-match barely helped: 0.63x human velocity)")
    ap.add_argument("--acc-weight", type=float, default=0.5,
                    help="per-frame acceleration L1 weight — sharpness "
                         "of direction changes, the 'snap' of a beat")
    ap.add_argument("--mag-weight", type=float, default=4.0,
                    help="motion-energy matching weight (left hand, "
                         "right hand and wrists each matched "
                         "separately). Per-frame derivative losses "
                         "alone teach stillness when the model is "
                         "unsure — v3's first cut moved at 0.11x "
                         "human speed without this term")
    ap.add_argument("--corr-weight", type=float, default=1.0,
                    help="hand-coordination matching weight: each "
                         "window's predicted L/R closure correlation "
                         "must match that window's HUMAN correlation "
                         "(BEAT2 global ~0.04; v3 drifted to 0.77 — "
                         "the energy term made mirroring the cheap "
                         "way to move)")
    ap.add_argument("--channels", type=int, default=96,
                    help="TCN width (recorded in the checkpoint; "
                         "v1-v3 = 96)")
    ap.add_argument("--noise-dim", type=int, default=8,
                    help="style-latent channels (0 disables); recorded "
                         "in the checkpoint and re-seeded per sentence "
                         "at serve time")
    a = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        xs, ys = _synthetic_shards()
        a.steps = min(a.steps, 300)
    else:
        if not a.shards:
            print("shard dir required (or --smoke)")
            return 1
        xs, ys = _load_shards(Path(a.shards))
        if not xs:
            print("no shards found")
            return 1
    n_val = max(1, len(xs) // 10)
    vx, vy = xs[:n_val], ys[:n_val]
    tx, ty = xs[n_val:] or xs, ys[n_val:] or ys
    print(f"{len(tx)} train / {len(vx)} val clips, device={device}")

    model = build_model(noise_dim=a.noise_dim,
                        channels=a.channels).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"GestureTCN: {n_par/1e6:.2f} M params, noise_dim={a.noise_dim}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    gen = _windows(tx, ty, rng)
    model.train()
    for step in range(1, a.steps + 1):
        batch = [next(gen) for _ in range(8)]
        L = min(len(b[0]) for b in batch)
        x = torch.from_numpy(np.stack([b[0][:L] for b in batch])).to(device)
        y = torch.from_numpy(np.stack([b[1][:L] for b in batch])).to(device)
        noise = (torch.randn(len(batch), a.noise_dim, device=device)
                 if a.noise_dim else None)
        p = model(x, noise)
        mse = torch.nn.functional.mse_loss(p, y)
        # per-frame motion derivatives, not scalar means: the model must
        # move WHEN and HOW FAST the data moves, frame by frame
        dp, dy = p[:, 1:] - p[:, :-1], y[:, 1:] - y[:, :-1]
        vel = torch.nn.functional.l1_loss(dp, dy)
        acc = torch.nn.functional.l1_loss(dp[:, 1:] - dp[:, :-1],
                                          dy[:, 1:] - dy[:, :-1])
        # motion-energy match per item — LEFT, RIGHT and wrists each
        # matched separately (pooled, one hand can hide behind the
        # other; wrists are 2 of 12 dims and would drown): the
        # per-frame terms teach WHEN to move, this forbids stillness
        mag = (torch.nn.functional.l1_loss(
                   dp[..., :5].abs().mean(dim=(1, 2)),
                   dy[..., :5].abs().mean(dim=(1, 2)))
               + torch.nn.functional.l1_loss(
                   dp[..., 5:10].abs().mean(dim=(1, 2)),
                   dy[..., 5:10].abs().mean(dim=(1, 2)))
               + torch.nn.functional.l1_loss(
                   dp[..., 10:].abs().mean(dim=(1, 2)),
                   dy[..., 10:].abs().mean(dim=(1, 2))))
        # hand-coordination match per item: the window's predicted L/R
        # correlation must equal the window's human one — sometimes the
        # hands DO move together; mostly they don't (global ~0.04).
        # Matching per window is honest where a global decorrelation
        # penalty would just be a statistic to game.
        def _corr(u, v, eps=1e-6):
            u = u - u.mean(dim=1, keepdim=True)
            v = v - v.mean(dim=1, keepdim=True)
            return ((u * v).mean(dim=1)
                    / (u.std(dim=1) * v.std(dim=1) + eps))
        cor = torch.nn.functional.mse_loss(
            _corr(p[..., :5].mean(-1), p[..., 5:10].mean(-1)),
            _corr(y[..., :5].mean(-1), y[..., 5:10].mean(-1)))
        loss = (mse + a.vel_weight * vel + a.acc_weight * acc
                + a.mag_weight * mag + a.corr_weight * cor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, a.steps // 10) == 0:
            print(f"  step {step:5d}  mse {mse.item():.4f}  "
                  f"vel {vel.item():.4f}  acc {acc.item():.4f}  "
                  f"mag {mag.item():.4f}  cor {cor.item():.4f}")
    model.eval()
    metrics = evaluate(model, vx, vy, device)
    print(f"\neval: {metrics}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_checkpoint(model, a.out, meta={"metrics": metrics,
                                        "smoke": a.smoke})
    print(f"checkpoint -> {a.out}"
          + ("   (SMOKE model — plumbing only, never serve it as real)"
             if a.smoke else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
