#!/usr/bin/env python3
"""Train the GestureTCN on N2 shards (Phase E, N3).

    python scripts/neural_train.py data/gesture_shards --out models/gesture/tcn_v1.pt
    python scripts/neural_train.py --smoke        # synthetic end-to-end check

Loss = MSE on targets + a velocity-matching term (the texture must MOVE
like the data, not just sit at its mean — plain MSE regresses gesture to
mush; velocity matching is the cheap defence). Eval prints the two
metrics the plan commits to before any robot time:
  * velocity distribution ratio vs data (want ~1, mush -> ~0)
  * beat alignment: lag of the peak audio-energy/motion-speed
    cross-correlation (want |lag| <= 150 ms)
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

    vel_pred, vel_true, lags = [], [], []
    with torch.no_grad():
        for x, y in zip(xs, ys):
            p = model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
            vp = np.abs(np.diff(p[:, :10], axis=0)).mean()
            vt = np.abs(np.diff(y[:, :10], axis=0)).mean()
            vel_pred.append(vp)
            vel_true.append(vt)
            speed = np.abs(np.diff(p[:, :10], axis=0)).mean(axis=1)
            e = x[1:, 0] - x[1:, 0].mean()
            s = speed - speed.mean()
            if e.std() > 1e-6 and s.std() > 1e-6:
                xc = np.correlate(s, e, mode="full")
                lag = (np.argmax(xc) - (len(e) - 1)) / FRAME_HZ
                lags.append(lag)
    vr = (np.mean(vel_pred) / max(np.mean(vel_true), 1e-9))
    return {"velocity_ratio": round(float(vr), 3),
            "beat_lag_ms": round(float(np.median(lags)) * 1000.0, 0)
            if lags else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="?", default=None)
    ap.add_argument("--out", default="models/gesture/tcn_v1.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
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

    model = build_model().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"GestureTCN: {n_par/1e6:.2f} M params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    gen = _windows(tx, ty, rng)
    model.train()
    for step in range(1, a.steps + 1):
        batch = [next(gen) for _ in range(8)]
        L = min(len(b[0]) for b in batch)
        x = torch.from_numpy(np.stack([b[0][:L] for b in batch])).to(device)
        y = torch.from_numpy(np.stack([b[1][:L] for b in batch])).to(device)
        p = model(x)
        mse = torch.nn.functional.mse_loss(p, y)
        vel = torch.nn.functional.l1_loss(
            (p[:, 1:] - p[:, :-1]).abs().mean(),
            (y[:, 1:] - y[:, :-1]).abs().mean())
        loss = mse + 2.0 * vel
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, a.steps // 10) == 0:
            print(f"  step {step:5d}  mse {mse.item():.4f}  "
                  f"vel {vel.item():.4f}")
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
