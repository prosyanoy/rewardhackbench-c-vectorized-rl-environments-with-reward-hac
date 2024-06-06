"""Train PPO on a (misspecified) variant and save history + checkpoints.

    python -m rewardhackbench.train --variant sparkle --iters 120 --out runs/sparkle
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from . import VARIANTS
from .ppo import PPOConfig, config_dict, train


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="sparkle")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--threads", type=int, default=int(os.environ.get("RHB_THREADS", 2)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="default: runs/<variant>")
    a = p.parse_args(argv)
    out = a.out or Path("runs") / a.variant
    out.mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig(variant=VARIANTS[a.variant], iters=a.iters, num_envs=a.num_envs,
                    num_threads=a.threads, seed=a.seed)
    _, history, ckpts = train(cfg)
    (out / "history.json").write_text(json.dumps({"config": config_dict(cfg), "history": history}, indent=1))
    torch.save(ckpts, out / "checkpoints.pt")
    print(f"saved {out}/history.json and checkpoints.pt")


if __name__ == "__main__":
    main()
