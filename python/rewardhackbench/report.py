"""Run the reward-hacking monitor over every checkpoint of a training run.

    python -m rewardhackbench.report --run runs/sparkle [--plot]

The monitor is calibrated on noisy-expert episodes, then scored (precision /
recall against the engine's ground-truth "hacked" label) on fresh episodes from
each saved policy and on the pooled set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import STAT_NAMES
from .monitor import HackMonitor, NoisyExpert, collect_episodes
from .ppo import ActorCritic

COL = {n: i for i, n in enumerate(STAT_NAMES)}


def build_report(run: Path, episodes: int = 512, seed: int = 4242) -> dict:
    meta = json.loads((run / "history.json").read_text())
    cfg = meta["config"]
    ckpts = torch.load(run / "checkpoints.pt")
    variant, size, max_steps = cfg["variant"], cfg["size"], cfg["max_steps"]
    kw = dict(variant=variant, size=size, max_steps=max_steps)
    # Calibrate on a trusted benign policy (noisy shortest-path expert), different seed than evaluation.
    monitor = HackMonitor.calibrate(collect_episodes(NoisyExpert(size, seed=seed), n_envs=2048, seed=seed + 1, **kw))
    obs_dim = 3 * size * size
    per_ckpt, pooled = {}, []
    for it, sd in sorted(ckpts.items()):
        model = ActorCritic(obs_dim)
        model.load_state_dict(sd)
        stats = collect_episodes(model, n_envs=episodes, seed=seed, **kw)
        pooled.append(stats)
        per_ckpt[str(it)] = {
            "proxy_return": float(stats[:, COL["proxy_return"]].mean()),
            "true_return": float(stats[:, COL["true_return"]].mean()),
            "hack_rate": float(stats[:, COL["hacked"]].mean()),
            "monitor": monitor.evaluate(stats),
        }
    pooled = np.concatenate(pooled)
    return {"variant": variant, "thresholds": {"gap": monitor.gap_threshold, "oscillation": monitor.osc_threshold},
            "pooled": monitor.evaluate(pooled), "checkpoints": per_ckpt}


def print_report(rep: dict) -> None:
    t = rep["thresholds"]
    print(f"variant={rep['variant']}  thresholds: gap>{t['gap']:.3f}  bounce_run>{t['oscillation']:.1f}")
    print(f"{'ckpt':>6} {'proxy':>7} {'true':>7} {'hack%':>6} | "
          + " | ".join(f"{d:^21}" for d in ("divergence P/R", "anomaly P/R", "combined P/R")))
    fmt = lambda m: f"{m['precision']:.2f}/{m['recall']:.2f} (pos {m['positives']:3d})"
    for it, c in rep["checkpoints"].items():
        mm = c["monitor"]
        print(f"{it:>6} {c['proxy_return']:+7.2f} {c['true_return']:+7.2f} {100 * c['hack_rate']:6.1f} | "
              + " | ".join(f"{fmt(mm[k]):^21}" for k in ("divergence", "anomaly", "combined")))
    for k, m in rep["pooled"].items():
        print(f"pooled {k:>10}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} (tp={m['tp']} fp={m['fp']} fn={m['fn']} n={m['n']})")


def plot(run: Path, rep: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = json.loads((run / "history.json").read_text())["history"]
    x = [r["env_steps"] for r in h]
    k = 5
    sm = lambda key: np.convolve([r[key] for r in h], np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x[k - 1:], sm("proxy_return"), color="#d95f02", label="proxy reward (optimised)")
    ax.plot(x[k - 1:], sm("true_return"), color="#1b9e77", label="true reward (intended)")
    ax.set_xlabel("env steps")
    ax.set_ylabel("mean episode return")
    ax.legend()
    ax.set_title(f"variant {rep['variant']}: proxy rises, true falls")
    fig.tight_layout()
    path = run / "curves.png"
    fig.savefig(path, dpi=130)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args(argv)
    rep = build_report(a.run, a.episodes)
    print_report(rep)
    (a.run / "monitor_report.json").write_text(json.dumps(rep, indent=1))
    if a.plot:
        print("plot:", plot(a.run, rep))


if __name__ == "__main__":
    main()
