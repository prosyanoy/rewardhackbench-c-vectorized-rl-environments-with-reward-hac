"""Generate the README figures from benchmark + training runs.

    python -m rewardhackbench.assets --runs runs --out docs

Writes docs/throughput.png, docs/training_curves.png and docs/exploit.gif.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from . import STAT_NAMES, VecEnv
from .benchmark import run as run_benchmark
from .ppo import ActorCritic

PROXY, TRUE, PY = "#d95f02", "#1b9e77", "#555555"
COL = {n: i for i, n in enumerate(STAT_NAMES)}


def throughput_chart(out: Path, seconds: float = 0.5) -> dict:
    r = run_benchmark(seconds=seconds, threads=[1, 2, 4, 8])
    fig, ax = plt.subplots(figsize=(7, 4))
    shades = ["#7570b3", "#1b6ca8"]
    for (n, per), c in zip(r["cpp"].items(), shades):
        ts = sorted(per)
        ax.plot(ts, [per[t] for t in ts], "o-", color=c, label=f"C++ engine, {n:,} envs")
    ax.axhline(r["python_baseline_sps"], color=PY, ls="--", label="pure-Python baseline (1 thread)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel(f"threads  ({r['cpu_count']} physical cores on this machine)")
    ax.set_ylabel("environment steps / second")
    ax.set_title("Throughput vs. threads")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=130)
    plt.close(fig)
    (out / "bench.json").write_text(json.dumps(r, indent=1))
    return r


def training_chart(runs: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, v in zip(axes, ("sparkle", "shaping")):
        h = json.loads((runs / v / "history.json").read_text())["history"]
        x = np.array([r["env_steps"] for r in h]) / 1e6
        k = 5
        for key, color, label in (("proxy_return", PROXY, "proxy reward (optimised)"),
                                  ("true_return", TRUE, "true reward (intended)")):
            y = np.convolve([r[key] for r in h], np.ones(k) / k, mode="valid")
            ax.plot(x[k - 1:], y, color=color, lw=2, label=label)
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(f"'{v}' loophole")
        ax.set_xlabel("env steps (millions)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean episode return")
    axes[0].legend(frameon=False)
    fig.suptitle("PPO sees only the proxy: proxy rises, true reward falls")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)


def _find_hacked_episode(model, size, max_steps, variant):
    for seed in range(200):
        env = VecEnv(1, 1, size, max_steps, variant, 10_000 + seed)
        frames = []
        for _ in range(max_steps):
            frames.append((env.agent_pos(0), env.goal_pos(0), env.sparkle_pos(0)))
            env.step(model.act(env.obs.copy()))
            frames[-1] += (float(env.proxy_reward[0]), float(env.true_reward[0]))
            if env.done[0]:
                st = env.ep_stats[0]
                if st[COL["hacked"]] > 0.5 and st[COL["length"]] >= 30:
                    return frames
                break
    raise RuntimeError("no hacked episode found; is the checkpoint trained?")


def exploit_gif(runs: Path, out: Path) -> None:
    meta = json.loads((runs / "sparkle" / "history.json").read_text())["config"]
    ck = torch.load(runs / "sparkle" / "checkpoints.pt")
    size, steps = meta["size"], meta["max_steps"]
    model = ActorCritic(3 * size * size)
    model.load_state_dict(ck[max(ck)])
    frames = _find_hacked_episode(model, size, steps, meta["variant"])

    fig, ax = plt.subplots(figsize=(4.4, 4.9))
    ax.set_xlim(0, size); ax.set_ylim(size, 0); ax.set_aspect("equal")
    ax.set_xticks(range(size + 1)); ax.set_yticks(range(size + 1))
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(color="#ccc")
    cell = lambda p: (p % size + 0.5, p // size + 0.5)
    _, g, s, _, _ = frames[0]
    ax.add_patch(Rectangle((g % size, g // size), 1, 1, color=TRUE, alpha=0.85))
    ax.text(*cell(g), "goal", ha="center", va="center", color="white", fontsize=8)
    ax.add_patch(Rectangle((s % size, s // size), 1, 1, color="#e6ab02", alpha=0.85))
    ax.text(*cell(s), "sparkle", ha="center", va="center", fontsize=7)
    dot = Circle(cell(frames[0][0]), 0.3, color=PROXY, zorder=5)
    ax.add_patch(dot)
    title = ax.set_title("", fontsize=10)
    acc = {"p": 0.0, "t": 0.0}

    def draw(i):
        a, _, _, pr, tr = frames[i]
        # frame i holds the state BEFORE step i; show the result of step i
        if i == 0:
            acc["p"] = acc["t"] = 0.0
        acc["p"] += pr
        acc["t"] += tr
        # agent position after step i is the position recorded at frame i+1
        nxt = frames[i + 1][0] if i + 1 < len(frames) else a
        dot.center = cell(nxt)
        title.set_text(f"step {i + 1:2d}   proxy return {acc['p']:+.2f}   true return {acc['t']:+.2f}")
        return dot, title

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=140)
    anim.save(out / "exploit.gif", writer=PillowWriter(fps=7))
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument("--bench-seconds", type=float, default=0.5)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    r = throughput_chart(a.out, a.bench_seconds)
    training_chart(a.runs, a.out)
    exploit_gif(a.runs, a.out)
    print("wrote", ", ".join(sorted(f.name for f in a.out.iterdir())))
    best = max(v for per in r["cpp"].values() for v in per.values())
    print(f"peak {best:,.0f} steps/s = {best / r['python_baseline_sps']:.0f}x python baseline")


if __name__ == "__main__":
    main()
