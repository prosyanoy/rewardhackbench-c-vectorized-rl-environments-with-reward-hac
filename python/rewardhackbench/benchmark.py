"""Steps/second: C++ engine at several thread counts vs the pure-Python baseline.

    python -m rewardhackbench.benchmark [--num-envs 1024 32768] [--seconds 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import VecEnv
from .pure_python import PyVecEnv


def _measure(step, n_envs: int, actions: np.ndarray, seconds: float, warmup: int = 5) -> float:
    for _ in range(warmup):
        step(actions[0])
    t0, steps, i = time.perf_counter(), 0, 0
    while time.perf_counter() - t0 < seconds:
        step(actions[i % len(actions)])
        steps += n_envs
        i += 1
    return steps / (time.perf_counter() - t0)


def run(num_envs=(1024, 32768), seconds=1.0, threads=None, variant=1, size=7) -> dict:
    threads = threads or sorted({1, 2, 4, 8, os.cpu_count() or 1})
    rng = np.random.default_rng(0)
    py_n = 256
    py = PyVecEnv(py_n, variant=variant, size=size)
    py_actions = rng.integers(0, 5, (64, py_n)).astype(np.int32)
    res = {"cpu_count": os.cpu_count(), "python_baseline_sps": _measure(py.step, py_n, py_actions, seconds),
           "cpp": {}}
    for n in num_envs:
        actions = rng.integers(0, 5, (64, n)).astype(np.int32)
        res["cpp"][n] = {}
        for t in threads:
            env = VecEnv(n, t, size, 50, variant, 0)
            res["cpp"][n][t] = _measure(env.step, n, actions, seconds)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, nargs="*", default=[1024, 32768])
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--threads", type=int, nargs="*")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    r = run(a.num_envs, a.seconds, a.threads)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    base = r["python_baseline_sps"]
    print(f"cpu cores: {r['cpu_count']}  (pure-Python baseline: 256 envs, 1 thread)")
    print(f"{'engine':<14}{'envs':>8}{'threads':>8}{'steps/s':>14}{'speedup':>10}")
    print(f"{'pure-python':<14}{256:>8}{1:>8}{base:>14,.0f}{1.0:>9.1f}x")
    for n, per in r["cpp"].items():
        for t, sps in per.items():
            print(f"{'cpp-vecenv':<14}{n:>8}{t:>8}{sps:>14,.0f}{sps / base:>9.1f}x")


if __name__ == "__main__":
    main()
