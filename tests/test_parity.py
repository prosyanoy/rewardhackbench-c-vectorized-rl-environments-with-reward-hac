"""C++ engine vs independent pure-Python implementation, same layouts + actions."""
import numpy as np
import pytest

from rewardhackbench import STAT_NAMES, VecEnv
from rewardhackbench.pure_python import PyGridEnv, PyVecEnv

COL = {n: i for i, n in enumerate(STAT_NAMES)}


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_dynamics_match(variant):
    n, rng = 16, np.random.default_rng(variant)
    env = VecEnv(n, 2, variant=variant, max_steps=30)
    refs = [PyGridEnv(variant=variant, max_steps=30) for _ in range(n)]
    layouts = []
    for i in range(n):
        a, g, s = rng.choice(49, 3, replace=False)
        layouts.append((int(a), int(g), int(s) if variant == 1 else -1))
        env.set_layout(i, *layouts[-1])
        refs[i].reset(layouts[-1])
    for step in range(200):
        acts = rng.integers(0, 5, n).astype(np.int32)
        env.step(acts)
        for i, r in enumerate(refs):
            obs, rew, term, trunc, info = r.step(int(acts[i]))
            assert env.proxy_reward[i] == pytest.approx(rew, abs=1e-6)
            assert env.true_reward[i] == pytest.approx(info["true_reward"], abs=1e-6)
            assert bool(env.done[i]) == (term or trunc)
            if term or trunc:
                st, ep = env.ep_stats[i], info["episode"]
                assert st[COL["proxy_return"]] == pytest.approx(ep["proxy_return"], abs=1e-4)
                assert st[COL["true_return"]] == pytest.approx(ep["true_return"], abs=1e-4)
                assert st[COL["length"]] == ep["length"]
                assert st[COL["oscillations"]] == ep["oscillations"]
                assert st[COL["osc_run"]] == ep["osc_run"]
                assert st[COL["unique_cells"]] == ep["unique_cells"]
                # both auto-reset into a fresh random layout; re-sync them
                layouts[i] = (env.agent_pos(i), env.goal_pos(i), env.sparkle_pos(i))
                refs[i].reset(layouts[i])
            else:
                assert np.array_equal(env.obs[i], obs)


def test_py_vec_env_runs():
    v = PyVecEnv(4, seed=1)
    for _ in range(120):
        obs, r, d = v.step(np.zeros(4, dtype=int))
    assert obs.shape == (4, 147)
