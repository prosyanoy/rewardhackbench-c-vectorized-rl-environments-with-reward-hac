import numpy as np

from rewardhackbench import STAT_NAMES
from rewardhackbench.monitor import collect_episodes
from rewardhackbench.ppo import ActorCritic, PPOConfig, train

COL = {n: i for i, n in enumerate(STAT_NAMES)}


def test_ppo_learns_aligned_task():
    cfg = PPOConfig(variant=0, num_envs=64, num_threads=1, rollout_len=32, iters=80,
                    checkpoint_every=40, seed=0)
    model, hist, ckpts = train(cfg, log=lambda *_: None)
    assert len(hist) == 80 and set(ckpts) == {0, 40, 80}
    col = COL["true_return"]
    rand = collect_episodes(None, 0, 512, seed=9)[:, col].mean()
    trained = collect_episodes(model, 0, 512, seed=9)[:, col].mean()
    assert trained > rand + 0.2               # aligned reward: agent clearly improves


def test_policy_act_shapes():
    m = ActorCritic(147)
    a = m.act(np.zeros((5, 147), dtype=np.float32))
    assert a.shape == (5,) and a.dtype == np.int32 and ((0 <= a) & (a < 5)).all()
