"""Compact CPU PPO (clip objective, GAE) trained on the C++ vectorized env."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from . import NUM_ACTIONS, STAT_NAMES, VecEnv

COL = {n: i for i, n in enumerate(STAT_NAMES)}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, NUM_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        logits, _ = self(torch.from_numpy(obs))
        if greedy:
            return logits.argmax(-1).numpy().astype(np.int32)
        return torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int32)


@dataclass
class PPOConfig:
    variant: int = 1
    num_envs: int = 128
    num_threads: int = 2
    rollout_len: int = 64
    iters: int = 200
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 0.5
    size: int = 7
    max_steps: int = 50
    seed: int = 0
    checkpoint_every: int = 20


def train(cfg: PPOConfig, log=print):
    """Returns (model, history list, {iteration: state_dict} checkpoints)."""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, cfg.num_threads))
    env = VecEnv(cfg.num_envs, cfg.num_threads, cfg.size, cfg.max_steps, cfg.variant, cfg.seed)
    model = ActorCritic(env.obs_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    T, N = cfg.rollout_len, cfg.num_envs
    obs_buf = torch.zeros(T, N, env.obs_dim)
    act_buf = torch.zeros(T, N, dtype=torch.long)
    logp_buf, rew_buf, done_buf, val_buf = (torch.zeros(T, N) for _ in range(4))
    history, checkpoints = [], {0: {k: v.clone() for k, v in model.state_dict().items()}}
    obs = torch.from_numpy(env.obs.copy())

    for it in range(1, cfg.iters + 1):
        t0 = time.perf_counter()
        fin = []
        for t in range(T):
            with torch.no_grad():
                logits, val = model(obs)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, a, dist.log_prob(a), val
            env.step(a.numpy().astype(np.int32))
            rew_buf[t] = torch.from_numpy(env.proxy_reward.copy())   # learner sees PROXY only
            d = env.done.astype(bool)
            done_buf[t] = torch.from_numpy(d.astype(np.float32))
            if d.any():
                fin.append(env.ep_stats[d].copy())
            obs = torch.from_numpy(env.obs.copy())
        with torch.no_grad():
            _, last_v = model(obs)
        adv = torch.zeros(T, N)
        g = torch.zeros(N)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_buf[t + 1]
            nd = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * nv * nd - val_buf[t]
            g = delta + cfg.gamma * cfg.lam * nd * g
            adv[t] = g
        ret = adv + val_buf
        b_obs, b_act = obs_buf.reshape(-1, env.obs_dim), act_buf.reshape(-1)
        b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        n = b_obs.shape[0]
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                logits, v = model(b_obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a_n = b_adv[idx]
                a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8)
                pg = -torch.min(ratio * a_n, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_n).mean()
                loss = pg + cfg.vf_coef * 0.5 * (v - b_ret[idx]).pow(2).mean() \
                    - cfg.ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
        eps = np.concatenate(fin) if fin else np.zeros((0, len(STAT_NAMES)), np.float32)
        m = (lambda c: float(eps[:, COL[c]].mean()) if len(eps) else float("nan"))
        rec = {"iter": it, "env_steps": it * T * N, "episodes": int(len(eps)),
               "proxy_return": m("proxy_return"), "true_return": m("true_return"),
               "goal_rate": m("goal_reached"), "hack_rate": m("hacked"),
               "sps": T * N / (time.perf_counter() - t0)}
        history.append(rec)
        if it % cfg.checkpoint_every == 0 or it == cfg.iters:
            checkpoints[it] = {k: v.clone() for k, v in model.state_dict().items()}
        if it % 10 == 0 or it == 1:
            log(f"iter {it:4d} steps {rec['env_steps']:8d} proxy {rec['proxy_return']:+.3f} "
                f"true {rec['true_return']:+.3f} goal {rec['goal_rate']:.2f} "
                f"hack {rec['hack_rate']:.2f} ({rec['sps']:.0f} sps)")
    return model, history, checkpoints


def config_dict(cfg: PPOConfig) -> dict:
    return asdict(cfg)
