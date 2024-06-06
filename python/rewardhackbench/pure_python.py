"""Pure-Python Gym-style reference implementation of the same gridworld.

Serves two purposes: the throughput baseline for the benchmark, and an
independent oracle for the C++ dynamics (see tests/test_parity.py).
"""
from __future__ import annotations

import random

import numpy as np

_DX = (0, 0, -1, 1, 0)
_DY = (-1, 1, 0, 0, 0)


class PyGridEnv:
    """Gym-style API: reset() -> obs, step(a) -> obs, reward, terminated, truncated, info."""

    def __init__(self, size=7, max_steps=50, variant=1, seed=0, step_cost=0.01,
                 goal_reward=1.0, sparkle_bonus=0.5, shaping_bonus=0.2):
        self.size, self.max_steps, self.variant = size, max_steps, variant
        self.step_cost, self.goal_reward = step_cost, goal_reward
        self.sparkle_bonus, self.shaping_bonus = sparkle_bonus, shaping_bonus
        self.rng = random.Random(seed)
        self.n = size * size
        self.obs = np.zeros(3 * self.n, dtype=np.float32)

    def _dist(self, a, b):
        s = self.size
        return abs(a // s - b // s) + abs(a % s - b % s)

    def reset(self, layout=None):
        if layout is None:
            a = self.rng.randrange(self.n)
            g = self.rng.randrange(self.n)
            while g == a:
                g = self.rng.randrange(self.n)
            s = -1
            if self.variant == 1:
                s = self.rng.randrange(self.n)
                while s in (a, g):
                    s = self.rng.randrange(self.n)
            layout = (a, g, s)
        self.agent, self.goal, sp = layout
        self.sparkle = sp if self.variant == 1 else -1
        self.t, self.prev = 0, -1
        self.bonus_events = self.osc = self.osc_cur = self.osc_run = 0
        self.d0 = self._dist(self.agent, self.goal)
        self.visited = {self.agent}
        self.proxy_ret = self.true_ret = 0.0
        self.obs[:] = 0
        self.obs[self.agent] = 1
        self.obs[self.n + self.goal] = 1
        if self.sparkle >= 0:
            self.obs[2 * self.n + self.sparkle] = 1
        return self.obs.copy()

    def step(self, action):
        s = self.size
        old = self.agent
        x = min(max(old % s + _DX[action], 0), s - 1)
        y = min(max(old // s + _DY[action], 0), s - 1)
        new = y * s + x
        self.t += 1
        true_r = -self.step_cost
        proxy_r = -self.step_cost
        if self.variant == 1 and new == self.sparkle and new != old:
            proxy_r += self.sparkle_bonus
            self.bonus_events += 1
        elif self.variant == 2 and self._dist(new, self.goal) < self._dist(old, self.goal):
            proxy_r += self.shaping_bonus
            self.bonus_events += 1
        if new == self.prev and new != old:
            self.osc += 1
            self.osc_cur += 1
            self.osc_run = max(self.osc_run, self.osc_cur)
        else:
            self.osc_cur = 0
        if new != old:
            self.obs[old] = 0
            self.obs[new] = 1
            self.visited.add(new)
        self.prev, self.agent = old, new
        terminated = new == self.goal
        if terminated:
            true_r += self.goal_reward
            proxy_r += self.goal_reward
        self.proxy_ret += proxy_r
        self.true_ret += true_r
        truncated = (not terminated) and self.t >= self.max_steps
        info = {"true_reward": true_r}
        if terminated or truncated:
            info["episode"] = {"proxy_return": self.proxy_ret, "true_return": self.true_ret,
                               "length": self.t, "oscillations": self.osc, "osc_run": self.osc_run,
                               "unique_cells": len(self.visited)}
        return self.obs.copy(), proxy_r, terminated, truncated, info


class PyVecEnv:
    """Naive sequential vectorization with auto-reset (the usual Gym-style loop)."""

    def __init__(self, num_envs, seed=0, **kw):
        self.envs = [PyGridEnv(seed=seed * 1000003 + i, **kw) for i in range(num_envs)]
        self.num_envs = num_envs
        self.obs = np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, _ = e.step(int(a))
            if term or trunc:
                o = e.reset()
                dones[i] = True
            self.obs[i] = o
            rewards[i] = r
        return self.obs, rewards, dones
