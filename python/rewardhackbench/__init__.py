"""RewardHackBench: vectorized RL environments with reward-hacking monitors."""
from ._core import CHANNELS, NUM_ACTIONS, STAT_NAMES, VecEnv

VARIANTS = {"aligned": 0, "sparkle": 1, "shaping": 2}

__all__ = ["VecEnv", "VARIANTS", "NUM_ACTIONS", "CHANNELS", "STAT_NAMES"]
