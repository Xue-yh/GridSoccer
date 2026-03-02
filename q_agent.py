# grid_soccer/q_agent.py

from __future__ import annotations
import random
from typing import Dict, Tuple

from config import GameConfig

class QLearningAgent:
    def __init__(self, cfg: GameConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        # Q-table as dict: key=(state, action) -> value
        self.Q: Dict[Tuple[Tuple[int, ...], int], float] = {}
        self.epsilon = cfg.epsilon_start

    def choose_action(self, state: Tuple[int, ...], train: bool = True) -> int:
        # ε-greedy exploration (only in training)
        if train and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.cfg.n_actions)

        # greedy action selection
        qs = [self.Q.get((state, a), 0.0) for a in range(self.cfg.n_actions)]
        max_q = max(qs)
        best = [a for a, q in enumerate(qs) if q == max_q]

        # Prefer RIGHT > LEFT > UP > DOWN > STAY
        pref_order = [3, 2, 0, 1, 4]
        for a in pref_order:
            if a in best:
                return a

        # fallback (shouldn't happen)
        return best[0]

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool
    ) -> None:
        alpha = self.cfg.alpha
        gamma = self.cfg.gamma

        old = self.Q.get((state, action), 0.0)
        if done:
            target = reward
        else:
            next_max = max(self.Q.get((next_state, a), 0.0) for a in range(self.cfg.n_actions))
            target = reward + gamma * next_max

        self.Q[(state, action)] = old + alpha * (target - old)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
