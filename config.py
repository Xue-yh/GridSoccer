# grid_soccer/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class GameConfig:
    grid_w: int = 14
    grid_h: int = 10

    goal_x: int = 13
    goal_y1: int = 4
    goal_y2: int = 5

    max_steps_per_episode: int = 120

    # Rewards
    step_penalty: float = -0.01

    agent_goal_reward: float = 20.0
    player_goal_penalty: float = -20.0

    gain_possession_reward: float = 2.0
    lose_possession_penalty: float = -2.0

    # Q-learning hyperparams
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0

    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995  # per episode

    # Actions
    n_actions: int = 5
