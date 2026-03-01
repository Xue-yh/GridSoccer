# grid_soccer/train.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from config import GameConfig
from env_soccer import GridSoccerEnv, scripted_player_policy, POS_AGENT, POS_PLAYER
from q_agent import QLearningAgent

@dataclass
class TrainStats:
    episode_rewards: List[float]
    success_rate: List[float]  # agent goal rate over windows

def train(cfg: GameConfig, episodes: int = 30000, seed: int = 0, log_every: int = 1000) -> Tuple[QLearningAgent, TrainStats]:
    env = GridSoccerEnv(cfg, seed=seed)
    agent = QLearningAgent(cfg, seed=seed)

    episode_rewards: List[float] = []
    success_rate: List[float] = []

    win_count = 0
    window = 1000

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_r = 0.0
        scored_by = 0

        while not done:
            a_action = agent.choose_action(state, train=True)
            p_action = scripted_player_policy(env)
            next_state, reward, done, info = env.step(a_action, p_action)
            agent.update(state, a_action, reward, next_state, done)
            state = next_state
            total_r += reward
            scored_by = info.scored_by

        # Count agent goals as "wins"
        if scored_by == POS_AGENT:
            win_count += 1

        episode_rewards.append(total_r)
        agent.decay_epsilon()

        if ep % log_every == 0:
            start = max(0, ep - window)
            rate = evaluate_success_rate(cfg, agent, eval_episodes=300, seed=seed + ep)
            success_rate.append(rate)
            avg_r = float(np.mean(episode_rewards[-log_every:]))
            print(f"[train] ep={ep:6d} eps={agent.epsilon:.3f} avg_reward={avg_r:.3f} eval_success={rate:.3f}")

    return agent, TrainStats(episode_rewards=episode_rewards, success_rate=success_rate)

def evaluate_success_rate(cfg: GameConfig, agent: QLearningAgent, eval_episodes: int = 200, seed: int = 123) -> float:
    env = GridSoccerEnv(cfg, seed=seed)
    wins = 0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        scored_by = 0
        while not done:
            a_action = agent.choose_action(state, train=False)
            p_action = scripted_player_policy(env)
            state, reward, done, info = env.step(a_action, p_action)
            scored_by = info.scored_by
        if scored_by == POS_AGENT:
            wins += 1
    return wins / eval_episodes
