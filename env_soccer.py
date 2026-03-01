# grid_soccer/env_soccer.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import random

from config import GameConfig
from utils import clamp

# possession: 0 none, 1 player, 2 agent
POS_NONE = 0
POS_PLAYER = 1
POS_AGENT = 2

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

@dataclass
class StepInfo:
    agent_action: int
    player_action: int
    possession: int
    scored_by: int  # 0 none, 1 player, 2 agent
    step: int

class GridSoccerEnv:


    def __init__(self, cfg: GameConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.player_pos = (0, 0)
        self.agent_pos = (0, 0)
        self.ball_pos = (0, 0)
        self.possession = POS_NONE
        self.step_count = 0

        self.goal_cells = {(cfg.goal_x, cfg.goal_y1), (cfg.goal_x, cfg.goal_y2)}

    def reset(self) -> Tuple[Tuple[int, int, int, int, int], Dict]:
        self.step_count = 0
        w, h = self.cfg.grid_w, self.cfg.grid_h

        self.agent_pos = (1, h // 2)
        self.player_pos = (1, clamp(h // 2 + 1, 0, h - 1))
        self.possession = POS_NONE
        for _ in range(200):
            bx = self.rng.randrange(0, w)
            by = self.rng.randrange(0, h)
            if (bx, by) in self.goal_cells:
                continue
            if (bx, by) == self.agent_pos or (bx, by) == self.player_pos:
                continue
            self.ball_pos = (bx, by)
            break
        else:
            # fallback
            self.ball_pos = (w // 2, h // 2)

        obs = self._get_state()
        return obs, {}

    def step(self, agent_action: int, player_action: int) -> Tuple[Tuple[int, int, int, int, int], float, bool, StepInfo]:
        self.step_count += 1
        reward = self.cfg.step_penalty
        done = False
        scored_by = 0

        prev_possession = self.possession

        prev_agent_pos = self.agent_pos
        prev_ball_pos = self.ball_pos


        self.player_pos = self._move(self.player_pos, player_action)
        self.agent_pos = self._move(self.agent_pos, agent_action)

        if self.player_pos == self.ball_pos and self.agent_pos == self.ball_pos:
            self.possession = POS_AGENT
        elif self.player_pos == self.ball_pos:
            self.possession = POS_PLAYER
        elif self.agent_pos == self.ball_pos:
            self.possession = POS_AGENT


        if self.possession == POS_PLAYER:
            self.ball_pos = self.player_pos
        elif self.possession == POS_AGENT:
            self.ball_pos = self.agent_pos

        if prev_possession != self.possession:
            if self.possession == POS_AGENT:
                reward += self.cfg.gain_possession_reward
            elif prev_possession == POS_AGENT and self.possession != POS_AGENT:
                reward += self.cfg.lose_possession_penalty

        if self.possession == POS_AGENT:
            goal_x = self.cfg.goal_x
            prev_dist = abs(goal_x - prev_agent_pos[0])
            new_dist = abs(goal_x - self.agent_pos[0])

            if new_dist < prev_dist:
                reward += 0.20
            elif new_dist > prev_dist:
                reward -= 0.20
            else:
                reward -= 0.05


        if self.ball_pos in self.goal_cells:
            done = True
            if self.possession == POS_AGENT:
                reward = self.cfg.agent_goal_reward
                scored_by = POS_AGENT
            elif self.possession == POS_PLAYER:
                reward = self.cfg.player_goal_penalty
                scored_by = POS_PLAYER
            else:
                reward = 0.0
                scored_by = 0


        if not done and self.step_count >= self.cfg.max_steps_per_episode:
            done = True
            scored_by = 0

        obs = self._get_state()
        info = StepInfo(
            agent_action=agent_action,
            player_action=player_action,
            possession=self.possession,
            scored_by=scored_by,
            step=self.step_count,
        )
        return obs, float(reward), bool(done), info

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        x, y = pos
        if action == ACTION_UP:
            y -= 1
        elif action == ACTION_DOWN:
            y += 1
        elif action == ACTION_LEFT:
            x -= 1
        elif action == ACTION_RIGHT:
            x += 1
        elif action == ACTION_STAY:
            pass

        x = clamp(x, 0, self.cfg.grid_w - 1)
        y = clamp(y, 0, self.cfg.grid_h - 1)
        return (x, y)

    def _get_state(self) -> Tuple[int, int, int, int, int, int, int]:
        """
        Add goal-relative info to avoid dithering near the goal.
        state = (dx_ball, dy_ball, dx_player, dy_player, dx_goal, dy_goal, possession)
        """
        ax, ay = self.agent_pos
        bx, by = self.ball_pos
        px, py = self.player_pos

        dx_ball = bx - ax
        dy_ball = by - ay
        dx_player = px - ax
        dy_player = py - ay

        goal_x = self.cfg.goal_x
        goal_y_center = (self.cfg.goal_y1 + self.cfg.goal_y2) // 2
        dx_goal = goal_x - ax
        dy_goal = goal_y_center - ay

        return (dx_ball, dy_ball, dx_player, dy_player, dx_goal, dy_goal, self.possession)

    def snapshot(self) -> Dict:
        return {
            "player_pos": self.player_pos,
            "agent_pos": self.agent_pos,
            "ball_pos": self.ball_pos,
            "possession": self.possession,
            "goal_cells": self.goal_cells,
            "step": self.step_count,
        }

def scripted_player_policy(env: GridSoccerEnv) -> int:

    (px, py) = env.player_pos
    (bx, by) = env.ball_pos
    cfg = env.cfg

    r = env.rng.random()
    if r < 0.12:
        return ACTION_STAY
    if r < 0.20:
        return env.rng.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])

    if env.possession == POS_PLAYER:
        if px < cfg.goal_x:
            return ACTION_RIGHT
        return ACTION_STAY

    if bx > px:
        return ACTION_RIGHT
    if bx < px:
        return ACTION_LEFT
    if by > py:
        return ACTION_DOWN
    if by < py:
        return ACTION_UP
    return ACTION_STAY
