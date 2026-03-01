# grid_soccer/main.py

from __future__ import annotations
import argparse
import pygame

from config import GameConfig
from train import train
from env_soccer import GridSoccerEnv, POS_AGENT, POS_PLAYER
from q_agent import QLearningAgent
from render_pygame import PygameRenderer
from utils import save_pickle, load_pickle

Q_PATH = "q_table.pkl"

# Play-mode animation controls
ANIM_STEPS = 8   # 每一步插值帧数：越大越丝滑越慢（建议 6~10）
ANIM_FPS   = 60  # 动画帧率：建议 40~60

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    parser.add_argument("--episodes", type=int, default=30000)
    args = parser.parse_args()

    cfg = GameConfig()

    if args.mode == "train":
        agent, stats = train(cfg, episodes=args.episodes, seed=0, log_every=1000)
        save_pickle({"Q": agent.Q, "cfg": cfg}, Q_PATH)
        print(f"✅ saved {Q_PATH} (Q-table size: {len(agent.Q)})")
        return

    if args.mode == "play":
        data = load_pickle(Q_PATH)
        agent = QLearningAgent(cfg)
        agent.Q = data["Q"]
        agent.epsilon = 0.0  # ✅ play 模式必须 greedy（不探索）

        env = GridSoccerEnv(cfg, seed=42)
        renderer = PygameRenderer(cfg, cell_size=64)

        score_agent, score_player = 0, 0
        episode = 1

        state, _ = env.reset()
        last_reward = 0.0
        last_agent_action = ACTION_STAY = 4  # local alias

        running = True
        while running:
            # -------- handle events --------
            player_action = 4  # STAY default
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        state, _ = env.reset()
                        episode += 1

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player_action = 2
            elif keys[pygame.K_RIGHT]:
                player_action = 3
            elif keys[pygame.K_UP]:
                player_action = 0
            elif keys[pygame.K_DOWN]:
                player_action = 1
            else:
                player_action = 4

            # -------- one environment step (capture prev/curr for animation) --------
            prev_snap = env.snapshot()

            agent_action = agent.choose_action(state, train=False)
            next_state, reward, done, info = env.step(agent_action, player_action)

            curr_snap = env.snapshot()

            last_reward = reward
            last_agent_action = agent_action
            state = next_state

            if done:
                if info.scored_by == POS_AGENT:
                    score_agent += 1
                elif info.scored_by == POS_PLAYER:
                    score_player += 1
                state, _ = env.reset()
                episode += 1

            # -------- smooth render --------
            renderer.draw_transition(
                prev_snap,
                curr_snap,
                hud={
                    "state": state,
                    "agent_action": last_agent_action,
                    "reward": last_reward,
                    "episode": episode,
                    "score": (score_agent, score_player),
                },
                steps=ANIM_STEPS,
                fps=ANIM_FPS,
            )

        renderer.close()

if __name__ == "__main__":
    main()
