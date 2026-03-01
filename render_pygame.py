# grid_soccer/render_pygame.py

from __future__ import annotations
import pygame
from typing import Dict, Tuple

from config import GameConfig
from env_soccer import (
    POS_NONE, POS_PLAYER, POS_AGENT,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY
)

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_STAY: "STAY",
}

class PygameRenderer:
    def __init__(self, cfg: GameConfig, cell_size: int = 40, margin: int = 2):
        self.cfg = cfg
        self.cell_size = cell_size
        self.margin = margin

        # HUD sizes
        self.hud_right_w = 0
        self.hud_bottom_h = 200

        self.grid_w_px = cfg.grid_w * cell_size
        self.grid_h_px = cfg.grid_h * cell_size

        self.w = self.grid_w_px + self.hud_right_w
        self.h = self.grid_h_px + self.hud_bottom_h

        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Grid Soccer")

        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()

    def close(self):
        pygame.quit()


    def draw(self, snap: Dict, hud: Dict, fps: int = 30):
        self.clock.tick(fps)
        self._draw_scene(snap, hud, player_pos=snap["player_pos"], agent_pos=snap["agent_pos"], ball_pos=snap["ball_pos"])
        pygame.display.flip()

    def draw_transition(self, prev_snap: Dict, curr_snap: Dict, hud: Dict, steps: int = 8, fps: int = 60):

        p0x, p0y = prev_snap["player_pos"]
        p1x, p1y = curr_snap["player_pos"]
        a0x, a0y = prev_snap["agent_pos"]
        a1x, a1y = curr_snap["agent_pos"]
        b0x, b0y = prev_snap["ball_pos"]
        b1x, b1y = curr_snap["ball_pos"]

        for i in range(1, steps + 1):
            self.clock.tick(fps)
            t = i / steps

            def lerp(u, v):
                return u + (v - u) * t

            player_fx, player_fy = lerp(p0x, p1x), lerp(p0y, p1y)
            agent_fx, agent_fy   = lerp(a0x, a1x), lerp(a0y, a1y)
            ball_fx, ball_fy     = lerp(b0x, b1x), lerp(b0y, b1y)

            # Render using float positions for smooth movement
            self._draw_scene_float(
                curr_snap, hud,
                player_f=(player_fx, player_fy),
                agent_f=(agent_fx, agent_fy),
                ball_f=(ball_fx, ball_fy),
            )
            pygame.display.flip()


    def _draw_scene(self, snap: Dict, hud: Dict, player_pos: Tuple[int, int], agent_pos: Tuple[int, int], ball_pos: Tuple[int, int]):

        self.screen.fill((245, 245, 245))

        goal_cells = snap["goal_cells"]
        possession = snap["possession"]

        # Grid
        for y in range(self.cfg.grid_h):
            for x in range(self.cfg.grid_w):
                rect = pygame.Rect(
                    x * self.cell_size + self.margin,
                    y * self.cell_size + self.margin,
                    self.cell_size - 2 * self.margin,
                    self.cell_size - 2 * self.margin,
                )
                pygame.draw.rect(self.screen, (255, 245, 160) if (x, y) in goal_cells else (230, 230, 230), rect)

        # Entities
        self._draw_square(player_pos, (80, 140, 255))
        self._draw_square(agent_pos, (255, 90, 90))

        # Ball / possession indicator
        if possession == POS_PLAYER:
            self._draw_ball_above(player_pos)
        elif possession == POS_AGENT:
            self._draw_ball_above(agent_pos)
        else:
            self._draw_circle(ball_pos, (0, 0, 0))

        # HUD
        self._draw_hud(snap, hud)

    def _draw_scene_float(self, snap: Dict, hud: Dict, player_f: Tuple[float, float], agent_f: Tuple[float, float], ball_f: Tuple[float, float]):
        self.screen.fill((245, 245, 245))

        goal_cells = snap["goal_cells"]
        possession = snap["possession"]

        # Grid
        for y in range(self.cfg.grid_h):
            for x in range(self.cfg.grid_w):
                rect = pygame.Rect(
                    x * self.cell_size + self.margin,
                    y * self.cell_size + self.margin,
                    self.cell_size - 2 * self.margin,
                    self.cell_size - 2 * self.margin,
                )
                pygame.draw.rect(self.screen, (255, 245, 160) if (x, y) in goal_cells else (230, 230, 230), rect)

        # Entities (smooth)
        self._draw_square_float(player_f[0], player_f[1], (80, 140, 255))
        self._draw_square_float(agent_f[0], agent_f[1], (255, 90, 90))

        # Ball / possession indicator (smooth)
        if possession == POS_PLAYER:
            self._draw_ball_above_float(player_f[0], player_f[1])
        elif possession == POS_AGENT:
            self._draw_ball_above_float(agent_f[0], agent_f[1])
        else:
            self._draw_circle_float(ball_f[0], ball_f[1], (0, 0, 0))

        # HUD (not animated; show current values)
        self._draw_hud(snap, hud)

    def _draw_hud(self, snap: Dict, hud: Dict):
        possession = snap["possession"]
        pos_text = "NONE" if possession == POS_NONE else ("PLAYER" if possession == POS_PLAYER else "AGENT")

        base_y = self.grid_h_px + 18
        line = 28
        cx = self.w // 2

        state = hud.get("state")
        agent_action = hud.get("agent_action")
        reward = hud.get("reward")
        episode = hud.get("episode")
        score = hud.get("score")

        self._draw_text_center(f"Possession: {pos_text}", cx, base_y)
        if agent_action is not None:
            name = ACTION_NAMES.get(agent_action, str(agent_action))
            self._draw_text_center(f"Agent Action: {name}", cx, base_y + 1 * line)
        if episode is not None:
            self._draw_text_center(f"Episode: {episode}", cx, base_y + 2 * line)
        if score is not None:
            self._draw_text_center(f"Score (Agent-Player): {score[0]} - {score[1]}", cx, base_y + 3 * line)


    def _draw_square(self, pos: Tuple[int, int], color):
        x, y = pos
        rect = pygame.Rect(
            x * self.cell_size + 12,
            y * self.cell_size + 12,
            self.cell_size - 24,
            self.cell_size - 24,
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=10)

    def _draw_circle(self, pos: Tuple[int, int], color):
        x, y = pos
        cx = x * self.cell_size + self.cell_size // 2
        cy = y * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, color, (cx, cy), self.cell_size // 6)

    def _draw_ball_above(self, pos: Tuple[int, int]):
        x, y = pos
        cx = x * self.cell_size + self.cell_size // 2
        cy = y * self.cell_size + 10
        pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), self.cell_size // 7)

    # Float versions for smooth animation
    def _draw_square_float(self, gx: float, gy: float, color):
        x_px = gx * self.cell_size + 12
        y_px = gy * self.cell_size + 12
        rect = pygame.Rect(
            int(x_px),
            int(y_px),
            self.cell_size - 24,
            self.cell_size - 24,
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=10)

    def _draw_circle_float(self, gx: float, gy: float, color):
        cx = int(gx * self.cell_size + self.cell_size // 2)
        cy = int(gy * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, color, (cx, cy), self.cell_size // 6)

    def _draw_ball_above_float(self, gx: float, gy: float):
        cx = int(gx * self.cell_size + self.cell_size // 2)
        cy = int(gy * self.cell_size + 10)
        pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), self.cell_size // 7)

    def _draw_text(self, text: str, x: int, y: int):
        surf = self.font.render(text, True, (30, 30, 30))
        self.screen.blit(surf, (x, y))
    def _draw_text_center(self, text: str, center_x: int, y: int):
        surf = self.font.render(text, True, (30, 30, 30))
        rect = surf.get_rect(center=(center_x, y))
        self.screen.blit(surf, rect)