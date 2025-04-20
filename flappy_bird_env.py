import gym
from gym import spaces
import numpy as np
import random
import pygame

class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(FlappyBirdEnv, self).__init__()
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((350, 622))
            pygame.display.set_caption("Flappy Bird AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.gravity = 0.17
        self.flap_strength = -7
        self.pipe_gap = 120
        self.pipe_speed = 3

        # Observation: bird_y, bird_velocity, pipe_x, pipe_y
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0], dtype=np.float32),
            high=np.array([622, 10, 350, 622], dtype=np.float32)
        )

        # Action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        self.reset()

    def reset(self):
        self.bird_y = 311
        self.bird_velocity = 0
        self.pipe_x = 350
        self.pipe_y = random.randint(200, 450)
        self.score = 0
        self.done = False

        return self._get_obs()

    def step(self, action):
        reward = 1  # reward for surviving
        if action == 1:
            self.bird_velocity = self.flap_strength

        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        self.pipe_x -= self.pipe_speed

        if self.pipe_x < -50:
            self.pipe_x = 350
            self.pipe_y = random.randint(200, 450)
            self.score += 1
            reward += 10  # reward for passing pipe

        # Collision detection
        if (
            self.bird_y <= 0
            or self.bird_y >= 622
            or (
                67 < self.pipe_x < 100 and
                (self.bird_y < self.pipe_y - self.pipe_gap // 2 or self.bird_y > self.pipe_y + self.pipe_gap // 2)
            )
        ):
            self.done = True
            reward = -100

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return np.array([
            self.bird_y,
            self.bird_velocity,
            self.pipe_x,
            self.pipe_y
        ], dtype=np.float32)

    def render(self, mode="human"):
        if not self.render_mode:
            return

        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 0), (67, self.bird_y, 30, 30))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, 0, 50, self.pipe_y - self.pipe_gap // 2))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, self.pipe_y + self.pipe_gap // 2, 50, 622))

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode:
            pygame.quit()
