from stable_baselines3 import PPO
from flappy_bird_env import FlappyBirdEnv

env = FlappyBirdEnv(render_mode=True)
model = PPO.load("flappy_bird_ppo")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()

