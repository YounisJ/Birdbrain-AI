from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from flappy_bird_env import FlappyBirdEnv

def make_env():
    return FlappyBirdEnv(render_mode=False)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional, but safe on Windows

    # Use 4 parallel environments
    env = SubprocVecEnv([make_env for _ in range(4)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=4096,
        batch_size=1024,
        n_epochs=4,
        learning_rate=5e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )

    model.learn(total_timesteps=1_000_000)
    model.save("flappy_bird_ppo")
    
