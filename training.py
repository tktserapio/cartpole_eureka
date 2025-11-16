import gymnasium as gym
from stable_baselines3 import PPO
from env_wrapper import CustomRewardCartPole
from reward_code import build_reward_fn_from_code

def train_model(reward_code, total_timesteps=30_000, seed=0):
    reward_fn = build_reward_fn_from_code(reward_code)
    env = CustomRewardCartPole(gym.make("CartPole-v1"), reward_fn)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu")
    model.learn(total_timesteps=total_timesteps)
    return model, env

def evaluate(model, env, episodes=10):
    total = 0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_len = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_len += 1
        total += ep_len
    return total / episodes
