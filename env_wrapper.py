import gymnasium as gym

class CustomRewardCartPole(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        x, x_dot, theta, theta_dot = obs

        reward, components = self.reward_fn(x, x_dot, theta, theta_dot, action)
        info["reward_components"] = components

        return obs, reward, terminated, truncated, info