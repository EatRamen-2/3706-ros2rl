# 3,5,7
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import random


class MovingTargetEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, grid_size=3, max_steps=50):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Tuple((gym.spaces.Discrete(grid_size), gym.spaces.Discrete(grid_size))),
            "target": gym.spaces.Tuple((gym.spaces.Discrete(grid_size), gym.spaces.Discrete(grid_size)))
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        self.target_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        while self.target_pos == self.agent_pos:
            self.target_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        self.step_count = 0
        return {"agent": self.agent_pos, "target": self.target_pos}, {}

    def step(self, action):
        self.step_count += 1
        reward = -0.1

        # Move agent
        if action == 0:  # up
            new_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # down
            new_pos = (min(self.grid_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 2:  # left
            new_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 3:  # right
            new_pos = (self.agent_pos[0], min(self.grid_size - 1, self.agent_pos[1] + 1))

        if new_pos != self.agent_pos:
            self.agent_pos = new_pos
        else:
            reward -= 1  # penalty for trying to move out of bounds

        # Move target
        target_action = random.randint(0, 3)
        if target_action == 0:  # up
            self.target_pos = (max(0, self.target_pos[0] - 1), self.target_pos[1])
        elif target_action == 1:  # down
            self.target_pos = (min(self.grid_size - 1, self.target_pos[0] + 1), self.target_pos[1])
        elif target_action == 2:  # left
            self.target_pos = (self.target_pos[0], max(0, self.target_pos[1] - 1))
        elif target_action == 3:  # right
            self.target_pos = (self.target_pos[0], min(self.grid_size - 1, self.target_pos[1] + 1))

        # Check termination
        terminated = self.agent_pos == self.target_pos or self.step_count >= self.max_steps
        if self.agent_pos == self.target_pos:
            reward += 10

        return {"agent": self.agent_pos, "target": self.target_pos}, reward, terminated, False, {}

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.agent_pos:
                    print("A", end=" ")
                elif (i, j) == self.target_pos:
                    print("T", end=" ")
                else:
                    print(".", end=" ")
            print()

# Test the environment
env = MovingTargetEnv()
observation, info = env.reset()
env.render()
terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"Reward: {reward}, Terminated: {terminated}")


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create the environment
env = MovingTargetEnv()
eval_env = MovingTargetEnv()

# Create a vectorized environment for training
from gymnasium.wrappers import FlattenObservation
env = FlattenObservation(env)
eval_env = FlattenObservation(eval_env)
vec_env = make_vec_env(lambda: env)

# Create an evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=0.0003, batch_size=64, 
            gamma=0.99, n_steps=128, 
            n_epochs=4, clip_range=0.2)

# Train the model
model.learn(total_timesteps=10000, callback=eval_callback)

# Save the trained model
model.save("ppo_moving_target")

# Load the trained model
model = PPO.load("ppo_moving_target")

class GridSizeCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(GridSizeCallback, self).__init__(verbose)
        self.env = env
        self.grid_sizes = [3, 5, 7]
        self.current_grid_size_index = 0
        self.timesteps_per_grid_size = 3333  # Change grid size every 3333 timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps >= (self.current_grid_size_index + 1) * self.timesteps_per_grid_size:
            self.current_grid_size_index += 1
            if self.current_grid_size_index >= len(self.grid_sizes):
                return True  # Stop training if exceeded the last grid size
            new_grid_size = self.grid_sizes[self.current_grid_size_index]
            print(f"Grid size changed to {new_grid_size}x{new_grid_size}")
        return True

# Create an evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

# Create a callback to change the grid size
grid_size_callback = GridSizeCallback(env)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=0.0003, batch_size=64, 
            gamma=0.99, n_steps=128, 
            n_epochs=4, clip_range=0.2)

# Train the model
model.learn(total_timesteps=10000, callback=[eval_callback, grid_size_callback])

# Save the trained model
model.save("ppo_moving_target_curriculum")