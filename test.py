# test.py
import gymnasium as gym
import gym_genesis
import numpy as np
import imageio

num_envs = 1
env = gym.make("gym_genesis/CubePick-v0", enable_pixels=True, num_envs=num_envs)
obs, info = env.reset()
frames = []

for _ in range(1000):
    # sample a batch of actions
    actions = np.stack([env.action_space.sample() for _ in range(num_envs)])
    obs, reward, terminated, truncated, info = env.step(actions)

    # render returns a single image representing all envs
    image = env.render()
    frames.append(image)

    # reset if any env is done
    if np.any(terminated) or np.any(truncated):
        obs, info = env.reset()

imageio.mimsave("test.mp4", np.stack(frames), fps=25)