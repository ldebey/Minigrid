#!/usr/bin/env python3
from minigrid.utils.window import Window
from minigrid.wrappers import QLearningWrapper, State, RGBImgObsWrapper
import gymnasium as gym
import matplotlib

matplotlib.use('TkAgg')

# Create the environment
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgObsWrapper(env)

# Create the Q-learning wrapper
ql_wrapper = QLearningWrapper(env)

# Train the agent

window = Window("Level 1.5")
ql_wrapper.train(num_episodes=50)

# Test the agent
ql_wrapper.test(num_episodes=10,window=window)

window.show(True)