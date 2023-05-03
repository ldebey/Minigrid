#!/usr/bin/env python3
from argparse import ArgumentParser

from minigrid.utils.window import Window
from minigrid.wrappers import QLearningWrapper, AgentObsWrapper, ObjectifWrapper
import gymnasium as gym
import matplotlib

matplotlib.use('TkAgg')

# Create the environment
parser = ArgumentParser()
parser.add_argument("--env", help="gym environment to load", default="MiniGrid-FourRooms-v0")
parser.add_argument("--train", help="number of episode during training", default=50)
parser.add_argument("--test", help="number of episode during testing", default=10)
args = parser.parse_args()
# Create the environment
env = gym.make(args.env)
env = ObjectifWrapper(env)
env = AgentObsWrapper(env)

# Create the Q-learning wrapper
ql_wrapper = QLearningWrapper(env)

# Train the agent

window = Window("Level 1.5")
ql_wrapper.train(num_episodes=args.train)

# Test the agent
ql_wrapper.test(num_episodes=args.test, window=window)

window.show(True)