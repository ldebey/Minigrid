#!/usr/bin/env python3
from minigrid.wrappers import AgentObsWrapper, ActionBonus, QTableRewardBonus, ObjectifWrapper, ReseedWrapper
from minigrid.utils.window import Window
import gymnasium as gym
import matplotlib
import pandas as pd
import numpy as np
import random

matplotlib.use('TkAgg')


def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None):
    env.reset(seed=seed)

    if hasattr(env, "mission"):
        window.set_caption(env.mission)

    img = env.get_frame()

    redraw(window, img)


def step(env, window):
    obs, reward, terminated, truncated, info = env.step()
    # print(reward)

    if terminated:
        print("terminated!")
        reset(env, window)
    elif truncated:
        print("truncated!")
        reset(env, window)
    else:
        img = env.get_frame()
        redraw(window, img)

    return terminated, truncated


def step_learning(env, window):
    obs, reward, terminated, truncated, info = env.step()
    # print(reward)

    if terminated:
        # print("terminated!")
        reset(env, window)
    elif truncated:
        # print("truncated!")
        reset(env, window)

    return terminated, truncated


def start_learning(env, window):
    env.epsilon = 1
    for i in range(100):
        # print("epsilon = ",env.epsilon)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            terminated, truncated = step_learning(env, window)
        env.epsilon = env.epsilon - env.epsilon * (i/100)
    print("=====================================")
    print("Entrainement terminé")


def start_exploiting(env, window):
    env.epsilon = 0.1
    terminated = False
    truncated = False
    while not terminated and not truncated:
        terminated, truncated = step(env, window)


def key_handler(env, window, event):
    print("pressed", event.key)

    # Spacebar
    if event.key == " ":
        start_exploiting(env, window)
        return


if __name__ == "__main__":
    # Create the environment
    env = gym.make("MiniGrid-FourRooms-v0")

    env = ReseedWrapper(env)
    env = AgentObsWrapper(env)
    env = ActionBonus(env)
    env = ObjectifWrapper(env)
    env = QTableRewardBonus(env)

    window = Window("Level 1.5")
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    reset(env, window)

    start_learning(env, window)

    # Blocking event loop
    window.show(block=True)
