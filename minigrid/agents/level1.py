#!/usr/bin/env python3
import matplotlib
import pandas as pd
import numpy as np
import random

matplotlib.use('TkAgg')
import gymnasium as gym

from minigrid.utils.window import Window
from minigrid.wrappers import RewardWrapper ,QTableRewardBonus, FullyObsWrapper


def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None):
    env.reset(seed=seed)

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    img = env.get_frame()

    redraw(window, img)

def step(env, window):
    obs, reward, terminated, truncated, info = env.step()
    #print(reward)

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


# def step(env, window, action):
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(reward)
#     #print(obs["image"][:,:,0]) 

#     #obs => image = tableau de 7x7x3 contenant des entiers représentant les object vu par l'agent
#     #obs => direction = de 0 à 3 sachant que 0 = droite, 1 = bas, 2 = gauche, 3 = haut
#     #terminated = boolean qui passe à vrai lorsque l'agent tombe dans la lave ou sur la sortie
#     #truncated = boolean qui passe à vrai lorsque step_count >= max_step_count

#     # print(f"obs : image={obs['image']}, direction={obs['direction']},mission={obs['mission']}")

#     if terminated:
#         print("terminated!")
#         reset(env, window)
#     elif truncated:
#         print("truncated!")
#         reset(env, window)
#     else:
#         img = env.get_frame()
#         redraw(window, img)

#     return terminated

#Exploration function

# def explore(env, window):
#     terminated = False
#     while not terminated:
#         val = random.randint(0, 3)
#         if val == 0:
#             terminated = step(env, window, env.actions.left)
#         elif val == 1:
#             terminated= step(env, window, env.actions.right)
#         elif val == 2:
#             terminated = step(env, window, env.actions.forward)

def start_learning(env,window):
    env.epsilon = 1
    while env.epsilon > 0:
        print("epsilon = ",env.epsilon)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            terminated, truncated = step(env,window)
        env.epsilon -= 0.01
    env.show_q_table()

def key_handler(env, window, event):
    print("pressed", event.key)

    # Spacebar
    if event.key == " ":
        start_learning(env,window)
        return


if __name__ == "__main__":
    # Create the environment
    env = gym.make("MiniGrid-Empty-8x8-v0")

    env = FullyObsWrapper(env)
    env = RewardWrapper(env)
    env = QTableRewardBonus(env)

    window = Window("Level 1")
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    reset(env, window)

    # Blocking event loop
    window.show(block=True)
