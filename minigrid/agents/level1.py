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

def step_learning(env, window):
    obs, reward, terminated, truncated, info = env.step()
    #print(reward)

    if terminated:
        #print("terminated!")
        reset(env, window)
    elif truncated:
        #print("truncated!")
        reset(env, window)

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
    for i in range(100):
        #print("epsilon = ",env.epsilon)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            terminated, truncated = step_learning(env,window)
        env.epsilon = env.epsilon - env.epsilon * (i/100)
    print("=====================================")
    print("Entrainement terminé")

    print(env.q_table)

def start_exploiting(env,window):
    env.epsilon = 0.1
    terminated = False
    truncated = False
    while not terminated and not truncated:
        terminated, truncated = step(env,window)

def key_handler(env, window, event):
    print("pressed", event.key)

    # Spacebar
    if event.key == " ":
        start_exploiting(env,window)
        return


if __name__ == "__main__":
    # Create the environment
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-FourRooms-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    env = gym.make(
        args.env,
        tile_size=args.tile_size,
    )

    env = FullyObsWrapper(env)
    env = RewardWrapper(env)
    env = QTableRewardBonus(env)

    window = Window("Level 1")
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    reset(env, window)

    start_learning(env,window)

    # Blocking event loop
    window.show(block=True)
