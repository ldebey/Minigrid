import math
import operator
import time
from functools import reduce
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from minigrid.core.actions import Actions

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX, IDX_TO_OBJECT
from minigrid.core.world_object import Goal, Door

DIRECTION_FOR_AGENT = {
    "left": 0,
    "topLeft": 1,
    "top": 2,
    "topRight": 3,
    "right": 4
}


class State():
    def __init__(self, image, previous_state=None, previous_action=None, terminated=False):
        self.goal_visible = False
        self.goal_direction = None
        self.wall_front_of_agent = False
        self.wall_left_of_agent = False
        self.wall_right_of_agent = False
        self.wall_left_distance = 0
        self.wall_right_distance = 0
        self.wall_front_distance = 0
        self.previous_state = previous_state
        self.previous_action = previous_action
        self.terminated = terminated
        index = 0
        for table in image:
            if 8 in table:
                self.goal_visible = True
                if index == 6:
                    if np.where(table == 8)[0][0] < 3:
                        self.goal_direction = DIRECTION_FOR_AGENT["left"]
                    else:
                        self.goal_direction = DIRECTION_FOR_AGENT["right"]
                else:
                    if np.where(table == 8)[0][0] < 3:
                        self.goal_direction = DIRECTION_FOR_AGENT["topLeft"]
                    elif np.where(table == 8)[0][0] == 3:
                        self.goal_direction = DIRECTION_FOR_AGENT["top"]
                    else:
                        self.goal_direction = DIRECTION_FOR_AGENT["topRight"]
            index += 1

            visibility = 7

            # Recherche des murs
            for i in range(visibility):
                if image[6 - i][3][0] == 2:
                    self.wall_front_of_agent = True
                    self.wall_front_distance = 6 + i - visibility
                    break

        left_distances = []
        right_distances = []
        table = image[6]
        if 2 in table:
            if np.where(table == 2)[0][0] < 3:
                self.wall_left_of_agent = True
                left_distances.append(2 - np.where(table == 2)[0][0])
            elif np.where(table == 2)[0][0] > 3:
                self.wall_right_of_agent = True
                right_distances.append(np.where(table == 2)[0][0] - 4)

        if len(left_distances) > 0:
            self.wall_left_distance = min(left_distances)
        else:
            self.wall_left_distance = 0

        if len(right_distances) > 0:
            self.wall_right_distance = min(right_distances)
        else:
            self.wall_right_distance = 0





    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.goal_visible == other.goal_visible and
                    self.goal_direction == other.goal_direction and
                    self.wall_front_of_agent == other.wall_front_of_agent and
                    self.wall_left_of_agent == other.wall_left_of_agent and
                    self.wall_right_of_agent == other.wall_right_of_agent and
                    self.wall_left_distance == other.wall_left_distance and
                    self.wall_right_distance == other.wall_right_distance and
                    self.wall_front_distance == other.wall_front_distance and
                    self.previous_state == other.previous_state and
                    self.terminated == other.terminated)
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        return f"Is goal visible : {self.goal_visible} / Direction : {self.goal_direction} \nWall front of agent : {self.wall_front_distance} \nWall left of agent : {self.wall_left_distance} / Wall right agent : {self.wall_right_distance}"

class ReseedWrapper(Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        # print("Seed : ", seed)
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)


class ActionBonus(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        # print("Reward: %f" % reward)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.counts = {}
        return self.env.reset(**kwargs)


class StateBonus(Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.counts = {}
        return self.env.reset(**kwargs)


class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"]


class OneHotPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space["image"].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(
            self.observation_space.spaces["image"].shape, dtype="uint8")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=True, tile_size=self.tile_size)

        return {**obs, "image": rgb_img}


class RGBImgPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img_partial = self.get_frame(
            tile_size=self.tile_size, agent_pov=True)

        return {**obs, "image": rgb_img_partial}


def get_state(obj):
    if obj is None:
        return -1
    elif type(obj) == Door:
        if obj.is_locked:
            return STATE_TO_IDX["locked"]
        elif obj.is_open:
            return STATE_TO_IDX["open"]
        else:
            return STATE_TO_IDX["closed"]
    else:
        return -1


class AgentObsWrapper(ObservationWrapper):
    """
    Grille basée sur la vision de l'agent (7x7) contenant les id des objets
    """

    def __init__(self, env, tile_size=7):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        grid, vis_mask = self.gen_obs_grid()

        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else 3 for o in grid.grid]
        )

        out = np.zeros((self.tile_size, self.tile_size, 2), dtype="uint8")
        # np zeros with tuple

        for i in range(self.tile_size - 1, -1, -1):
            for j in range(self.tile_size - 1, -1, -1):
                if out[i, j, 0] == 0:
                    out[i, j, 0] = objects[i * self.tile_size + j]
                    if out[i, j, 0] == 2:
                        for k in range(i - 1, 0, -1):
                            out[k, j, 0] = 1
                    out[i, j, 1] = get_state(grid.grid[i * self.tile_size + j])

        obs["image"] = out
        return obs


class ObjectifWrapper(Wrapper):
    """
    Wrapper qui ajoute une récompense lorsque l'agent voit l'objectif
    """

    def __init__(self, env):
        super().__init__(env)
        self.doors_opened = 0
        self.doors_passed = 0
        self.doors_pos = {}
        self.front_door = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if State(obs["image"]).goal_visible:
            objectif_pos = np.where(obs['image'] == 8)
            dist = math.dist((6, 3), (objectif_pos[0][0], objectif_pos[1][0]))
            if dist > 1:
                reward += 5 / dist
            else:
                reward += 5

            #print(State(obs["image"]))

            if State(obs["image"]).goal_direction == 2:
                reward += 2

            #print(f'dist_objectif = {dist}')

        if State(obs["image"]).wall_front_of_agent and State(obs["image"]).wall_front_distance <= 1 and not State(obs["image"]).goal_direction == 2:

            reward = -1

        if 4 in obs['image']:
            doors_pos = np.where(obs['image'] == 4)
            for d in range(len(doors_pos[0])):
                dist = math.dist((6, 3), (doors_pos[0][d], doors_pos[1][d]))
                door_state = obs["image"][doors_pos[0][d]][doors_pos[1][d]][1]
                door_state_str = [k for k, v in STATE_TO_IDX.items() if v == door_state][0]
                previous_door_opened = self.doors_opened
                if action == Actions.toggle and (doors_pos[0][d], doors_pos[1][d]) == (5, 3):
                    match door_state_str:
                        case 'open':
                            self.doors_opened += 1
                        case 'closed':
                            self.doors_opened -= 1
                # si la porte est sur la même ligne que l'agent (3)
                if doors_pos[1][d] == 3 and doors_pos[0][d] < 6:
                    if door_state_str == 'open' and previous_door_opened < self.doors_opened:
                        reward += 32 / dist
                    elif door_state_str == 'closed':
                        reward += 4 / dist
                else:
                    if door_state_str == 'open' and previous_door_opened < self.doors_opened:
                        reward += 16 / dist
                    elif door_state_str == 'closed':
                        reward += 1 / dist
        reward += (self.doors_opened * 1000)
        agent_x, agent_y = self.unwrapped.agent_pos
        if (agent_x, agent_y) in self.doors_pos and action != Actions.toggle:
            if self.doors_pos[(agent_x, agent_y)][0] == obs['direction']:
                self.doors_passed += 1
            elif self.doors_pos[(agent_x, agent_y)][1] == obs['direction']:
                self.doors_passed -= 1
        reward += self.doors_passed
        # print(f'agent_pos: {agent_x, agent_y}')
        # print(f'direction: {obs["direction"]}')
        # print(f'doors_passed: {self.doors_passed}')
        if obs['image'][5][3][0] == 4:
            match obs['direction']:
                case 0:
                    if (agent_x + 1, agent_y) not in self.doors_pos:
                        self.doors_pos[(agent_x + 1, agent_y)] = (0, 2)
                case 1:
                    if (agent_x, agent_y + 1) not in self.doors_pos:
                        self.doors_pos[(agent_x, agent_y + 1)] = (1, 3)
                case 2:
                    if (agent_x - 1, agent_y) not in self.doors_pos:
                        self.doors_pos[(agent_x - 1, agent_y)] = (2, 0)
                case 3:
                    if (agent_x, agent_y - 1) not in self.doors_pos:
                        self.doors_pos[(agent_x, agent_y - 1)] = (3, 1)

        if terminated:
            reward += 10
            self.doors_opened = 0

        # print("reward: ", reward)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.doors_opened = 0
        self.doors_passed = 0
        self.doors_pos = {}
        return self.env.reset(**kwargs)


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "image": full_grid}


class DictObservationSpaceWrapper(ObservationWrapper):
    """
    Transforms the observation space (that has a textual component) to a fully numerical observation space,
    where the textual instructions are replaced by arrays representing the indices of each word in a fixed vocabulary.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.
    """

    def __init__(self, env, max_words_in_mission=50, word_dict=None):
        """
        max_words_in_mission is the length of the array to represent a mission, value 0 for missing words
        word_dict is a dictionary of words to use (keys=words, values=indices from 1 to < max_words_in_mission),
                  if None, use the Minigrid language
        """
        super().__init__(env)

        if word_dict is None:
            word_dict = self.get_minigrid_words()

        self.max_words_in_mission = max_words_in_mission
        self.word_dict = word_dict

        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": spaces.MultiDiscrete(
                    [len(self.word_dict.keys())] * max_words_in_mission
                ),
            }
        )

    @staticmethod
    def get_minigrid_words():
        colors = ["red", "green", "blue", "yellow", "purple", "grey"]
        objects = [
            "unseen",
            "empty",
            "wall",
            "floor",
            "box",
            "key",
            "ball",
            "door",
            "goal",
            "agent",
            "lava",
        ]

        verbs = [
            "pick",
            "avoid",
            "get",
            "find",
            "put",
            "use",
            "open",
            "go",
            "fetch",
            "reach",
            "unlock",
            "traverse",
        ]

        extra_words = [
            "up",
            "the",
            "a",
            "at",
            ",",
            "square",
            "and",
            "then",
            "to",
            "of",
            "rooms",
            "near",
            "opening",
            "must",
            "you",
            "matching",
            "end",
            "hallway",
            "object",
            "from",
            "room",
        ]

        all_words = colors + objects + verbs + extra_words
        assert len(all_words) == len(set(all_words))
        return {word: i for i, word in enumerate(all_words)}

    def string_to_indices(self, string, offset=1):
        """
        Convert a string to a list of indices.
        """
        indices = []
        # adding space before and after commas
        string = string.replace(",", " , ")
        for word in string.split():
            if word in self.word_dict.keys():
                indices.append(self.word_dict[word] + offset)
            else:
                raise ValueError(f"Unknown word: {word}")
        return indices

    def observation(self, obs):
        obs["mission"] = self.string_to_indices(obs["mission"])
        assert len(obs["mission"]) < self.max_words_in_mission
        obs["mission"] += [0] * \
                          (self.max_words_in_mission - len(obs["mission"]))

        return obs


class FlatObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (
                    len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            strArray = np.zeros(
                shape=(self.maxStrLen, self.numCharCodes), dtype="float32"
            )

            for idx, ch in enumerate(mission):
                if ch >= "a" and ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, "%s : %d" % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs


class ViewSizeWrapper(Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "image": image}


class DirectionObsWrapper(ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """

    def __init__(self, env, type="slope"):
        super().__init__(env)
        self.goal_position: tuple = None
        self.type = type

    def reset(self):
        obs = self.env.reset()
        if not self.goal_position:
            self.goal_position = [
                x for x, y in enumerate(self.grid.grid) if isinstance(y, Goal)
            ]
            # in case there are multiple goals , needs to be handled for other env types
            if len(self.goal_position) >= 1:
                self.goal_position = (
                    int(self.goal_position[0] / self.height),
                    self.goal_position[0] % self.width,
                )
        return obs

    def observation(self, obs):
        slope = np.divide(
            self.goal_position[1] - self.agent_pos[1],
            self.goal_position[0] - self.agent_pos[0],
        )
        obs["goal_direction"] = np.arctan(
            slope) if self.type == "angle" else slope
        return obs


class SymbolicObsWrapper(ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
        w, h = self.width, self.height
        grid = np.mgrid[:w, :h]
        grid = np.concatenate([grid, objects.reshape(1, w, h)])
        grid = np.transpose(grid, (1, 2, 0))
        obs["image"] = grid
        return obs


# Q - Table Rewards

class QLearningWrapper:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0):
        self.env = env
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def get_state(self, observation, previous_state=None, previous_action=None, terminated=False):
        image = observation['image']
        return State(image,previous_state,previous_action,terminated)

    def redraw(self, window, img):
        window.show_img(img)

    def get_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate :
            # Exploration : choisir une action aléatoire
            action = self.env.action_space.sample()
        else:
            # Exploitation : choisir l'action avec la plus grande valeur Q
            if state not in self.q_table:
                # Si l'état n'a jamais été vu, initialiser les valeurs Q à 0 pour toutes les actions possibles
                self.q_table[state] = np.zeros(self.env.action_space.n)
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            if np.sum(q_values == max_q) > 1:
                # Si plusieurs actions ont la même valeur Q max, choisir une action aléatoire parmi celles-ci
                best_actions = np.argwhere(q_values == max_q).flatten()
                action = np.random.choice(best_actions)
            else:
                # Sinon, choisir l'action avec la plus grande valeur Q
                action = np.argmax(q_values)

        if state.previous_action is not None:
            if action > 2 and state.previous_action > 2:
                return self.get_action(state)

        return action

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            # Si l'état n'a jamais été vu, initialiser les valeurs Q à 0 pour toutes les actions possibles
            self.q_table[state] = np.zeros(self.env.action_space.n)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.env.action_space.n)
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value
        # Si une porte fermée est devant l'agent, lui donner une récompense positive
        # Selon la position de l'agent, la porte peut-être à gauche ou à droite ou en haut ou en bas
        # match self.env.agent_dir:
        #     case 0:  # right
        #         if self.env.grid.get(self.env.agent_pos[0] + 1, self.env.agent_pos[1]) == Door:
        #             self.q_table[next_state][self.env.unwrapped.actions.forward] += 1
        #     case 1:  # down
        #         if self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1] + 1) == Door:
        #             self.q_table[next_state][self.env.unwrapped.actions.forward] += 1
        #     case 2:  # left
        #         if self.env.grid.get(self.env.agent_pos[0] - 1, self.env.agent_pos[1]) == Door:
        #             self.q_table[next_state][self.env.unwrapped.actions.forward] += 1
        #     case 3:  # up
        #         if self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1] - 1) == Door:
        #             self.q_table[next_state][self.env.unwrapped.actions.forward] += 1



    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            print(f"Episode {episode}")
            state = self.get_state(self.env.reset()[0])
            done = False
            truncated = False
            previous_action = None
            while not done and not truncated:
                action = self.get_action(state)
                observation, reward, done, truncated, _ = self.env.step(action)
                next_state = self.get_state(observation, previous_action=previous_action, terminated=done)

                self.update_q_table(state, action, reward, next_state)
                state = next_state
                previous_action = action

                #print(f"Reward: {reward}")

            # Diminuer le taux d'exploration après chaque épisode
            self.exploration_rate -= self.exploration_rate / num_episodes

    def test(self, window, num_episodes=100):
        rewards = []
        for episode in range(num_episodes):
            state = self.get_state(self.env.reset()[0])
            done = False
            truncated = False
            total_reward = 0
            previous_action = None

            if not state.goal_visible:
                self.exploration_rate = 0.4
            else:
                self.exploration_rate = 0

            while not done and not truncated:
                action = self.get_action(state)
                observation, reward, done, truncated, _ = self.env.step(action)
                next_state = self.get_state(observation, previous_action=previous_action, terminated=done)

                print(f"Reward: {reward}")

                state = next_state
                total_reward += reward
                previous_action = action

                self.redraw(window, self.env.get_frame())

            rewards.append(total_reward)

        avg_reward = sum(rewards) / num_episodes
        print(f"Average reward over {num_episodes} test episodes: {avg_reward}")


class RewardWrapper(Wrapper):
    """
    Calculates the reward for the agent.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        agent_pos = self.env.agent_pos
        image_object = obs["image"][:, :, 0]
        mission_pos = np.where(image_object == 8)
        if mission_pos[0].shape[0] != 0 and mission_pos[1].shape[0] != 0:
            mission_pos = (mission_pos[0][0], mission_pos[1][0])
            dist = math.dist(agent_pos, mission_pos)
            # print(f"a: {agent_pos}, m: {mission_pos}, dist: {dist}, direction: {obs['direction']}")
            reward = 0.5 / dist
            match obs['direction']:
                case 0:  # right
                    if agent_pos[0] < mission_pos[0] and agent_pos[1] == mission_pos[1]:
                        reward += 0.25 + 0.25 / \
                                  (abs(agent_pos[0] - mission_pos[0]))
                    else:
                        if agent_pos[0] < mission_pos[0]:
                            reward += 0.25
                    if image_object[agent_pos[0] + 1, agent_pos[1]] == 0:
                        reward /= 2
                case 1:  # down
                    if agent_pos[1] < mission_pos[1] and agent_pos[0] == mission_pos[0]:
                        reward += 0.25 + 0.25 / \
                                  (abs(agent_pos[1] - mission_pos[1]))
                    else:
                        if agent_pos[1] < mission_pos[1]:
                            reward += 0.25 / \
                                      (abs(agent_pos[1] - mission_pos[1]))
                    if image_object[agent_pos[0]][agent_pos[1] + 1] == 2:
                        reward /= 2
                case 2:  # left
                    if agent_pos[0] > mission_pos[0] and agent_pos[1] == mission_pos[1]:
                        reward += 0.25 + 0.25 / \
                                  (abs(agent_pos[0] - mission_pos[0]))
                    else:
                        if agent_pos[0] > mission_pos[0]:
                            reward += 0.25 / \
                                      (abs(agent_pos[0] - mission_pos[0]))
                    if image_object[agent_pos[0] - 1][agent_pos[1]] == 2:
                        reward /= 2
                case 3:  # up
                    if agent_pos[1] > mission_pos[1] and agent_pos[0] == mission_pos[0]:
                        reward += 0.25 + 0.25 / \
                                  (abs(agent_pos[1] - mission_pos[1]))
                    else:
                        if agent_pos[1] > mission_pos[1]:
                            reward += 0.25 / \
                                      (abs(agent_pos[1] - mission_pos[1]))
                    if image_object[agent_pos[0]][agent_pos[1] - 1] == 2:
                        reward /= 2
            if reward == 1:
                reward = 0.99
            return obs, reward, terminated, truncated, info
        else:
            return obs, 1, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
