import torch

from MadelAIneAgent import MadelAIneAgent

from CelesteEnv import CelesteEnv

import os
import gymnasium as gym

from utils import *

import numpy as np

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

SHOULD_TRAIN = True
CKPT_SAVE_INTERVAL = 100
NUM_OF_EPISODES = 1

env = CelesteEnv()

input_dims = gym.spaces.utils.flatdim(env.observation_space)
num_actions = 2 ** env.action_space.n
agent = MadelAIneAgent(input_dims=input_dims, num_actions=num_actions)

if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()

for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False
    state_dict, _ = env.reset()
    state = gym.spaces.utils.flatten(env.observation_space, state_dict)
    total_reward = 0
    while not done:
        a = agent.choose_action(state)
        action_multi_binary = [int(x) for x in format(a, f'0{env.action_space.n}b')]
        action = np.array(action_multi_binary, dtype=np.float32)

        new_state, reward, terminated, truncated, info  = env.step(action)
        done = terminated or truncated
        total_reward += reward

        new_state = gym.spaces.utils.flatten(env.observation_space, new_state)
        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    print("Total reward:", total_reward)

env.close()