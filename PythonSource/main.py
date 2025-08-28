import torch
import logging
from tqdm import trange

from MadelAIneAgent import MadelAIneAgent

from CelesteEnv import CelesteEnv

import os
import gymnasium as gym

from utils import *

import numpy as np

import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# Usage:
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = TqdmLoggingHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]  # Replace existing handlers

# Now use logging.info(), logging.warning(), etc. as usual

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

SHOULD_TRAIN = True
CKPT_SAVE_INTERVAL = 1000
NUM_OF_EPISODES = 100000

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

for i in trange(NUM_OF_EPISODES, desc="Training Episodes"):
    done = False
    state, _ = env.reset()
    state_flatten = gym.spaces.utils.flatten(env.observation_space, state)
    total_reward = 0
    batch_losses = []
    while not done:
        a = agent.choose_action(state_flatten)
        action_multi_binary = [int(x) for x in format(a, f'0{env.action_space.n}b')]
        action = np.array(action_multi_binary, dtype=np.float32)

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        new_state_flatten = gym.spaces.utils.flatten(env.observation_space, new_state)

        if SHOULD_TRAIN:
            agent.store_in_memory(state_flatten, a, reward, new_state_flatten, done)
            loss = agent.learn()
            if loss is not None:
                batch_losses.append(loss)

        state_flatten = new_state_flatten

    avg_loss = np.mean(batch_losses) if batch_losses else None
    logging.info(f"Total reward: {total_reward} | Avg loss: {avg_loss} | Epsilon: {agent.epsilon} | Replay buffer size: {len(agent.replay_buffer)} | Learn step counter: {agent.learn_step_counter}")

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

env.close()