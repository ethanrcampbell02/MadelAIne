import os
import numpy as np
from datetime import datetime

from CelesteEnv import CelesteEnv
from wrappers import apply_wrappers

def run_random_policy_and_save_states(env, num_steps=1000):
    states = []
    obs, info = env.reset()  # Unpack the tuple returned by reset()
    for _ in range(num_steps):
        if obs is not None:  # Only append valid observations
            states.append(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            obs, info = env.reset()  # Unpack the tuple returned by reset()
    # Stack states into a single array
    if len(states) > 0:
        states_array = np.stack(states)
        os.makedirs("holdouts", exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"holdouts/{current_time}_holdouts.npy"
        np.save(output_path, states_array)
        print(f"Saved {len(states)} states to {output_path}")
    else:
        print("No valid states collected")

if __name__ == "__main__":
    env = CelesteEnv()
    env = apply_wrappers(env)
    run_random_policy_and_save_states(env)
