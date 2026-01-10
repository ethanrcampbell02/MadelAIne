import numpy as np
from gymnasium import Wrapper
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym


class SimplifiedActionSpace(Wrapper):
    """
    Simplifies the action space to 5 discrete actions with grab always held:
    0: Idle (grab only)
    1: Jump (grab + jump)
    2: Up (grab + up)
    3: Right (grab + right)
    4: Right + Jump (grab + right + jump)
    """
    def __init__(self, env):
        super().__init__(env)
        # Change action space from MultiBinary(7) to Discrete(5)
        self.action_space = gym.spaces.Discrete(5)
        
        # Action mapping: each index maps to [up, down, left, right, jump, dash, grab]
        self.action_mapping = {
            0: [0, 0, 0, 0, 0, 0, 1],  # Idle (grab only)
            1: [0, 0, 0, 0, 1, 0, 1],  # Jump
            2: [1, 0, 0, 0, 0, 0, 1],  # Up
            3: [0, 0, 0, 1, 0, 0, 1],  # Right
            4: [0, 0, 0, 1, 1, 0, 1],  # Right + Jump
        }
    
    def step(self, action):
        # Convert discrete action to multi-binary action
        multi_binary_action = np.array(self.action_mapping[action], dtype=np.float32)
        return self.env.step(multi_binary_action)


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info
    
class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)
    

def apply_wrappers(env):
    env = SimplifiedActionSpace(env)  # Reduce action space to 5 discrete actions
    env = SkipFrame(env, skip=4) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=(160, 90)) # Resize frame from 320x180 to 160x90
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    return env