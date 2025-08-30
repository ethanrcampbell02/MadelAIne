import numpy as np
from gymnasium import Wrapper
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation


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
    

def apply_wrappers(env):
    env = SkipFrame(env, skip=4) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=(160, 90)) # Resize frame from 320x180 to 160x90
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    return env