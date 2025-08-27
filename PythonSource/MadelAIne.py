from CelesteEnv import CelesteEnv

env = CelesteEnv()
options = {"json_debug": False}
obs = env.reset(options=options)
done = False

while not done:
    action = env.action_space.sample()  # Replace with your agent's action if available
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()