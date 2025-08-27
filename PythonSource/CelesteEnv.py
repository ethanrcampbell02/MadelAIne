import socket
import json
import pynput.keyboard
import numpy as np
import time
from typing import Optional

from CelesteInputs import CelesteInputs
import gymnasium as gym

class CelesteEnv(gym.Env):

    TCP_IP = "127.0.0.1"
    TCP_PORT = 5000
    BUFFER_SIZE = 4096

    def __init__(self):
        super().__init__()

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._server_sock.bind((CelesteEnv.TCP_IP, CelesteEnv.TCP_PORT))
        self._server_sock.listen(1)

        print(f"Waiting for connection from C# client on {CelesteEnv.TCP_IP}:{CelesteEnv.TCP_PORT}...")
        self._conn, self._addr = self._server_sock.accept()
        print(f"Connected to {self._addr}")

        self._json_data = None
        self._celeste_inputs = CelesteInputs()

        self._steps = 0

        self.observation_space = gym.spaces.Dict(
            {
            "playerPosition": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "playerVelocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "playerCanDash": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.bool),
            "playerStamina": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "targetPosition": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "roomTileData": gym.spaces.MultiBinary((23, 40))
            }
        )

        self.action_space = gym.spaces.MultiBinary(7)  # up, down, left, right, jump, dash, grab


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        print("Resetting environment")

        self._options = options

        # Perform keyboard sequence to restart chapter
        self._celeste_inputs.reset_keyboard()
        # self._celeste_inputs.restart_chapter_celeste() TODO: Restart chapter more elegantly

        self._steps = 0

        observation = self._get_obs()
        info = self._get_info()

        # DEBUG: Write JSON data to file
        with open("debug.json", "w") as f:
            json.dump(self._json_data, f)

        return observation, info

    def step(self, action):        
        # Perform desired action by updating keyboard state
        self._celeste_inputs = CelesteInputs.from_action(action)
        self._celeste_inputs.update_keyboard()

        # Request the game state
        observation = self._get_obs()
        info = self._get_info()

        # TODO: Terminate if the agent died or reached the next room
        terminated = False

        # Truncate after 10 seconds (600 steps)
        self._steps += 1
        truncated = self._steps > 600

        # Reward is inversely proportional to distance from target
        distance = info["distance"] if info["distance"] is not None else float('inf')
        reward = -0.1 * distance

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # If in JSON debug mode, just read from the JSON file
        if self._options is not None and "json_debug" in self._options and self._options["json_debug"]:
            with open("debug.json", "r") as f:
                self._json_data = json.load(f)
        else:
            try:
                data = self._conn.recv(self.BUFFER_SIZE)
                if not data:
                    self._json_data = None
                else:
                    try:
                        self._json_data = json.loads(data.decode('utf-8'))
                    except json.JSONDecodeError:
                        print(f"Received invalid JSON from {self._addr}: {data}")
                        self._json_data = None
            except Exception as e:
                print(f"Error receiving data: {e}")
                self._json_data = None

        if self._json_data is None:
            return None

        observation = {
            "playerPosition": np.array([self._json_data["playerXPosition"], self._json_data["playerYPosition"]], dtype=np.float32),
            "playerVelocity": np.array([self._json_data["playerXVelocity"], self._json_data["playerYVelocity"]], dtype=np.float32),
            "playerCanDash": np.array([self._json_data["playerCanDash"]], dtype=np.bool),
            "playerStamina": np.array([self._json_data["playerStamina"]], dtype=np.float32),
            "targetPosition": np.array([self._json_data["targetXPosition"], self._json_data["targetYPosition"]], dtype=np.float32),
            "roomTileData": np.array(self._json_data["solidTileData"], dtype=np.bool)
        }

        print(f"Finished step {self._steps}")

        return observation
    
    def _get_info(self):
        """Compute auxiliary information for debugging.
        
        Returns:
            dict: Info with distance between agent and target
        """
        if self._json_data is not None:
            return {
                "distance": np.linalg.norm(
                    np.array([self._json_data["playerXPosition"], self._json_data["playerYPosition"]], dtype=np.float32) -
                    np.array([self._json_data["targetXPosition"], self._json_data["targetYPosition"]], dtype=np.float32)
                ),
                "steps": self._steps
            }
        else:
            return None