import socket
import json
import pynput.keyboard
import numpy as np
import time
from typing import Optional
import logging

from CelesteInputs import CelesteInputs
import gymnasium as gym

logging.basicConfig(level=logging.INFO)

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

        logging.info(f"Waiting for connection from C# client on {CelesteEnv.TCP_IP}:{CelesteEnv.TCP_PORT}...")
        self._conn, self._addr = self._server_sock.accept()
        logging.info(f"Connected to {self._addr}")

        self._json_data = None
        self._celeste_inputs = CelesteInputs()

        self._steps = 0
        self._prev_distance = None

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

    def close(self):
        logging.debug("Closing environment")
        self._conn.close()
        self._server_sock.close()
        self._celeste_inputs.reset_keyboard()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        logging.debug("Resetting environment")

        self._options = options
        
        if options and "reward_mode" in options and options["reward_mode"] == "delta":
            self._use_delta_dist = True
        else:
            self._use_delta_dist = False

        # Clear keyboard state
        self._celeste_inputs.reset_keyboard()

        self._steps = 0

        # Receive a dummy message and send the reset message
        self._conn.recv(self.BUFFER_SIZE)
        reset_msg = json.dumps({"type": "reset"}).encode('utf-8')
        self._conn.sendall(reset_msg)

        observation = self._get_obs()
        info = self._get_info()

        self._starting_distance = info["distance"] if info["distance"] is not None else float('999999')
        self._best_distance = self._starting_distance
        self._prev_distance = self._starting_distance

        # DEBUG: Write JSON data to file
        with open("debug.json", "w") as f:
            json.dump(self._json_data, f)

        return observation, info

    def step(self, action):        
        # Perform desired action by updating keyboard state
        self._celeste_inputs = CelesteInputs.from_action(action)
        # self._celeste_inputs.update_keyboard()

        # Request the game state
        observation = self._get_obs()
        info = self._get_info()

        # TODO: Terminate if reached the next room
        terminated = info["playerDied"] if info is not None and "playerDied" in info else False
        if terminated:
            logging.debug("Episode terminated: player died")

        # Truncate after 30 seconds
        truncated = self._steps >= 1800
        if truncated:
            logging.debug("Episode truncated: time limit reached")

        # Reward is inversely proportional to distance from target
        distance = info["distance"] if info["distance"] is not None else float('inf')
        
        if self._use_delta_dist:
            reward = -info["distance_gain"]
        else:
            # Keep track of the best distance to the target. If it improves, update the reward
            if self._best_distance is None or distance < self._best_distance:
                reward = self._best_distance - distance
                self._best_distance = distance
                logging.debug(f"New best distance: {self._best_distance:.2f}")
            else:
                reward = 0

        # Big reward for making it to the next room
        if info is not None and "nextRoom" in info and info["nextRoom"]:
            reward = reward + 50.0
            terminated = True
            logging.debug("Episode terminated: reached next room")

        # Penalize if died
        if info is not None and "playerDied" in info and info["playerDied"]:
            reward = reward - 10.0

        # Penalize for each step taken
        reward = reward - 0.1

        self._steps += 1


        logging.debug(f"Finished step {self._steps}")
        self._prev_distance = distance

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
                        # Send ACK after successful receipt
                        ack_msg = json.dumps({"type": "ACK"}).encode('utf-8')
                        self._conn.sendall(ack_msg)
                    except json.JSONDecodeError:
                        logging.warning(f"Received invalid JSON from {self._addr}: {data}")
                        self._json_data = None
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
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

        return observation
    
    def _get_info(self):
        """Compute auxiliary information for debugging.
        
        Returns:
            dict: Info with distance between agent and target
        """
        if self._json_data is not None:
            distance = np.linalg.norm(
                    np.array([self._json_data["playerXPosition"], self._json_data["playerYPosition"]], dtype=np.float32) -
                    np.array([self._json_data["targetXPosition"], self._json_data["targetYPosition"]], dtype=np.float32)
            )
            if self._prev_distance:
                distance_gain = self._prev_distance - distance
            else:
                distance_gain = 0.0
            return {
                "distance": distance,
                "distance_gain": distance_gain,
                "steps": self._steps,
                "playerDied": self._json_data["playerDied"] if "playerDied" in self._json_data else False,
                "nextRoom": self._json_data["nextRoom"] if "nextRoom" in self._json_data else False
            }
        else:
            return None