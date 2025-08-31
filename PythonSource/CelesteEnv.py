import base64
import socket
import json
import numpy as np
from typing import Optional
import logging
from CelesteInputs import CelesteInputs
import gymnasium as gym

logging.basicConfig(level=logging.INFO)

class CelesteEnv(gym.Env):

    TCP_IP = "127.0.0.1"
    TCP_PORT = 5000
    BUFFER_SIZE = 2**19

    def __init__(self, reward_mode="best"):
        super().__init__()

        self.reward_mode = reward_mode

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

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8)

        self.action_space = gym.spaces.MultiBinary(7)  # up, down, left, right, jump, dash, grab

    def close(self):
        logging.debug("Closing environment")
        self._conn.close()
        self._server_sock.close()
        self._celeste_inputs.reset_keyboard()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        logging.debug("Resetting environment")

        self._options = options

        # Perform keyboard sequence to restart chapter
        self._celeste_inputs.reset_keyboard()

        self._steps = 0

        # Receive a dummy message and send the reset message
        self._conn.recv(self.BUFFER_SIZE)
        reset_msg = json.dumps({"type": "reset"}).encode('utf-8')
        self._conn.sendall(reset_msg)

        observation = self._get_obs()
        info = self._get_info()

        self._starting_distance = info["distance"] if info["distance"] is not None else 500.0
        self._prev_distance = self._starting_distance
        self._best_distance = self._starting_distance

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

        # TODO: Terminate if reached the next room
        terminated = info["playerDied"] if info is not None and "playerDied" in info else False
        if terminated:
            logging.debug("Episode terminated: player died")

        # Truncate after 15 seconds
        truncated = self._steps >= 900
        if truncated:
            logging.debug("Episode truncated: time limit reached")

        reward = 0

        distance = info["distance"] if info["distance"] is not None else float('inf')

        # Compute the reward differently depending on reward mode
        if self.reward_mode == "prev":
            reward += self._prev_distance - distance
        elif self.reward_mode == "prev_positive":
            if distance < self._prev_distance:
                reward += self._prev_distance - distance
        elif self.reward_mode == "best":
            if distance < self._best_distance:
                reward += self._best_distance - distance

        # Update previous and best distances
        self._prev_distance = distance
        if distance < self._best_distance:
            self._best_distance = distance

        # Big reward for making it to the next room
        if info is not None and "playerReachedNextRoom" in info and info["playerReachedNextRoom"]:
            reward = reward + 50.0
            terminated = True
            logging.debug("Episode terminated: reached next room")

        # Penalize if died
        if info is not None and "playerDied" in info and info["playerDied"]:
            reward = reward - 20.0

        # Penalize for each step taken
        reward = reward - 0.2

        self._steps += 1

        logging.debug(f"Finished step {self._steps}")

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # If in JSON debug mode, just read from the JSON file
        if self._options is not None and "json_debug" in self._options and self._options["json_debug"]:
            with open("debug.json", "r") as f:
                self._json_data = json.load(f)
        else:
            data = b""
            max_receipts = 5
            for attempt in range(max_receipts):
                try:
                    chunk = self._conn.recv(self.BUFFER_SIZE)
                    if not chunk:
                        break
                    data += chunk
                    try:
                        self._json_data = json.loads(data.decode('utf-8'))
                        # Send ACK after successful receipt
                        ack_msg = json.dumps({"type": "ACK"}).encode('utf-8')
                        self._conn.sendall(ack_msg)
                        break
                    except json.JSONDecodeError:
                        if attempt == max_receipts - 1:
                            truncated_data = data[:200] + b' ... ' + data[-200:] if len(data) > 400 else data
                            logging.warning(f"Received invalid JSON after {max_receipts} attempts from {self._addr}: {truncated_data}")
                            self._json_data = None
                except Exception as e:
                    logging.error(f"Error receiving data: {e}")
                    self._json_data = None
                    break

        if self._json_data is None:
            return None

        img_base64 = self._json_data["screenPixelsBase64"] if "screenPixelsBase64" in self._json_data else None
        width = self._json_data["screenWidth"] if "screenWidth" in self._json_data else 320
        height = self._json_data["screenHeight"] if "screenHeight" in self._json_data else 180
        if img_base64 is not None:
            observation = self._parse_image_base64(img_base64, width, height)
        else:
            observation = None

        return observation

    @staticmethod
    def _parse_image_base64(img_base64, width, height):
        img_data = base64.b64decode(img_base64)
        return np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 4))[:,:,:3]

    def _get_info(self):
        if self._json_data is not None:
            return {
                "distance": np.linalg.norm(
                    np.array([self._json_data["playerXPosition"], self._json_data["playerYPosition"]], dtype=np.float32) -
                    np.array([self._json_data["targetXPosition"], self._json_data["targetYPosition"]], dtype=np.float32)
                ),
                "steps": self._steps,
                "playerDied": self._json_data["playerDied"] if "playerDied" in self._json_data else False,
                "playerReachedNextRoom": self._json_data["playerReachedNextRoom"] if "playerReachedNextRoom" in self._json_data else False
            }
        else:
            return None