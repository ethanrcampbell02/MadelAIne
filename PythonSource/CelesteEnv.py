import base64
import socket
import json
import numpy as np
from typing import Optional
import logging
from CelesteInputs import CelesteInputs
import gymnasium as gym
import cv2

logging.basicConfig(level=logging.INFO)

class CelesteEnv(gym.Env):

    TCP_IP = "127.0.0.1"
    TCP_PORT = 5000
    BUFFER_SIZE = 2**19

    def __init__(self, reward_mode="best", render_mode="human"):
        super().__init__()

        self.reward_mode = reward_mode
        self.render_mode = render_mode

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._server_sock.bind((CelesteEnv.TCP_IP, CelesteEnv.TCP_PORT))
        self._server_sock.listen(1)

        logging.info(f"Waiting for connection from C# client on {CelesteEnv.TCP_IP}:{CelesteEnv.TCP_PORT}...")
        self._conn, self._addr = self._server_sock.accept()
        # self._conn.settimeout(0.1)  # 0.1 second timeout for all recv operations
        logging.info(f"Connected to {self._addr}")

        self._json_data = None
        self._celeste_inputs = CelesteInputs()

        self._steps = 0
        
        # Rendering setup
        self._window_name = "Celeste Environment"
        self._render_window_created = False

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8)

        self.action_space = gym.spaces.MultiBinary(7)  # up, down, left, right, jump, dash, grab

    def close(self):
        logging.debug("Closing environment")
        if self.render_mode == "human" and self._render_window_created:
            cv2.destroyWindow(self._window_name)
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
        dummy = self._recv_json()
        if dummy is None:
            logging.error("Failed to receive dummy message from C# client during reset")
            return None, None
        reset_msg = json.dumps({"type": "reset"}).encode('utf-8')
        self._conn.sendall(reset_msg)

        observation = self._get_obs()
        info = self._get_info()

        self._starting_distance = info["distance"] if info and info["distance"] is not None else 500.0
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

        # Request the game statesasasa
        observation = self._get_obs()
        info = self._get_info()
        
        # Render the environment if render mode is human
        if self.render_mode == "human":
            self.render(observation, info)

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
            self._json_data = None
            while self._json_data is None:
                self._json_data = self._recv_json()
                if self._json_data is None:
                    logging.error("Failed to receive valid JSON")
                    return None
                self._send_ack()

        img_base64 = self._json_data["screenPixelsBase64"] if "screenPixelsBase64" in self._json_data else None
        width = self._json_data["screenWidth"] if "screenWidth" in self._json_data else 320
        height = self._json_data["screenHeight"] if "screenHeight" in self._json_data else 180
        if img_base64 is not None:
            observation = self._parse_image_base64(img_base64, width, height)
        else:
            observation = None

        return observation

    def _recv_json(self):
        try:
            data = b''
            while True:
                chunk = self._conn.recv(self.BUFFER_SIZE)
                if not chunk:
                    break
                data += chunk
                try:
                    return json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logging.error(f"Error receiving JSON: {e}")
            return None

    def _send_ack(self):
        try:
            ack_msg = json.dumps({"type": "ACK"}).encode('utf-8')
            self._conn.sendall(ack_msg)
            return True
        except Exception as e:
            logging.error(f"Error sending ACK: {e}")
            return False

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

    def render(self, observation=None, info=None):
        """Render the environment state in a window"""
        if self.render_mode != "human":
            return
            
        if observation is None:
            observation = self._get_obs()
        if info is None:
            info = self._get_info()
            
        if observation is None:
            return
            
        # Create display image
        display_img = observation.copy()
        
        # Convert from RGB to BGR for OpenCV
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        
        # Scale up the image for better visibility (2x scaling)
        display_img = cv2.resize(display_img, (640, 360), interpolation=cv2.INTER_NEAREST)
        
        # Add info text overlay if info is available
        if info is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)  # Green text
            thickness = 2
            
            # Add distance info
            distance_text = f"Distance: {info['distance']:.2f}"
            cv2.putText(display_img, distance_text, (10, 30), font, font_scale, color, thickness)
            
            # Add steps info
            steps_text = f"Steps: {info['steps']}"
            cv2.putText(display_img, steps_text, (10, 60), font, font_scale, color, thickness)
            
            # Add player position if available
            if self._json_data is not None:
                pos_text = f"Player: ({self._json_data.get('playerXPosition', 0):.1f}, {self._json_data.get('playerYPosition', 0):.1f})"
                cv2.putText(display_img, pos_text, (10, 90), font, font_scale, color, thickness)
                
                target_text = f"Target: ({self._json_data.get('targetXPosition', 0):.1f}, {self._json_data.get('targetYPosition', 0):.1f})"
                cv2.putText(display_img, target_text, (10, 120), font, font_scale, color, thickness)
            
            # Add status indicators
            if info.get('playerDied', False):
                cv2.putText(display_img, "DIED", (10, 150), font, font_scale, (0, 0, 255), thickness)  # Red
            if info.get('playerReachedNextRoom', False):
                cv2.putText(display_img, "NEXT ROOM!", (10, 180), font, font_scale, (255, 255, 0), thickness)  # Cyan
        
        # Show the image
        cv2.imshow(self._window_name, display_img)
        cv2.waitKey(1)  # Non-blocking wait
        
        if not self._render_window_created:
            self._render_window_created = True