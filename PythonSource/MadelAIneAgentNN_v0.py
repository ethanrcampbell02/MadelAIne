import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Linear input: playerPosition(2), playerVelocity(2), playerCanDash(1), playerStamina(1), targetPosition(2)
        self.linear_input_dim = 2 + 2 + 1 + 1 + 2  # =8

        # Linear path
        self.linear_fc = nn.Sequential(
            nn.Linear(self.linear_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Conv path for roomTileData (shape: 23x40)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # (8, 23, 40)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # (16, 12, 20)
            nn.ReLU(),
            nn.Flatten()
        )
        # Calculate conv output size
        dummy_input = torch.zeros(1, 1, 23, 40)
        conv_out_size = self.conv(dummy_input).shape[1]

        # Combine
        self.combined_fc = nn.Sequential(
            nn.Linear(64 + conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 7 actions (MultiBinary(7))
        )

    def forward(self, obs):
        # obs: dict with keys as described
        # Linear input
        linear_features = [
            obs["playerPosition"],
            obs["playerVelocity"],
            obs["playerCanDash"].float(),
            obs["playerStamina"],
            obs["targetPosition"]
        ]
        linear_input = torch.cat(linear_features, dim=-1)
        linear_out = self.linear_fc(linear_input)

        # Conv input
        room_tile = obs["roomTileData"].float().unsqueeze(1)  # (batch, 1, 23, 40)
        conv_out = self.conv(room_tile)

        # Combine
        combined = torch.cat([linear_out, conv_out], dim=-1)
        output = self.combined_fc(combined)
        return output

network = DQN()

# Print the number of parameters in the network
print("Number of parameters:", sum(p.numel() for p in network.parameters()))
