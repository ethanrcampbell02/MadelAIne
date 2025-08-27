import numpy as np
import torch
import torch.nn as nn

class MadelAIneNN(nn.Module):
    def __init__(self, input_dims, num_actions, freeze=False):
        super(MadelAIneNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        if freeze:
            self._freeze()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.fc(x)
    
    def _freeze(self):
        for p in self.fc.parameters():
            p.requires_grad = False
