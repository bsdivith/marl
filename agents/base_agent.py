# agents/base_agent.py

import torch
import torch.nn as nn

class BaseAgent:
    def __init__(self, state_dim, action_dim, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, state):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError