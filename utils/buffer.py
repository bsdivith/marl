# utils/buffer.py

import numpy as np
import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.terminals = []

    def add(self, state, action, reward, log_prob, value, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.terminals.append(terminal)

    def get_batch(self, device=None):
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        terminals = torch.FloatTensor(np.array(self.terminals)).to(device)
        
        return states, actions, rewards, log_probs, values, terminals

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.terminals.clear()