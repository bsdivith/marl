from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import time

class PPONetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.policy(x), self.value(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.network = PPONetwork(state_dim, 256, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        self.clip_epsilon = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs, _ = self.network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        for _ in range(10):  # PPO epochs
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train():
    # Initialize Unity environment
    env = UnityEnvironment(file_name="SoccerTwos", seed=42)
    env.reset()
    
    # Get behavior names and specs
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    
    # Initialize agents (one for each team)
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.discrete_branches[0]
    team1_agent = PPOAgent(state_dim, action_dim)
    team2_agent = PPOAgent(state_dim, action_dim)
    
    # Training parameters
    episodes = 1000
    max_steps = 1000
    
    for episode in range(episodes):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        # Storage for experience
        states, actions, rewards = [], [], []
        log_probs = []
        
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            # Get current states
            current_states = decision_steps.obs[0]
            
            # Get actions for both teams
            team1_actions = []
            team1_log_probs = []
            team2_actions = []
            team2_log_probs = []
            
            for i in range(len(current_states)):
                if i % 2 == 0:  # Team 1
                    action, log_prob = team1_agent.get_action(current_states[i])
                    team1_actions.append(action)
                    team1_log_probs.append(log_prob)
                else:  # Team 2
                    action, log_prob = team2_agent.get_action(current_states[i])
                    team2_actions.append(action)
                    team2_log_probs.append(log_prob)
            
            # Combine actions and send to environment
            all_actions = team1_actions + team2_actions
            action_tuple = ActionTuple(discrete=np.array([all_actions]))
            
            # Take step in environment
            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # Store experience
            states.extend(current_states)
            actions.extend(all_actions)
            log_probs.extend(team1_log_probs + team2_log_probs)
            rewards.extend(decision_steps.reward)
            
            episode_reward += sum(decision_steps.reward)
            step += 1
            
            if len(terminal_steps) > 0:
                break
        
        # Update agents using collected experience
        # (Simplified version - you might want to add proper advantage calculation)
        returns = np.array(rewards)
        advantages = returns - returns.mean()
        
        team1_agent.update(states[::2], actions[::2], log_probs[::2], 
                          returns[::2], advantages[::2])
        team2_agent.update(states[1::2], actions[1::2], log_probs[1::2],
                          returns[1::2], advantages[1::2])
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {episode_reward/step:.2f}")
    
    env.close()

if __name__ == "__main__":
    train()