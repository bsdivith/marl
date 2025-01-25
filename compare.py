import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import time
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3

# Neural Network Architectures
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Base Agent Class
class BaseAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

# PPO Agent
class PPOAgent(BaseAgent):
    def __init__(self, observation_space, action_space, learning_rate=3e-4, gamma=0.99, epsilon=0.2):
        super().__init__(observation_space, action_space)
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n
        
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.value = ValueNetwork(self.input_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = []

    def select_action(self, observation):
        state = torch.FloatTensor(observation)
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob
        })
        
        return action.item()

    def update(self, rewards):
        states = torch.stack([m['state'] for m in self.memory])
        actions = torch.stack([m['action'] for m in self.memory])
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])
        
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        for _ in range(5):
            logits = self.policy(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * returns
            policy_loss = -torch.min(surr1, surr2).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        values = self.value(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        self.memory = []

# MAPPO Agent
class MAPPOAgent(PPOAgent):
    def __init__(self, observation_space, action_space, num_agents, learning_rate=3e-4):
        super().__init__(observation_space, action_space, learning_rate)
        extended_input_dim = self.input_dim * num_agents
        self.policy = PolicyNetwork(extended_input_dim, self.output_dim)
        self.value = ValueNetwork(extended_input_dim)

    def select_action(self, observation, all_observations):
        state = torch.FloatTensor(np.concatenate(all_observations))
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob
        })
        
        return action.item()

# DQN Agent
class DQNAgent(BaseAgent):
    def __init__(self, observation_space, action_space, learning_rate=1e-3, gamma=0.99):
        super().__init__(observation_space, action_space)
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n
        
        self.q_network = PolicyNetwork(self.input_dim, self.output_dim)
        self.target_network = PolicyNetwork(self.input_dim, self.output_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_freq = 100
        self.steps = 0

    def select_action(self, observation):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        state = torch.FloatTensor(observation)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.steps += 1
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Environment Setup
def make_env(num_agents=2, max_cycles=25):
    env = simple_spread_v3.env(
        N=num_agents,
        max_cycles=max_cycles,
        continuous_actions=False
    )
    env.reset()
    return env

# Training Function
def train_and_compare(num_episodes=1000, num_agents=2):
    env = make_env(num_agents)
    results = {
        'PPO': [],
        'MAPPO': [],
        'DQN': []
    }
    training_times = {}
    
    for algo_name in results.keys():
        print(f"\nTraining {algo_name}")
        start_time = time.time()
        
        if algo_name == 'PPO':
            agents = {agent: PPOAgent(env.observation_space(agent), env.action_space(agent))
                     for agent in env.agents}
        elif algo_name == 'MAPPO':
            agents = {agent: MAPPOAgent(env.observation_space(agent), env.action_space(agent), num_agents)
                     for agent in env.agents}
        else:  # DQN
            agents = {agent: DQNAgent(env.observation_space(agent), env.action_space(agent))
                     for agent in env.agents}
        
        for episode in range(num_episodes):
            observations = env.reset()[0]
            episode_rewards = []
            
            for agent in env.agent_iter():
                observation = observations[agent]
                
                if algo_name == 'MAPPO':
                    all_obs = [observations[a] for a in env.agents]
                    action = agents[agent].select_action(observation, all_obs)
                else:
                    action = agents[agent].select_action(observation)
                
                next_observations, reward, termination, truncation, _ = env.step(action)
                episode_rewards.append(reward)
                
                if algo_name == 'DQN':
                    agents[agent].update(
                        observation,
                        action,
                        reward,
                        next_observations[agent],
                        termination or truncation
                    )
                
                observations = next_observations
                if termination or truncation:
                    break
            
            if algo_name in ['PPO', 'MAPPO']:
                for agent in agents.values():
                    agent.update(episode_rewards)
            
            total_reward = sum(episode_rewards)
            results[algo_name].append(total_reward)
            
            if episode % 100 == 0:
                print(f"{algo_name} Episode {episode}: {total_reward}")
        
        training_times[algo_name] = time.time() - start_time
    
    return results, training_times

# Plotting Function
def plot_results(results, training_times):
    plt.figure(figsize=(12, 6))
    
    # Smoothing window
    window = 50
    
    for algo_name, rewards in results.items():
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_rewards, label=f'{algo_name} (training time: {training_times[algo_name]:.1f}s)')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run training and comparison
    results, training_times = train_and_compare(num_episodes=1000, num_agents=2)
    
    # Plot results
    plot_results(results, training_times)