import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicePPOAgent:
    def __init__(self, env, learning_rate=0.0003, gamma=0.99, 
                 clip_epsilon=0.2, entropy_coef=0.01):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
        # Policy and value networks
        self.policy = self._build_policy_network()
        self.value = self._build_value_network()
        
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), 
                                  lr=learning_rate)
        
        self.memory = []
        
    def _build_policy_network(self):
        """Build policy network for police"""
        return nn.Sequential(
            nn.Linear(self._get_state_size(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.env.action_space.n),
            nn.Softmax(dim=-1)
        )
        
    def _build_value_network(self):
        """Build value network for police"""
        return nn.Sequential(
            nn.Linear(self._get_state_size(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def _get_state_size(self):
        """Police sees: [my_x, my_y, thief_x, thief_y]"""
        return 4  # 2 positions (x,y) for both agents
        
    def _preprocess_state(self, state):
        """Normalize positions to [0,1] range"""
        my_pos = state["police_obs"]["my_position"] / max(self.env.grid_size)
        target_pos = state["police_obs"]["target_position"] / max(self.env.grid_size)
        return np.concatenate([my_pos, target_pos]).astype(np.float32)
        
    def act(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(self._preprocess_state(state))
        action_probs = self.policy(state_tensor)
        
        # Add small constant to avoid NaN
        action_probs = action_probs + 1e-8
        action_probs = action_probs / torch.sum(action_probs)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store for training
        self.memory.append(dist.log_prob(action))
        
        return action.item()
        
    def remember(self, reward, value, done):
        """Store experience for training"""
        self.memory[-1] = (self.memory[-1], reward, value, done)
        
    def train(self):
        """PPO training step"""
        if len(self.memory) < 1:
            return 0.0
            
        # Unpack memory
        log_probs, rewards, values, dones = zip(*self.memory)
        
        # Convert to tensors
        log_probs = torch.stack(log_probs)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)
        
        # Calculate returns
        returns = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + (1 - dones[i]) * self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Calculate advantages
        advantages = returns - values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO loss
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = 0.5 * (returns - values).pow(2).mean()
        
        # Entropy bonus
        entropy = -torch.sum(torch.exp(log_probs) * log_probs).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear memory
        self.memory = []
        
        return loss.item()