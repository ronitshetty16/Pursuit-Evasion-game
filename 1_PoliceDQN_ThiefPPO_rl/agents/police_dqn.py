import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class PoliceDQNAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        
        # Neural Network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def _build_model(self):
        """Build DQN model for police"""
        return nn.Sequential(
            nn.Linear(self._get_state_size(), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.env.action_space.n)
        )
        
    def _get_state_size(self):
        """Police sees: [my_x, my_y, thief_x, thief_y]"""
        return 4  # 2 positions (x,y) for both agents
        
    def _preprocess_state(self, state):
        """Normalize positions to [0,1] range"""
        my_pos = state["police_obs"]["my_position"] / max(self.env.grid_size)
        target_pos = state["police_obs"]["target_position"] / max(self.env.grid_size)
        return np.concatenate([my_pos, target_pos]).astype(np.float32)
        
    def update_target_model(self):
        """Update target network with current weights"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n-1)
            
        state_tensor = torch.FloatTensor(self._preprocess_state(state))
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
        
    def replay(self, batch_size):
        """Train on a batch from memory"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q.squeeze(), target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()