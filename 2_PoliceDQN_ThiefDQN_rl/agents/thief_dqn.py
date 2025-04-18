import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from utils.config import *

class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

class ThiefDQNAgent:
    def __init__(self, env):
        self.env = env
        self.action_size = env.action_space.n
        self.state_size = 4  # [my_x, my_y, target_x, target_y]
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        
        self.model = DQNModel(self.state_size, self.action_size)
        self.target_model = DQNModel(self.state_size, self.action_size)
        self.update_target_model()
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _preprocess_state(self, state):
        thief = state["thief_obs"]
        return np.array([
            thief["my_position"][0], thief["my_position"][1],
            thief["target_position"][0], thief["target_position"][1]
        ], dtype=np.float32)

    def act(self, state):
        state_array = self._preprocess_state(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        s = self._preprocess_state(state)
        s2 = self._preprocess_state(next_state)
        self.memory.append((s, action, reward, s2, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for s, a, r, s2, done in minibatch:
            state_tensor = torch.FloatTensor(s)
            next_state_tensor = torch.FloatTensor(s2)
            target = self.model(state_tensor).detach().numpy()
            if done:
                target[a] = r
            else:
                next_q = self.target_model(next_state_tensor).detach().numpy()
                target[a] = r + self.gamma * np.max(next_q)
            states.append(s)
            targets.append(target)

        states = torch.FloatTensor(states)
        targets = torch.FloatTensor(targets)

        self.optimizer.zero_grad()
        output = self.model(states)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
