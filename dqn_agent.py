import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

from symbolic_regression import SymbolicRegressor


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, batch_size=64, memory_size=10000):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = []
        self.memory_size = memory_size
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.output_size)
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def store_transition(self, transition):
        if len(self.memory) < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory.pop(0)
            self.memory.append(transition)

    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.sample_memory()
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


df = pd.read_csv('insurance.csv')
df = pd.get_dummies(df)
features = ['age', 'bmi', 'children', 'sex_female', 'sex_male',
       'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
       'region_southeast', 'region_southwest']

target = 'charges'
X = df[features].values
y = df[target].values

constants = list(map(str, range(1, 11)))
operators = ['+', '-', '*']

env = SymbolicRegressor(operators, constants, features, depth=4)


# define the state size
state_size = env.count_nodes(env.tree)
action_size = len(features + constants + operators)

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Define training parameters
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

# Training loop
for episode in range(num_episodes):
    state = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, epsilon)
        next_state = env.apply_action(state, action)
        reward = env.evaluate_expression(next_state)
        agent.store_transition(Transition(state, action, next_state, reward))
        total_reward += reward
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        agent.update_policy()
        agent.update_target_network()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    if episode % 10 == 0:
        print("Episode: {}, Total Reward: {}, Epsilon: {:.2f}".format(episode, total_reward, epsilon))