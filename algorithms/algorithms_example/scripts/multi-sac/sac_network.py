import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Dueling_DQN(nn.Module):
    def __init__(self, int_dims, hidden, actions):
        super(Dueling_DQN,self).__init__()
        self.fc1 = nn.Linear(int_dims, hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.Value = nn.Linear(hidden, actions)
        self.Value.weight.data.normal_(0,0.1) # initialization
        self.Advantage = nn.Linear(hidden, actions)
        self.Advantage.weight.data.normal_(0,0.1) # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.Value(x)
        advantage = self.Advantage(x)
        return value + advantage - advantage.mean(1, keepdim=True)  # 此处的V是指状态值函数， A是指每个动作值函数的优势函数
        return value


class Actor_discrete(nn.Module):
    def __init__(self, state, hidden, action):
        super(Actor_discrete,self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(hidden,action)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        action_probability = F.softmax(actions_value)
        return action_probability