from DQN.dqn import DQN_Agent
import gym
import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from tensorboardX import SummaryWriter

DEVICE = 'cuda'

class Noisy_Q_Network(nn.Module):
    def __init__(self, obs_size, actor_size, hidden_size):
        super(Noisy_Q_Network, self).__init__()
        self.layer1 = nn.Linear(obs_size, hidden_size)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        self.adv_flow = nn.Linear(hidden_size, actor_size)
        nn.init.xavier_normal_(self.adv_flow.weight, gain=1)
        self.v_flow = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.v_flow.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        adv = self.adv_flow(x)
        v = self.v_flow(x)
        return v + adv - adv.mean()

class Noisy_DQN_Agent(DQN_Agent):
    def __init__(self,
                 env,
                 Net=Noisy_Q_Network,
                 env_name='CartPole-v1',
                 mode='train',
                 hidden_size=64,
                 buffer_size=2096,
                 batch_size=32,
                 gamma=0.9,
                 max_epsilon=0.3,
                 anneal_explore=True,
                 learning_rate=0.001,
                 device=DEVICE,
                 synchronize=200,
                 model_name='Noisy_DQN'):
        super(Noisy_DQN_Agent, self).__init__(env,
                                              Net=Net,
                                              env_name=env_name,
                                              mode=mode,
                                              hidden_size=hidden_size,
                                              buffer_size=buffer_size,
                                              batch_size=batch_size,
                                              gamma=gamma,
                                              max_epsilon=max_epsilon,
                                              anneal_explore=anneal_explore,
                                              learning_rate=learning_rate,
                                              device=device,
                                              synchronize=synchronize,
                                              model_name=model_name)
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Noisy_DQN_Agent(env)
    agent.train_with_traje_reward(430)
    agent.play()