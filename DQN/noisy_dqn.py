from DQN.dqn import DQN_Agent
import gym
import random
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from tensorboardX import SummaryWriter

DEVICE = 'cpu'

class Noisy_Linear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(Noisy_Linear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init), requires_grad=True)
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init), requires_grad=True)
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data
        return F.linear(x, weight, bias)


class Noisy_Q_Network(nn.Module):
    def __init__(self, obs_size, actor_size, hidden_size):
        super(Noisy_Q_Network, self).__init__()
        self.layer = nn.Sequential(Noisy_Linear(obs_size, hidden_size),
                                   nn.ReLU(),
                                   Noisy_Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   Noisy_Linear(hidden_size, actor_size))

    def forward(self, x):
        return self.layer(x)


class Noisy_DQN_Agent(DQN_Agent):
    def __init__(self,
                 env,
                 net,
                 env_name='CartPole-v1',
                 mode='train',
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
                                              net=net,
                                              env_name=env_name,
                                              mode=mode,
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
    net = Noisy_Q_Network(env.observation_space.shape[0], env.action_space.n, 100)
    agent = Noisy_DQN_Agent(env, net)
    agent.train_with_traje_reward(430)
    agent.play()