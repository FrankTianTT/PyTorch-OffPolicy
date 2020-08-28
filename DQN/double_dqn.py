from DQN.dqn import DQN_Agent
from DQN.dqn import Q_Network
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

DEVICE = 'cpu'

class Double_DQN_Agent(DQN_Agent):
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
                 model_name='Double-DQN'):
        super(Double_DQN_Agent, self).__init__(env,
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

    def get_batch_loss(self):
        obss, actions, rewards, next_obss, dones = self.buffer.sample()
        obss = torch.tensor(obss).to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_obss = torch.tensor(next_obss).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        now_obs_action_values = self.q_net(obss).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
        max_action_indexes = torch.argmax(self.q_net(next_obss), dim=1).unsqueeze(-1)
        next_obs_values = self.target_net(next_obss).gather(dim=1, index=max_action_indexes).squeeze(-1)
        next_obs_values[dones] = 0.0
        next_obs_values = next_obs_values.detach()

        expected_obs_action_values = rewards + self.get_gamma() * next_obs_values
        return nn.MSELoss()(now_obs_action_values, expected_obs_action_values)
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = Q_Network(env.observation_space.shape[0], env.action_space.n, 100)
    agent = Double_DQN_Agent(env, net)
    agent.train_with_traje_reward(430)
    agent.play()