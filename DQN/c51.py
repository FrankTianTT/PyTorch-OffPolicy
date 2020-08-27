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

class Comb_Q_Network(nn.Module):
    def __init__(self, obs_size, actor_size, hidden_size, atoms=51):
        super(Comb_Q_Network, self).__init__()
        self.actor_size = actor_size
        self.atoms = atoms
        self.min_atom_value = None
        self.max_atom_value = None
        self.layer = nn.Sequential(nn.Linear(obs_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, actor_size * atoms))

    def forward(self, x):
        z = self.layer(x).view(self.actor_size, self.atoms)
        normal_z = F.softmax(z, -1)
        return normal_z

    def set_atom_value(self, min_atom_value, max_atom_value):
        assert min_atom_value < max_atom_value
        self.min_atom_value = min_atom_value
        self.max_atom_value = max_atom_value

    def expected_value(self, x, device):
        assert self.min_atom_value is not None and self.max_atom_value is not None
        normal_z = self.forward(x)
        atom_spacing = (self.max_atom_value - self.min_atom_value)/(self.atoms - 1)
        atoms_value = torch.arange(self.min_atom_value, self.max_atom_value + atom_spacing, atom_spacing)
        return (normal_z * atoms_value.to(device)).sum(dim=-1)

class C51_DQN_Agent(DQN_Agent):
    def __init__(self,
                 env,
                 Net=Comb_Q_Network,
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
                 model_name='C51_DQN'):
        super(C51_DQN_Agent, self).__init__(env,
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
        self.q_net.set_atom_value(0, 500)
        self.target_net.set_atom_value(0, 500)

    def best_action(self, obs):
        q = self.q_net.expected_value(torch.tensor(obs).float().to(self.device), self.device)
        return int(torch.argmax(q))

    def get_batch_loss(self):
        obss, actions, rewards, next_obss, dones = self.buffer.sample()
        obss = torch.tensor(obss).to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_obss = torch.tensor(next_obss).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        now_obs_action_values = self.q_net(obss).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
        next_obs_values = self.target_net(next_obss).max(1)[0]
        next_obs_values[dones] = 0.0
        next_obs_values = next_obs_values.detach()

        expected_obs_action_values = rewards + self.gamma * next_obs_values
        return nn.MSELoss()(now_obs_action_values, expected_obs_action_values)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = C51_DQN_Agent(env)
    agent.full_buffer()