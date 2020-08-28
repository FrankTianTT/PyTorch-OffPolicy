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

DEVICE = 'cpu'

class Comb_Q_Network(nn.Module):
    def __init__(self, obs_size, actor_size, hidden_size, min_atom_value, max_atom_value, atoms=51):
        super(Comb_Q_Network, self).__init__()
        assert min_atom_value < max_atom_value
        self.min_atom_value = min_atom_value
        self.max_atom_value = max_atom_value
        self.actor_size = actor_size
        self.atoms = atoms
        self.layer = nn.Sequential(nn.Linear(obs_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, actor_size * atoms))

    def forward(self, x):
        z = self.layer(x).view(-1, self.actor_size, self.atoms)
        normal_z = F.softmax(z, -1)
        return normal_z

    def expected_value(self, x, device):
        assert self.min_atom_value is not None and self.max_atom_value is not None
        normal_z = self.forward(x)
        atom_spacing = (self.max_atom_value - self.min_atom_value)/(self.atoms - 1)
        atoms_value = torch.arange(self.min_atom_value, self.max_atom_value + atom_spacing, atom_spacing)
        return (normal_z * atoms_value.to(device)).sum(dim=-1)

class C51_DQN_Agent(DQN_Agent):
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
                 model_name='C51_DQN'):
        super(C51_DQN_Agent, self).__init__(env,
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
        assert self.q_net.min_atom_value is not None and\
               self.q_net.max_atom_value is not None and\
               self.q_net.atoms is not None
        self.min_atom_value = self.q_net.min_atom_value
        self.max_atom_value = self.q_net.max_atom_value
        self.atoms = self.q_net.atoms

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

        now_distr = self.q_net(obss)
        now_obs_action_distribs = now_distr[torch.arange(now_distr.size(0)), actions]

        next_distribs = self.q_net(next_obss)
        best_action = torch.argmax(self.target_net.expected_value(next_obss, self.device), dim=-1)
        next_obs_distribs = next_distribs[torch.arange(next_distribs.size(0)), best_action]
        expected_obs_action_distribs = self.normalize_comb(next_obs_distribs, rewards, dones)
        expected_obs_action_distribs = expected_obs_action_distribs.detach()

        loss = self.cal_cross_entropy(now_obs_action_distribs, expected_obs_action_distribs)
        return loss

    def normalize_comb(self, next_obs_distribs, rewards, dones):
        atom_spacing = (self.max_atom_value - self.min_atom_value) / (self.atoms - 1)
        # atoms_value is the support {z} in the article
        atoms_value = torch.arange(self.min_atom_value, self.max_atom_value + atom_spacing, atom_spacing)

        rewards = rewards.view(self.batch_size, 1).repeat(1, self.atoms)
        atoms_value_add_reward = rewards + self.get_gamma() * atoms_value

        max_atoms_value = torch.tensor(self.max_atom_value).repeat(atoms_value_add_reward.size()).float()
        min_atoms_value = torch.tensor(self.min_atom_value).repeat(atoms_value_add_reward.size()).float()
        atoms_value_add_reward = torch.where(atoms_value_add_reward > self.max_atom_value, max_atoms_value, atoms_value_add_reward)
        atoms_value_add_reward = torch.where(atoms_value_add_reward < self.min_atom_value, min_atoms_value, atoms_value_add_reward)

        b = (atoms_value_add_reward - self.min_atom_value)/atom_spacing
        l = b.floor()
        u = b.ceil()

        expected_obs_action_distrs = torch.zeros(next_obs_distribs.size())
        for batch in range(self.batch_size):
            for atom in range(self.atoms):
                l_index = l[batch][atom].int()
                u_index = u[batch][atom].int()
                expected_obs_action_distrs[batch][l_index] = \
                    expected_obs_action_distrs[batch][l_index] + next_obs_distribs[batch][atom] * (
                            u[batch][atom] - b[batch][atom])
                expected_obs_action_distrs[batch][u_index] = \
                    expected_obs_action_distrs[batch][u_index] + next_obs_distribs[batch][atom] * (
                            b[batch][atom] - l[batch][atom])
        expected_obs_action_distrs = F.softmax(expected_obs_action_distrs, -1)
        return expected_obs_action_distrs
    def cal_cross_entropy(self, now_obs_action_distrs, expected_obs_action_distrs):
        log_softmax = torch.log(now_obs_action_distrs)
        log_softmax = log_softmax.view(-1)
        expected_obs_action_distrs = expected_obs_action_distrs.view(-1)
        return - torch.dot(log_softmax, expected_obs_action_distrs).sum()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = Comb_Q_Network(obs_size=env.observation_space.shape[0],
                         actor_size=env.action_space.n,
                         hidden_size=100,
                         min_atom_value=0,
                         max_atom_value=500,
                         atoms=51)
    agent = C51_DQN_Agent(env, net)
    agent.train_with_traje_reward(450)
    agent.play()