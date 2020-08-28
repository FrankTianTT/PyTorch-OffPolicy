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

DEVICE = 'cuda'
class Prioritized_Replay_Buffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.reply_buffer = collections.deque(maxlen=self.buffer_size)
        self.priority = collections.deque(maxlen=self.buffer_size)
        self.buffer_len = 0
        self.indexs = None

    def __len__(self):
        return len(self.reply_buffer)

    def add(self, obs, action, reward, next_obs, done):
        self.reply_buffer.append([obs, action, reward, next_obs, done])
        if len(self.priority) == 0:
            self.priority.append(1)
        else:
            self.priority.append(max(self.priority))
        self.buffer_len = len(self.reply_buffer)
        assert len(self.reply_buffer) == len(self.priority)

    def update_priority(self, td_error):
        for i in range(self.batch_size):
            self.priority[self.indexs[i]] = abs(float(td_error[i]))

    def get_weights(self):
        beta = 0.6
        batch_priority = []
        for i in range(self.batch_size):
            batch_priority.append(self.priority[self.indexs[i]])
        batch_priority = (np.array(batch_priority)/sum(batch_priority))
        weights = np.power(batch_priority * self.batch_size, -beta)
        weights = weights / max(weights)
        return torch.tensor(weights)

    def sample(self):
        assert self.buffer_len >= self.batch_size
        probability = np.array(self.priority)/sum(self.priority)
        self.indexs = np.random.choice(self.buffer_len, self.batch_size, replace=False, p=probability)
        obss, actions, rewards, next_obss, dones = zip(*[self.reply_buffer[i] for i in self.indexs])
        weights = self.get_weights()
        return np.array(obss, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_obss, dtype=np.float32), np.array(dones, dtype=np.bool), weights

class Prioritized_Replay_DQN_Agent(DQN_Agent):
    def __init__(self,
                 env,
                 net,
                 Buffer=Prioritized_Replay_Buffer,
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
                 model_name='Prioritized-Replay-DQN'):
        super(Prioritized_Replay_DQN_Agent, self).__init__(env,
                                                           net,
                                                           Buffer=Buffer,
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
        obss, actions, rewards, next_obss, dones, weights = self.buffer.sample()
        obss = torch.tensor(obss).to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_obss = torch.tensor(next_obss).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        now_obs_action_values = self.q_net(obss).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
        next_obs_values = self.target_net(next_obss).max(1)[0]
        next_obs_values[dones] = 0.0
        next_obs_values = next_obs_values.detach()

        expected_obs_action_values = rewards + self.get_gamma() * next_obs_values
        self.buffer.update_priority((now_obs_action_values - expected_obs_action_values).detach())
        return (weights.to(self.device)*(now_obs_action_values - expected_obs_action_values)**2).mean()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = Q_Network(env.observation_space.shape[0], env.action_space.n, 100)
    agent = Prioritized_Replay_DQN_Agent(env, net)
    agent.train_with_traje_reward(430)
    agent.play()
