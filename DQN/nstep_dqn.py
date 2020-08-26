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

class N_Step_DQN_Agent(DQN_Agent):
    def __init__(self,
                 env,
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
                 nstep=5,
                 model_name='N-Step-DQN'):
        super(N_Step_DQN_Agent, self).__init__(env,
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
        self.nstep = nstep
        self.nstep_buffer = collections.deque(maxlen=self.nstep)

    def step(self, action, render=False):
        obs, reward, done, _ = self.env.step(action)
        if self.mode == 'train':
            self.nstep_buffer.append((self.now_obs, action, reward, obs, done))
            if len(self.nstep_buffer) == self.nstep:
                if not done:
                    acc_reward = 0
                    for i in range(self.nstep):
                        acc_reward += self.gamma * acc_reward + self.nstep_buffer[self.nstep - i - 1][2]
                    self.buffer.add(self.now_obs, action, acc_reward, obs, done)
                else:
                    acc_reward = 0
                    for i in range(self.nstep):
                        acc_reward += self.gamma * acc_reward + self.nstep_buffer[self.nstep - i - 1][2]
                        self.buffer.add(self.nstep_buffer[self.nstep - i - 1][0],
                                        self.nstep_buffer[self.nstep - i - 1][1],
                                        acc_reward,
                                        obs,
                                        done)
                    self.nstep_buffer.clear()
        self.now_obs = obs
        self.this_trajectory_reward += reward
        self.total_step += 1
        if render:
            self.env.render()
        return obs, reward, done, _

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = N_Step_DQN_Agent(env)
    agent.train_with_traje_reward(430)
    agent.play()