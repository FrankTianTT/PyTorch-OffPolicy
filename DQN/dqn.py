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

'''
The input size of Q network is the size of observation, and the output size of Q network
is the size of actor. So it is obs -> Q of action, but not (obs, action) -> Q.
'''
class Q_Network(nn.Module):
    def __init__(self, obs_size, actor_size, hidden_size):
        super(Q_Network, self).__init__()
        self.layer1 = nn.Linear(obs_size, hidden_size)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        self.layer2 = nn.Linear(hidden_size, actor_size)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)


'''
Experience buffer
'''


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.reply_buffer = collections.deque(maxlen=self.buffer_size)
        self.buffer_len = 0

    def __len__(self):
        return len(self.reply_buffer)

    def add(self, obs, action, reward, next_obs, done):
        self.reply_buffer.append([obs, action, reward, next_obs, done])
        self.buffer_len = len(self.reply_buffer)

    def sample(self):
        assert self.buffer_len >= self.batch_size
        indexs = np.random.choice(self.buffer_len, self.batch_size, replace=False)
        obss, actions, rewards, next_obss, dones = zip(*[self.reply_buffer[i] for i in indexs])
        return np.array(obss, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_obss, dtype=np.float32), np.array(dones, dtype=np.bool)


class DQN_Agent:
    def __init__(self,
                 env,
                 Net=Q_Network,
                 Buffer=Replay_Buffer,
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
                 model_name='DQN'):
        """

        :param env:
        :param env_name:
        :param mode:
        :param hidden_size:
        :param buffer_size:
        :param batch_size:
        :param gamma:
        :param max_epsilon:
        :param anneal_explore:
        :param learning_rate:
        :param device:
        :param synchronize:
        :param model_name:
        """
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.env = env
        self.env_name = env_name
        self.mode = mode
        self.hidden_size = hidden_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.anneal_explore = anneal_explore,
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.synchronize = synchronize
        self.model_name = model_name

        self.buffer = Buffer(self.buffer_size, self.batch_size)

        self.obs_size = env.observation_space.shape[0]
        self.actor_size = env.action_space.n
        self.q_net = Net(self.obs_size, self.actor_size, self.hidden_size).to(self.device)
        self.target_net = Net(self.obs_size, self.actor_size, self.hidden_size).to(self.device)

        self.total_trajectory = 0
        self.total_step = 0
        self.this_trajectory_reward = 0
        self.recent_trajectory_rewards = collections.deque(maxlen=30)
        self.now_obs = None
        self.reset()

        self.timer_1k_steps = 0
        self.time = time.time()

        self.save_dir = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'saves')
        self.run_dir = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'runs')
        self.writer = None
        print(self.q_net)

    def set_writer(self):
        timestamp = time.time()
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(timestamp))
        self.log_name = '{}_{}_{}.dat'.format(self.model_name, self.env_name, timestamp)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, self.log_name))

    def reset(self):
        obs = self.env.reset()
        self.now_obs = obs
        self.total_trajectory += 1
        self.this_trajectory_reward = 0
        return self.now_obs

    def step(self, action, render=False):
        obs, reward, done, _ = self.env.step(action)
        if self.mode == 'train':
            self.buffer.add(self.now_obs, action, reward, obs, done)
        self.now_obs = obs
        self.this_trajectory_reward += reward
        self.total_step += 1
        if render:
            self.env.render()
        return obs, reward, done, _

    def set_mode(self, mode):
        self.mode = mode

    def full_buffer(self):
        while self.buffer.buffer_len < self.batch_size:
            action = self.select_action(self.now_obs)
            obs, reward, done, _ = self.step(action)

    def best_action(self, obs):
        q = self.q_net(torch.tensor(obs).float().to(self.device))
        return int(torch.argmax(q))

    def get_epsilon(self):
        if self.anneal_explore:
            return self.max_epsilon - self.total_step * 0.00001 if self.total_step < 25000 else 0.05
        else:
            return self.max_epsilon

    def select_action(self, obs):
        assert self.mode == "train"
        if random.random() < self.get_epsilon():
            return self.env.action_space.sample()
        else:
            return self.best_action(obs)

    def train_with_iters(self, iters=10000):
        assert self.mode == 'train'
        self.set_writer()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()
        self.total_step = 0
        reward_list = collections.deque(maxlen=100)
        while self.total_step < iters:
            obs, reward, done, _ = self.train_step()
            reward_list.append(reward)
            if self.total_step % self.synchronize == 0:
                self.synchronize_net()

    def train_with_traje_reward(self, target_reward):
        assert self.mode == 'train'
        self.set_writer()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()
        self.total_step = 0
        reward_list = collections.deque(maxlen=100)
        self.recent_trajectory_rewards.clear()
        while True:
            obs, reward, done, _ = self.train_step()
            reward_list.append(reward)
            if self.total_step % self.synchronize == 0:
                self.synchronize_net()
            if len(self.recent_trajectory_rewards) > 0:
                if sum(self.recent_trajectory_rewards) / len(self.recent_trajectory_rewards) > target_reward:
                    self.save_model()
                    break

    def train_step(self):
        action = self.select_action(self.now_obs)  # select action and step in env
        obs, reward, done, _ = self.step(action)
        if done:
            if self.total_trajectory % 10 == 0:
                print("Finish {} trajectories, get {} rewards this one.".format(self.total_trajectory,
                                                                                self.this_trajectory_reward))
            self.recent_trajectory_rewards.append(self.this_trajectory_reward)
            self.writer.add_scalar('average reward',
                                   sum(self.recent_trajectory_rewards) / len(self.recent_trajectory_rewards),
                                   global_step=self.total_trajectory)
            self.reset()
        if self.buffer.buffer_len > self.batch_size:  # train
            loss = self.get_batch_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return obs, reward, done, _

    def synchronize_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def play(self, done=False):
        self.set_mode('play')
        self.reset()
        for i in range(5000):
            action = self.best_action(self.now_obs)
            obs, reward, done, _ = self.step(action, render=True)
            if done:
                self.reset()

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

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        timestamp = time.time()
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(timestamp))
        aver_reward = sum(self.recent_trajectory_rewards) / len(self.recent_trajectory_rewards)
        fname = '{}_{}_{:.2}_{}.dat'.format(self.model_name, self.env_name, aver_reward, timestamp)
        fname = os.path.join(self.save_dir, fname)
        torch.save(self.q_net.state_dict(), fname)

    def load_model(self, fname):
        save_dir = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'saves')
        fname = os.path.join(save_dir, fname)
        model = torch.load(fname)
        self.q_net.load_state_dict(model)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQN_Agent(env)
    agent.train_with_traje_reward(430)
    agent.play()
