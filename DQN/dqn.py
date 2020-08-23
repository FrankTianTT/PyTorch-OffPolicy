import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
The input size of Q network is the size of observation, and the output size of Q network
is the size of actor. So it is obs -> Q of action, but not (obs, action) -> Q.
'''
class Q_Network(nn.Module):
    def __init__(self, obs_size, actor_size,hidden_size=30):
        super(Q_Network, self).__init__()
        self.layer1 = nn.Linear(obs_size, hidden_size)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        self.layer2 = nn.Linear(hidden_size, actor_size)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
    def forward(self,x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

class Buffer:
    def __init__(self, buffer_size=1000, batch_size=256):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.reply_buffer = []
        self.buffer_len = 0

    def add(self,last_obs , action, reward, obs, done):
        if len(self.reply_buffer) >= self.buffer_size:
            self.reply_buffer.pop(0)
        self.reply_buffer.append([last_obs, action, reward, obs, done])
        self.buffer_len = len(self.reply_buffer)

    def sample(self):
        assert self.buffer_len >= self.batch_size
        sample = random.sample(self.reply_buffer, self.batch_size)
        return sample

class Agent:
    def __init__(self,
                 env,
                 mode='train',
                 hidden_size=100,
                 buffer_size=1000,
                 batch_size=128,
                 gamma=0.9,
                 learning_rate=0.0001,
                 synchronize=50):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.env = env
        self.obs_size = env.observation_space.shape[0]
        self.actor_size = env.action_space.n
        self.hidden_size = hidden_size
        self.q_net = Q_Network(self.obs_size, self.actor_size, self.hidden_size)
        self.target_net = Q_Network(self.obs_size, self.actor_size, self.hidden_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = Buffer(self.buffer_size, self.batch_size)
        self.mode = mode
        self.gamma = gamma
        self.synchronize = synchronize
        self.total_trajectory = 0
        self.total_step = 0
        self.learning_rate = learning_rate
        self.this_trajectory_reward = 0
        self.action_sum = self.actor_size*[0]

        self.now_obs_tensor = None
        self.reset()
        print(self.q_net)

    def reset(self):
        obs = self.env.reset()
        self.now_obs_tensor = torch.tensor(obs).float()
        return torch.tensor(obs).float()

    def step(self, action, render=False):
        obs, reward, done, _ = self.env.step(action)
        obs = torch.tensor(obs).float()
        self.buffer.add(self.now_obs_tensor, action, reward, obs, done)
        self.now_obs_tensor = obs
        self.this_trajectory_reward += reward

        if(render):
            env.render()

        self.total_step += 1
        if done and self.mode=='train':
            self.reset()
            self.total_trajectory += 1
            if self.total_trajectory%10 == 0:
                print("Finish {} trajectories, get {} rewards this one.".format(self.total_trajectory,
                                                                                self.this_trajectory_reward ))
            self.this_trajectory_reward = 0

        return obs, reward, done, _

    def set_mode(self, mode):
        self.mode = mode

    def full_buffer(self):
        while(self.buffer.buffer_len < self.batch_size):
            action = self.select_action(self.now_obs_tensor)
            obs, reward, done, _ = self.step(action)

    def best_action(self, obs):
        q = self.q_net(obs)
        return int(torch.argmax(q))

    def select_action(self, obs, epsilon=0.3):
        assert self.mode == "train"
        if random.random() < epsilon:
            a = random.randint(0, self.actor_size - 1)
            self.action_sum[a] += 1
            return a
        else:
            a = self.best_action(obs)
            self.action_sum[a] += 1
            return a

    def train(self, iters=10000):
        assert self.mode == 'train'
        self.full_buffer()
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()
        t = 0
        reward_list = []
        while t < iters:
            action = self.select_action(self.now_obs_tensor)
            obs, reward, done, _ = self.step(action)

            x, y = self.get_batch_data()
            prediction = self.q_net(x)
            loss = loss_function(y, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if len(reward_list) >= 100:
                reward_list.pop(0)
            reward_list.append(reward)

            if t % self.synchronize == 0:
                self.synchronize_net()
            t += 1

    def synchronize_net(self):
        self.target_net.layer1.weight = self.q_net.layer1.weight
        self.target_net.layer2.weight = self.q_net.layer2.weight

    def play(self):
        self.reset()
        for i in range(1000):
            action = self.best_action(self.now_obs_tensor)
            self.step(action, render=True, reset=False)

    def get_batch_data(self):
        sample = self.buffer.sample()

        x = [s[0].numpy() for s in sample]
        y = [self.get_expect_out(s).detach().numpy() for s in sample]
        return torch.tensor(x), torch.tensor(y)

    #s: last_obs, action, reward, obs, done
    def get_expect_out(self, s):
        q = self.q_net(s[0])
        if not s[4]:
            q[s[1]] = s[2] + torch.max(self.target_net(s[3]))
        else:
            q[s[1]] = s[2]
        return q

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train(100000)
    agent.play()