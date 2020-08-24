from DQN.dqn import DQN_Agent
import gym

DEVICE = 'cpu'
env = gym.make('CartPole-v1')
agent = DQN_Agent(env,
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
                  synchronize=200)
agent.train_with_traje_reward(450)
agent.play()