from DQN.dqn import Agent
import gym

DEVICE = 'cpu'
env = gym.make('CartPole-v1')
agent = Agent(env,
              model_name='dqn',
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

agent.train_with_traje_reward(150)
agent.save_model()
# agent.load_model('dqn_CartPole-v1_2020-08-23-21-37-04.dat')
agent.play()