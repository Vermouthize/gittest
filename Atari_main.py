from utils import make_env
from DQN_network import Agent

env = make_env(env_name='PongNoFrameskip-v4')
agent =Agent(env, lr=0.0001, input_shape=(4, 84, 84), n_actions=env.action_space.n, maxmem=50000, batch_size=32, warmup=100, eps=1.0, eps_min=0.1, eps_frame=100000)

n_games = 500000

for n in range(n_games):
    done = False
    rewards = 0 
    observation = env.reset()
    step = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        rewards += reward 
        observation = observation_
        step += 1
    
    print('episode ', n, 'rewards= ', rewards, 'steps=', step, 'eps= ', agent.eps, 'framecounter= ', agent.frame_counter )
