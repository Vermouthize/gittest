import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim  
from buffer import MemoryBuffer
import numpy as np 

class DQN(nn.Module):
    def __init__(self, lr, n_actions):
        super(DQN, self).__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)

        self.fc1 = nn.Linear(49*64, 512) 
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
    
    def forward(self, observation):
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = T.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x 

class Agent:
    def __init__(self, env, lr, input_shape, n_actions, maxmem, batch_size, warmup, eps, eps_min, eps_frame):
        self.env = env
        self.lr = lr 
        self.input_shape = input_shape
        self.n_actions = n_actions 
        self.maxmem = maxmem 
        self.batch_size = batch_size 
        self.warmup = warmup 
        self.eps = eps 
        self.eps_min = eps_min
        self.eps_frame = eps_frame
        self.decrease_rate = 1e-5
        self.frame_counter = 0


        self.memory = MemoryBuffer(input_shape, n_actions, maxmem, batch_size)
        self.network = DQN(lr, n_actions)
        self.target_network = DQN(lr, n_actions)
        self.update_target()

    def choose_action(self, observation):
        if self.frame_counter < self.warmup:
            self.decrease_eps()
            return self.env.action_space.sample() 
        else:
            if np.random.random() < self.eps:
                if self.frame_counter < self.eps_frame:
                    self.decrease_eps()
                return self.env.action_space.sample() 
            else:
                observation = T.tensor([observation], dtype=T.float)
                action_matrix = self.network.forward(observation)
                action_choosen = T.argmax(action_matrix)
                if self.frame_counter < self.eps_frame:
                    self.decrease_eps()
                return action_choosen.item()
    
    def decrease_eps(self):
        self.eps -= self.decrease_rate

    def learn(self):
        self.frame_counter += 1
        if self.frame_counter < self.warmup:
             return 
        self.network.optimizer.zero_grad()
        observations, actions, rewards, observations_, dones = self.memory.sample_memory()
        observations = T.tensor(observations, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        rewards = T.tensor(rewards, dtype=T.float)
        observations_ = T.tensor(observations_, dtype=T.float)
        dones = T.tensor(dones, dtype=T.float)

        indices = np.arange(self.batch_size)

        critic_value_ = self.target_network.forward(observations_).max(dim=1)[0] #.max will return a tuple with value at 0 and indices at 1
        critic_value_ = critic_value_.view(-1)
        target = rewards + 0.99 * critic_value_ * (1-dones)
        target = target.reshape(self.batch_size, 1)
        critic_value = self.network.forward(observations)
        critic_value = critic_value[indices, actions.detach().numpy()].reshape(self.batch_size, 1) #have to have [indices] to correctly get the critic value for the specific action for each sample in the batch.
  

        loss = F.mse_loss(target, critic_value)
        loss.backward()
        self.network.optimizer.step() 

        if self.frame_counter % 1000 == 0:
            self.update_target()
    
    def remember(self, observation, action, reward, observation_, done):
        self.memory.save_memory(observation, action, reward, observation_, done)

    def update_target(self):
    
        self.target_network.load_state_dict(self.network.state_dict())


        