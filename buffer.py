import numpy as np

class MemoryBuffer:
    def __init__(self, input_shape, n_actions, maxmem, batch_size):
        self.maxmem = maxmem 
        self.batch_size = batch_size 
        self.mem_cntr = 0 
        self.observations = np.zeros((maxmem, *input_shape))
        self.actions = np.zeros((maxmem))
        self.rewards = np.zeros((maxmem))
        self.observations_ = np.zeros((maxmem, *input_shape))
        self.dones = np.zeros((maxmem))

    def save_memory(self, observations, actions, rewards, observations_, done):
        index = self.mem_cntr % self.maxmem
        self.observations[index] = observations 
        self.actions[index] = actions 
        self.rewards[index] = rewards 
        self.observations_[index] = observations_ 
        self.dones[index] = done
        self.mem_cntr += 1 
    
    def sample_memory(self):
        min_sample = min(self.mem_cntr, self.maxmem)

        samples = np.random.choice(min_sample, self.batch_size)

        observations = self.observations[samples]
        actions = self.actions[samples]
        rewards = self.rewards[samples]
        observations_ = self.observations_[samples]
        dones = self.dones[samples]
        return observations, actions, rewards, observations_, dones
        