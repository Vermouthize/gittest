import collections 
import numpy as np 
import gym 
import cv2 

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat #repead = 4 because we skipped four frames
        self.shape = env.observation_space.low.shape #this might be an old version thing, env.observation_space.shape should do.
        self.frame_buffer = np.zeros_like((2, self.shape)) #np.zeros cannot create a [0, 0] array that takes another array as an element
    
    def step(self, action):
        t_reward = 0.0 
        done = False 
        for i in range(self.repeat): #repeat four times because of skipping frame. We take the same action in the repeated frames and only record the last frame as the new observation
            obs, reward, done, info = self.env.step(action)
            t_reward += reward 
            idx = i % 2
            self.frame_buffer[idx] = obs 
            if done:
                break 
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1]) #take max elementally and return a matrix with maximum of two matrix on each index
        #this step is used because of the peculiar mechanics for Atari games
        return max_frame, t_reward, done, info #maxframe is already 4*84*84
    
    def reset(self):
        obs = self.env.reset()  #this reset will return the overwritten reset from stackframes, with shape 4-84-84
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs #this step is useless in my opinion

        return obs
    
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1]) #turn gym's shape (channel last) to pytorch's shape (channel first)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, #bc will divide by 255
                                                shape=self.shape, dtype=np.float32)
    
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA) #interpolation is just a parameter for resizing image
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32) #this changes the observation_space.shape from (1, 84, 84) to (4, 84, 84)
        self.stack = collections.deque(maxlen=repeat) #deque object
    
    def reset(self): #this reset will return its value to the reset functioned called in repeadtandmax 
        self.stack.clear() 
        observation = self.env.reset() 
        for _ in range (self.stack.maxlen):
            self.stack.append(observation)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape) #the observation_space.low.shape is (4, 84, 84)
    
    def observation(self, observation): #this observation function will take input the observation returned by the observation function in preprocessframe
        self.stack.append(observation) #deque object will automatically delete the oldest data if new data is appended when len = maxlen

        return np.array(self.stack).reshape(self.observation_space.low.shape)
    
def make_env(env_name, shape=(84, 84, 1), repeat=4):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    # change to env will be stacked
    return env


    