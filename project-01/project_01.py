import numpy as np
import matplotlib.pyplot as plt
import gym
import utilities
from gym import spaces
from utilities import LOGLEVEL

class MyEnvironment(gym.Env):

    '''
    Initializes the class
    Define action and observation space
    '''
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 4, (4,4))
        self.action_space = gym.spaces.Discrete(4)
        logger.log("Observation Space: " + str(self.observation_space), console=True)
        logger.log("Observation Space: " + str(self.action_space), console=True)
    
    '''
    Executes one timestep within the environment
    Input to the function is an action
    Returns observation, reward, done, info
    '''
    def step(self, action):
        pass

    '''
    Resets the state of the environment to an initial state
    Returns staring state
    '''
    def reset(self):
        pass

    '''
    Visualizes the environment
    Any form like vector representation or visualizing using matplotlib will be sufficient
    '''
    def render(self):
        pass

if __name__ == "__main__":
    global logger
    logger = utilities.Logger(LOGLEVEL.DEBUG, "project-01/log.txt")
    env = MyEnvironment()