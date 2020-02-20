import numpy as np
import matplotlib.pyplot as plt
import gym

class MyEnvironment(gym.Env):

    '''
    Initializes the class
    Define action and observation space
    '''
    def __init__(self):
        pass
    
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