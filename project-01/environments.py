import gym
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


class DeterministicEnvironment(gym.Env):
    '''
    Initializes the class
    Define action and observation space
    '''
    def __init__(self, size=4):
        self.size = size
        # Create a 4x4 grid envrionment
        self.observation_space = gym.spaces.Box(0, size, (size, size))
        # 4 Possible actions: up (0), right (1), down (2), and left(3)
        self.action_space = gym.spaces.Discrete(4)
        self.max_timestep = (self.size*2) + 1

    '''
    Executes one timestep within the environment
    Input to the function is an action
    Returns observation, reward, done, info
    0=up, 1=right, 2=down, 3=left
    '''
    def step(self, action):
        # up (0), right (1), down (2), and left(3)
        old_dist = math.sqrt((self.agent_pos[0]-self.goal_pos[0])**2 + (self.agent_pos[1]-self.goal_pos[1])**2)

        if action == 0 and (self.agent_pos[0]-1 >= 0):
            self.agent_pos[0] -= 1
        if action == 1 and (self.agent_pos[1]+1 < self.size):
            self.agent_pos[1] += 1
        if action == 2 and (self.agent_pos[0]+1 < self.size):
            self.agent_pos[0] += 1
        if action == 3 and (self.agent_pos[1]-1 >= 0):
            self.agent_pos[1] -= 1

        new_dist = math.sqrt(
            (self.agent_pos[0]-self.goal_pos[0])**2 + (self.agent_pos[1]-self.goal_pos[1])**2)

        # Reward of +1 for approaching goal, -1 for moving farther away, 0 for no change
        if new_dist < old_dist:
            reward = 1
        elif new_dist > old_dist:
            reward = -1
        else:
            reward = 0

        # Stop if the agent has reached the goal or run out of time
        if new_dist == 0 or self.timestep >= self.max_timestep:
            done = True
        else:
            done = False

        self.timestep += 1

        observation = self.agent_pos
        info = {}
        return observation, reward, done, info

    '''
    Resets the state of the environment to an initial state
    Returns staring state
    '''
    def reset(self, start_pos=[0, 0], goal_pos=None):
        if goal_pos is None:
            goal_pos = [self.size-1, self.size-1]
        self.timestep = 0
        self.agent_pos = copy.copy(start_pos)  # agent_pos[0] = ypos, agent_pos[1] = xpos
        self.goal_pos = goal_pos
        observation = self.agent_pos
        return self.agent_pos

    '''
    Visualizes the environment
    Any form like vector representation or visualizing using matplotlib will be sufficient
    '''
    def render(self):
        state = np.zeros((self.size,self.size))
        state[tuple(self.agent_pos)] = 1
        state[tuple(self.goal_pos)] = 0.5
        plt.imshow(state)
