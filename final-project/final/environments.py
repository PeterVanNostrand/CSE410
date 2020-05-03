import gym
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from utilities import Logger

'''
Compute the Euclidean distance between two, two dimension points
'''

def compute_distance(pa, pb):
    # dist = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
    dist = np.abs(pa[0]-pb[0]) + np.abs(pa[1]-pb[1])
    return dist


class MAEnvironment(gym.Env):
    '''
    Initializes the class
    Defines action and observation spaces
    '''

    def __init__(self, size=4, num_agents=2):
        self.size = size
        # Create a 4x4 grid envrionment
        self.observation_space = gym.spaces.Box(0, size, (size, size))
        # 4 Possible actions: up (0), right (1), down (2), left(3), none(4)
        self.action_space = gym.spaces.Discrete(5)
        self.max_timestep = int((self.size*3))
        self.num_agents = num_agents


    '''
    Compute the new position of an agent at position 'pos' after the provided action
    If the action results in an invalid location (off the board, inside another agent/goal)
    Then the new position is the same as the previous posision
    '''

    def move_agent(self, pos, action, agent_idx):
        # Compute the new position
        new_pos = copy.copy(pos)
        if action == 0:
            new_pos[0] -= 1
        elif action == 1:
            new_pos[1] += 1
        elif action == 2:
            new_pos[0] += 1
        elif action == 3:
            new_pos[1] -= 1
        else:  # action == 4, nomove
            pass

        # Check if the new position is valid
        valid = True
        # check off board in x dir
        if new_pos[0] < 0 or new_pos[0] >= self.size:
            valid = False
        # check off board in y dir
        if new_pos[1] < 0 or new_pos[1] >= self.size:
            valid = False
        # check spot filled by another agent
        for i in range(len(self.agents_pos)):   
            if i==agent_idx:
                continue
            elif new_pos[0] == self.agents_pos[i][0] and new_pos[1] == self.agents_pos[i][1]:
                valid = False
        # check spot filled by 'enemy'
        if new_pos[0] == self.goal_pos[0] and new_pos[1] == self.goal_pos[1]:
            valid = False

        # If the new position is not valid, the agent cannot move
        if not valid:
            new_pos = copy.copy(pos)
        return valid, new_pos



    '''
    Executes one timestep within the environment
    Input to the function is an action
    Returns observation, reward, done, info
    0=up, 1=right, 2=down, 3=left
    '''

    def step(self, actions):
        # up (0), right (1), down (2), left(3), nomove(4)
        old_positions = copy.copy(self.agents_pos)

        # Move the agents according to their actions, earlier agents get priority
        valid_moves = [True] * self.num_agents
        for i in range(self.num_agents):
            valid, self.agents_pos[i] = self.move_agent(self.agents_pos[i], actions[i], i)
            valid_moves[i] = valid

        # Compute reward for each agent
        # Reward of +1 for approaching goal, -1 for moving farther away, 0 for no change
        rewards = [0] * self.num_agents
        dists = [0] * self.num_agents
        for i in range(self.num_agents):
            old_dist = compute_distance(old_positions[i], self.goal_pos)
            new_dist = compute_distance(self.agents_pos[i], self.goal_pos)
            dists[i] = new_dist
            if new_dist < old_dist:
                rewards[i] = -1
            elif new_dist > old_dist or valid_moves[i] is False:
                rewards[i] = -2
            elif new_dist == old_dist:
                if new_dist == 1:
                    # if the agent is 'containing' the enemy give a reward
                    rewards[i] = 0
                else:
                    # otherwise, punish for maintaing the same distance
                    rewards[i] = -2
            # rewards[i] = -1 * dists[i]

        # Increment the time step
        self.timestep += 1

        # Continue until the enemy is surrounded or time has elapsed
        done = True
        for d in dists:
            if d!=1:
                done = False
        if self.timestep >= self.max_timestep:
            done = True

        observation = self.agents_pos
        info = {}
        return observation, rewards, done, info

    '''
    Resets the state of the environment to an initial state
    Returns staring state
    '''

    def reset(self, start_pos, goal_pos):
        # Reset the time step
        self.timestep = 1
        # Set the goal position
        self.goal_pos = goal_pos
        # Record the number of agents
        self.num_agents = len(start_pos)
        # Create the agents at the given positions
        self.agents_pos = []
        for pos in start_pos:
            self.agents_pos.append(copy.copy(pos))
        # Return initial observation, the starting agent locations
        observation = self.agents_pos
        return observation

    '''
    Visualizes the environment
    Any form like vector representation or visualizing using matplotlib will be sufficient
    '''

    def render(self):
        # Create an array of zeros to reprsent the env
        state = np.zeros((self.size, self.size))
        # Draw the agents
        for i in range(self.num_agents):
            state[tuple(self.agents_pos[i])] = 1 - (0.1 * i)
        # Draw the goal
        state[tuple(self.goal_pos)] = 0.5
        # Show the image
        # plt.imshow(state)
        return state

