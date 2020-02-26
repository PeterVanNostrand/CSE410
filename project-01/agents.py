import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, observation):
        return np.random.choice(self.action_space.n)

class QAgent:
    def __init__(self, env, epsilon=1.0, lr=0.1, gamma=0.8, decay_type=0, decay_amt=0.001):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.qtable = np.zeros((env.size, env.size, env.action_space.n))
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        # decay_type = 0 for linear, 1 for exponential
        self.decay_type = decay_type
        if decay_type==0:
            self.decay_amt = decay_amt
        else:
            self.decay_amt = 1-decay_amt

    def policy(self, observation):
        rand_num = np.random.uniform()

        if rand_num < self.epsilon:
            best_action = np.random.choice(self.action_space.n)
        else:
            row = observation[0]
            col = observation[1]
            possible_actions = self.qtable[row, col]
            best_action = np.argmax(possible_actions)
        return best_action

    def step(self, observation):
        return self.policy(observation)

    def update(self, state, action, reward, next_state):
        # Get the current row/col and next row/col
        curr_row = state[0]
        curr_col = state[1]

        next_row = next_state[0]
        next_col = next_state[1]

        # Update the Q-table to incentivize postive rewards and decentavize negative rewards
        # Learned value = the reward scaled by the probability of occurance, gamma
        learned_val = reward + self.gamma * np.max(self.qtable[next_row, next_col])
        new_val = (1-self.gamma) * self.qtable[curr_row, curr_col, action] + self.lr*learned_val
        self.qtable[curr_row, curr_col, action] = new_val

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def epsilon_decay(self):
        if self.decay_type==0: # linear
            self.epsilon -= self.decay_amt
        if self.decay_type==1: # exponential
            self.epsilon = self.epsilon * self.decay_amt