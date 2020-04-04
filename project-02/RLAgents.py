import numpy as np
import keras
import random
from collections import deque
from collections import namedtuple


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
        if decay_type == 0:
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
        learned_val = reward + self.gamma * \
            np.max(self.qtable[next_row, next_col])
        new_val = (1-self.gamma) * \
            self.qtable[curr_row, curr_col, action] + self.lr*learned_val
        self.qtable[curr_row, curr_col, action] = new_val

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def epsilon_decay(self):
        if self.decay_type == 0:  # linear
            self.epsilon -= self.decay_amt
        if self.decay_type == 1:  # exponential
            self.epsilon = self.epsilon * self.decay_amt


class DQAgent:
    def __init__(self, env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=0, decay_amt=0.995, batch_size=20, epsilon_floor=0.01):
        # def __init__(self, observation_space, action_space):
        self.epsilon = epsilon
        self.epsilon_floor = epsilon_floor
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.decay_amt = decay_amt
        self.decay_type = decay_type

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.memory = deque()
        self.model = self.generate_model()
        self.record = namedtuple(
            "Record", "state action reward next_state done")

    def generate_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_shape=(
            self.observation_space.shape[0],), activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        rec = self.record(state, action, reward, next_state, done)
        self.memory.append(rec)

    def policy(self, observation):
        rand_num = np.random.rand()
        if rand_num < self.epsilon:
            best_action = np.random.randint(self.action_space.n)
        else:
            possible_actions = self.model.predict(observation)[0]
            best_action = np.argmax(possible_actions)
        return best_action

    def step(self, state):
        return self.policy(state)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for rec in batch:
            if rec.done:
                y = rec.reward
            else:
                nextq = self.model.predict(rec.next_state)[0]
                y = (rec.reward + self.gamma * np.amax(nextq))
            q_values = self.model.predict(rec.state)
            q_values[0][rec.action] = y
            self.model.fit(rec.state, q_values, verbose=0)
        self.epsilon_decay()

    def epsilon_decay(self):
        if self.decay_type == 0:  # linear
            self.epsilon -= self.decay_amt
        if self.decay_type == 1:  # exponential
            self.epsilon = self.epsilon * self.decay_amt
        if self.epsilon < self.epsilon_floor:
            self.epsilon = self.epsilon_floor
