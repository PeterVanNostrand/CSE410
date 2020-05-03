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
    def __init__(self, env, epsilon=1.0, lr=0.1, gamma=0.8, decay_type=0, decay_amt=0.001, epsilon_floor=0.01, shape=[4, 4, 5]):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.qtable = np.zeros(shape)
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.epsilon_floor = epsilon_floor
        # decay_type = 0 for linear, 1 for exponential
        self.decay_type = decay_type
        if decay_type==0:
            self.decay_amt = decay_amt
        else:
            self.decay_amt = 1-decay_amt

    def policy(self, states):
        rand_num = np.random.uniform()

        if rand_num < self.epsilon:
            best_action = np.random.choice(self.action_space.n)
        else:
            # Get the possible actions, Given by self.qtable[a0row, a0col, a1row, a1col, .... , anrow, ancol]
            possible_actions = self.qtable
            for dim in np.array(states).flatten(): possible_actions = possible_actions[dim]
            best_action = np.argmax(possible_actions)
        return best_action

    def step(self, observation):
        return self.policy(observation)

    def update(self, states, action, reward, next_states):
        # Update the Q-table to incentivize postive rewards and decentavize negative rewards
        # Learned value = the reward scaled by the probability of occurance, gamma
        #   Get the possible actions, Given by self.qtable[a0row, a0col, a1row, a1col, .... , anrow, ancol]
        possible_actions = self.qtable
        for dim in np.array(states).flatten(): possible_actions = possible_actions[dim]
        
        next_possible_actions = self.qtable
        for dim in np.array(next_states).flatten(): next_possible_actions = next_possible_actions[dim]

        learned_val = reward + self.gamma * np.max(next_possible_actions)
        new_val = (1-self.gamma) * possible_actions[action] + self.lr*learned_val
        possible_actions[action] = new_val

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def epsilon_decay(self):
        if self.decay_type==0: # linear
            self.epsilon -= self.decay_amt
        if self.decay_type==1: # exponential
            self.epsilon = self.epsilon * self.decay_amt
        if self.epsilon < self.epsilon_floor:
            self.epsilon = self.epsilon_floor


class DQAgent:
    def __init__(self, env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=0, decay_amt=0.005, batch_size=20, epsilon_floor=0.01, shape=[4,4,5], size=(24, 24)):
        # def __init__(self, observation_space, action_space):
        self.epsilon = epsilon
        self.epsilon_floor = epsilon_floor
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.decay_amt = (1-decay_amt)
        self.decay_type = decay_type

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.memory = deque(maxlen=100)
        self.model = self.generate_model(size, shape)
        self.record = namedtuple(
            "Record", "state action reward next_state done")

    def generate_model(self, size, shape):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(size[0], input_shape=(self.env.num_agents*2,), activation="relu"))
        model.add(keras.layers.Dense(size[1], activation="relu"))
        model.add(keras.layers.Dense(self.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=self.lr))
        return model

    def remember(self, states, action, reward, next_states, done):
        rec = self.record(states, action, reward, next_states, done)
        self.memory.append(rec)

    def policy(self, states):
        rand_num = np.random.rand()
        if rand_num < self.epsilon:
            best_action = np.random.randint(self.action_space.n)
        else:
            possible_actions = self.model.predict(states)[0]
            best_action = np.argmax(possible_actions)
        return best_action

    def step(self, states):
        return self.policy(states)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for rec in batch:
            state = rec.state.reshape(1,self.env.num_agents*2)
            next_state = rec.next_state.reshape(1,self.env.num_agents*2)
            if rec.done:
                y = rec.reward
            else:
                nextq = self.model.predict(next_state)[0]
                y = (rec.reward + self.gamma * np.amax(nextq))
            q_values = self.model.predict(state)
            q_values[0][rec.action] = y
            self.model.fit(state, q_values, verbose=0)

    def epsilon_decay(self):
        if self.decay_type == 0:  # linear
            self.epsilon -= self.decay_amt
        if self.decay_type == 1:  # exponential
            self.epsilon = self.epsilon * self.decay_amt
        if self.epsilon < self.epsilon_floor:
            self.epsilon = self.epsilon_floor

