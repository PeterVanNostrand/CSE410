import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import os
import gym
from gym import spaces
from environments import DeterministicEnvironment
from environments import StochasticEnvironment
from agents import QAgent
from utilities import Logger

def train(agent, env, num_episodes, title, file_path):
    epsilon_log = [agent.epsilon]
    delta_epsilon = agent.epsilon/num_episodes

    for t in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            statet = copy.copy(observation)
            action = agent.step(observation)
            observation, reward, done, info = env.step(action)
            agent.update(statet, action, reward, observation)
        agent.epsilon_decay()
        epsilon_log.append(agent.epsilon)
    plt.figure()
    plt.plot(epsilon_log)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.savefig(file_path)

def test(agent, env, title, file_path, extra=False):
    # Plot the agents movements in tiles Left->Right Top->Bottom
    index = 10

    plt.figure()
    plt.suptitle(title)
    plt_size = math.ceil(math.sqrt(env.size*2.0))
    plt_x = plt_size
    plt_y = plt_size
    if extra: plt_y += 1

    index = 1
    agent.set_epsilon(0.0)

    observation = env.reset()
    done = False
    while not done:
        plt.subplot(plt_x, plt_y, index)
        plt.title("Step " + str(index))
        env.render()
        statet = copy.copy(observation)
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        index += 1
    plt.subplot(plt_x, plt_y, index)
    plt.title("Step " + str(index))
    env.render()
    plt.savefig(file_path)

def save_learned_path(table, file_path):
    l = Logger(file_path=file_path, write_mode="w")
    action_map_abbrv = ['U', 'R', 'D', 'L']

    l.log("Learned optimal action for each state")
    l.log("U=Up, R=Right, D=Down, L=Left")

    # Print header row
    l.log("|   |", end="")
    for i in range(0, table.shape[0]):
        l.log(" %1d |" % (i), end="")
    l.log("")

    # l.log divider row
    l.log("|", end="")
    for i in range(0, table.shape[0]+1):
        l.log("---|", end="")
    l.log("")

    # Fill in rows
    for i in range(0, table.shape[0]):
        l.log("| %1d |" % (i), end="")
        for j in range(0, table.shape[1]):
            action = np.argmax(table[i, j])
            direction = action_map_abbrv[action]
            l.log(" %c |" % (direction), end="")
        l.log("")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Deterministic Envrionment
    denv = DeterministicEnvironment()
    dagent = QAgent(denv, decay_type=1, decay_amt=0.005)
    train(dagent, denv, 1000, "Epsilon per Episode for Deterministic Environment", "results/epsilon_deterministic.png")
    test(dagent, denv, "QAgent in Deterministic Environment", "results/agent_deterministic.png")
    save_learned_path(dagent.qtable, file_path="results/qtable_deterministic.txt")

    # Stochastic Envrionment
    senv = StochasticEnvironment()
    sagent = QAgent(denv, decay_type=1, decay_amt=0.005)
    train(sagent, senv, 1000, "Epsilon per Episode for Stochastic Environment", "results/epsilon_stochastic.png")
    test(sagent, senv, "QAgent in Stochastic Environment", "results/agent_stochastic.png", extra=True)
    save_learned_path(sagent.qtable, file_path="results/qtable_stochastic.txt")

    print("See results folder for agent actions, stochastic transition probabilities, learned paths, and epsilon decay")
