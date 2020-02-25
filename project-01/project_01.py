import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import gym
from gym import spaces
from environments import DeterministicEnvironment
from environments import StochasticEnvironment
from agents import QAgent

def train(agent, env, num_episodes, title):
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

def test(agent, env, title):
    # Plot the agents movements in tiles Left->Right Top->Bottom
    plt.figure()
    plt.suptitle(title)
    plt_size = math.ceil(math.sqrt(env.size*2.0))
    index = 1
    agent.set_epsilon(0.0)

    observation = env.reset()
    done = False
    while not done:
        plt.subplot(plt_size, plt_size, index)
        plt.title("Step " + str(index))
        env.render()
        statet = copy.copy(observation)
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        index += 1
    plt.subplot(plt_size, plt_size, index)
    plt.title("Step " + str(index))
    env.render()

if __name__ == "__main__":
    # Deterministic Envrionment
    denv = DeterministicEnvironment()
    dagent = QAgent(denv, decay_type=1, decay_amt=0.005)
    train(dagent, denv, 1000, "Epsilon per Episode: Deterministic Environment")
    test(dagent, denv, "QAgent in Deterministic Environment")

    # Stochastic Envrionment
    senv = StochasticEnvironment()
    sagent = QAgent(denv, decay_type=1, decay_amt=0.005)
    train(sagent, senv, 1000, "Epsilon per Episode: Stochastic Environment")
    test(sagent, senv, "QAgent in Stochastic Environment")

    plt.show()

    

    