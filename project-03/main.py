import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import RLAgents
from utilities import Logger
import copy


def train_reinforce(agent, env, num_episodes, filepath):
    l = Logger(filepath + "score.csv")
    l.log("Epoch", end=",")
    l.log("Score", end=",")
    l.log("")
    score_log = []

    t = 0
    solved = False
    while t < num_episodes and not solved:
        score = 0
        gradients = []
        rewards = []
        done = False

        state = env.reset()
        while not done:
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            gradient = agent.gradient(state, action)
            rewards.append(reward)
            gradients.append(gradient)
            score += reward
            state = next_state
        agent.update(rewards, gradients)

        l.log("{e}, {s},".format(e=t, s=score))
        print("Epoch {e:3d}, Score: {s:3.0f}".format(e=t, s=score))
        score_log.append(score)

        t += 1
        if t > 100:
            avg_last_100 = np.mean(score_log[-100:])
            if avg_last_100 > 475.0:
                solved = True

    plt.figure()
    plt.plot(score_log)
    plt.title("Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(filepath + "score.png")


def train_tdac(agent, env, num_episodes, filepath):
    l = Logger(filepath + "score.csv")
    l.log("Epoch", end=",")
    l.log("Score", end=",")
    l.log("")
    score_log = []

    t = 0
    solved = False
    while t < num_episodes and not solved:    
        score = 0
        done = False

        state = env.reset()
        state = state.reshape((1, env.observation_space.shape[0]))
        while not done:
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((1, env.observation_space.shape[0]))
            score += reward
            agent.update(state, action, reward, next_state, done)
            state = copy.copy(next_state)

        l.log("{e}, {s},".format(e=t, s=score))
        print("Epoch {e:3d}, Score: {s:3.0f}".format(e=t, s=score))
        score_log.append(score)

        t += 1
        if t > 100:
            avg_last_100 = np.mean(score_log[-100:])
            if avg_last_100 > 475.0:
                solved = True
    plt.figure()
    plt.plot(score_log)
    plt.title("Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(filepath + "score.png")


if __name__ == "__main__":
    filepath = "./project-03/results/reinforce/"
    os.makedirs(filepath, exist_ok=True)
    env = gym.make("CartPole-v1")
    agent = RLAgents.PolicyAgent(env, lr = 0.0001, gamma = 0.99)
    train_reinforce(agent, env, 1, filepath)

    filepath = "./project-03/results/actor-critic/"
    os.makedirs(filepath, exist_ok=True)
    env = gym.make("CartPole-v1")
    agent = RLAgents.AC2Agent(env, lr_actor=0.001, lr_critic=0.001, gamma=0.99)
    train_tdac(agent, env, 10000, filepath)

    plt.show()
