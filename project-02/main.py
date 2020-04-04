import random
import os
import numpy as np
import matplotlib.pyplot as plt
import gym
import RLAgents
from utilities import Logger

def train(agent, env, num_episodes):
    epsilon_log = [0] * num_episodes
    score_log = [0] * num_episodes
    leps = Logger("./results/epsilon.csv")
    lscore = Logger("./results/score.csv")

    for t in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, (1, env.observation_space.shape[0]))
        score = 0
        done = False
        while not done:
            score += 1
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = -reward
            next_state = next_state.reshape((1, env.observation_space.shape[0]))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
        print("Epoch: {epoch}, Epsilon: {e:.4f} Score: {score}".format(
            epoch=t, score=score, e=agent.epsilon))
        epsilon_log[t] = agent.epsilon
        score_log[t] = score
        leps.log(agent.epsilon, end=",")
        lscore.log(score, end=",")

    plt.figure()
    plt.plot(epsilon_log)
    plt.title("Epsilon by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.savefig("./results/epsilon.png")

    plt.figure()
    plt.plot(score)
    plt.title("Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig("./results/score.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    env = gym.make("CartPole-v1")
    np.random.seed(0)
    random.seed(0)
    env.seed(0)
    agent = RLAgents.DQAgent(env, epsilon=1.0, lr=0.001, gamma=0.95,
                             decay_type=1, decay_amt=0.995, batch_size=20, epsilon_floor=0.01)
    train(agent, env, 100)
    plt.show()
