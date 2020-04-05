import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import RLAgents
from utilities import Logger

def train(agent, env, num_episodes, mode):
    epsilon_log = []
    score_log = [0] * num_episodes
    l = Logger("./results/" + mode + "/results.csv")
    l.log("Epoch", end=",")
    l.log("Epsilon", end=",")
    l.log("Score", end=",")
    if mode=="mountain-car":
        pos_log = [0] * num_episodes
        steps_log = [0] * num_episodes
        l.log("Steps", end=",")
        l.log("Position", end=",")
        l.log("Velocity", end=",")
    l.log("")

    for t in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, (1, env.observation_space.shape[0]))
        score = 0
        done = False
        if mode=="mountain-car":
            pos = 0
            vel = 0
            steps = 0
        while not done:
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((1, env.observation_space.shape[0]))
            if mode=="cart-pole":
                if done: reward = -reward
                score += 1
            else:
                reward += np.absolute(state[0,0]+0.6)
                score += reward
                pos = state[0,0]
                vel = state[0,1]
                steps += 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            epsilon_log.append(agent.epsilon)

        l.log(t, end=",")
        l.log(agent.epsilon, end=",")
        l.log(score, end=",")
        score_log[t] = score
        if mode=="cart-pole":
            print("Epoch: {epoch:4d}, Epsilon: {e:.4f} Score: {score}".format(epoch=t, score=score, e=agent.epsilon))
        else:
            print("Epoch: {epoch:4d}, Epsilon: {e:.4f}, Steps: {s:3d}, Position: {pos:+.4f}, Velocity: {vel:+.4f}, Reward: {rew:.2f},".format(epoch=t, s=steps, e=agent.epsilon, pos=pos, vel=vel, rew=score))
            pos_log[t] = state[0,0]
            steps_log[t] = steps
            l.log(steps, end=",")
            l.log(state[0,0], end=",")
            l.log(state[0,1], end=",")
        l.log("")

    plt.figure()
    plt.plot(epsilon_log)
    plt.title("Epsilon by Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Epsilon")
    plt.savefig("./results/" + mode + "/epsilon.png")

    plt.figure()
    plt.plot(score_log)
    plt.title("Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig("./results/" + mode + "/score.png")

    if mode=="mountain-car":
        plt.figure()
        plt.plot(pos_log)
        plt.title("Position per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Position")
        plt.savefig("./results/" + mode + "/position.png")

        plt.figure()
        plt.plot(steps_log)
        plt.title("Steps per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Number of Steps")
        plt.savefig("./results/" + mode + "/steps.png")


if __name__ == "__main__":
    # Parse command line arguemetns to determine task
    if len(sys.argv) == 2: # If a mode was provided, use it
        mode = sys.argv[1]
    else: # otherwise default to DQN cartpole
        mode = "cart-pole"

    os.makedirs("results/"+mode, exist_ok=True)

    if mode=="cart-pole" :
        print("Running CartPole....")
        env = gym.make("CartPole-v1")
        agent = RLAgents.DQAgent(env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=1, decay_amt=0.995, batch_size=20, epsilon_floor=0.01, size=(24,24))
    if mode=="mountain-car":
        print("Running MountainCar....")
        env = gym.make("MountainCar-v0")
        agent = RLAgents.DQAgent(env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=1, decay_amt=0.998, batch_size=20, epsilon_floor=0.01, size=(128,48))

    # Set seed to constant value for testing    
    np.random.seed(0)
    random.seed(0)
    env.seed(0)

    train(agent, env, 3, mode)
    plt.show()
