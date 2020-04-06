import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import RLAgents
from utilities import Logger


def train(agent, env, num_episodes, env_name, filepath):
    epsilon_log = []
    score_log = [0] * num_episodes
    l = Logger(filepath + "/results.csv")
    l.log("Epoch", end=",")
    l.log("Epsilon", end=",")
    l.log("Score", end=",")
    if env_name == "mountain-car":
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
        if env_name == "mountain-car":
            pos = 0
            vel = 0
            steps = 0
        while not done:
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(
                (1, env.observation_space.shape[0]))
            if env_name == "cart-pole":
                if done:
                    reward = -reward
                score += 1
            else:
                reward += np.absolute(state[0, 0]+0.6)
                score += reward
                pos = state[0, 0]
                vel = state[0, 1]
                steps += 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            epsilon_log.append(agent.epsilon)

        l.log(t, end=",")
        l.log(agent.epsilon, end=",")
        l.log(score, end=",")
        score_log[t] = score
        if env_name == "cart-pole":
            print("Epoch: {epoch:4d}, Epsilon: {e:.4f} Score: {score}".format(
                epoch=t, score=score, e=agent.epsilon))
        else:
            print("Epoch: {epoch:4d}, Epsilon: {e:.4f}, Steps: {s:3d}, Position: {pos:+.4f}, Velocity: {vel:+.4f}, Reward: {rew:.2f},".format(
                epoch=t, s=steps, e=agent.epsilon, pos=pos, vel=vel, rew=score))
            pos_log[t] = state[0, 0]
            steps_log[t] = steps
            l.log(steps, end=",")
            l.log(state[0, 0], end=",")
            l.log(state[0, 1], end=",")
        l.log("")

    plt.figure()
    plt.plot(epsilon_log)
    plt.title("Epsilon by Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Epsilon")
    plt.savefig(filepath + "/epsilon.png")

    plt.figure()
    plt.plot(score_log)
    plt.title("Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(filepath + "/score.png")

    if env_name == "mountain-car":
        plt.figure()
        plt.plot(pos_log)
        plt.title("Position per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Position")
        plt.savefig(filepath + "/position.png")

        plt.figure()
        plt.plot(steps_log)
        plt.title("Steps per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Number of Steps")
        plt.savefig(filepath + "/steps.png")


def config_match(my_env, my_alg):
    global env_name
    global alg
    if env_name == my_env or env_name == "all":
        if alg == my_alg or alg == "all":
            return True
    else:
        return False


if __name__ == "__main__":
    global env_name
    global alg
    # Parse command line arguemetns to determine task
    if len(sys.argv) == 4:  # If a configuation was provided, use it
        env_name = sys.argv[1]
        alg = sys.argv[2]
        its = int(sys.argv[3])
    else:  # otherwise run all
        env_name = "all"
        alg = "all"
        its = 250
        print("Running all agents all environments...")

    if config_match("cart-pole", "dqn"):
        print("Running CartPole DQN...")
        os.makedirs("./results/cart-pole/dqn", exist_ok=True)
        env = gym.make("CartPole-v1")
        agent = RLAgents.DQAgent(env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=1,
                                 decay_amt=0.995, batch_size=20, epsilon_floor=0.01, size=(24, 24))
        train(agent, env, its, "cart-pole", "./results/cart-pole/dqn")
    if config_match("cart-pole", "d2qn"):
        print("Running CartPole D2QN...")
        os.makedirs("./results/cart-pole/d2qn", exist_ok=True)
        env = gym.make("CartPole-v1")
        agent = RLAgents.D2QAgent(env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=1,
                                  decay_amt=0.995, batch_size=20, epsilon_floor=0.01, size=(24, 24))
        train(agent, env, its, "cart-pole", "./results/cart-pole/d2qn")
    if config_match("mountain-car", "dqn"):
        print("Running MountainCar DQN...")
        os.makedirs("./results/mountain-car/dqn", exist_ok=True)
        env = gym.make("MountainCar-v0")
        agent = RLAgents.DQAgent(env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=1,
                                 decay_amt=0.998, batch_size=20, epsilon_floor=0.01, size=(128, 48))
        train(agent, env, its, "mountain-car", "./results/mountain-car/dqn")
    if config_match("mountain-car", "d2qn"):
        print("Running MountainCar D2QN...")
        os.makedirs("./results/mountain-car/d2qn", exist_ok=True)
        env = gym.make("MountainCar-v0")
        agent = RLAgents.D2QAgent(env, epsilon=1.0, lr=0.001, gamma=0.95, decay_type=1,
                                  decay_amt=0.998, batch_size=20, epsilon_floor=0.01, size=(128, 48))
        train(agent, env, its, "mountain-car", "./results/mountain-car/d2qn")

    plt.show()
