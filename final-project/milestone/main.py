import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import os
import gym
import environments
from agents import QAgent
from utilities import Logger


def step_agents(agents, states):
    actions = []
    for agent, state in zip(agents, states):
        action = agent.step(state)
        actions.append(action)
    return actions


def update_agents(agents, states, actions, rewards, next_states):
    for agent, state, action, reward, next_state in zip(agents, states, actions, rewards, next_states):
        agent.update(state, action, reward, next_state)


def log_epsilon(epsilon_log, agents):
    global L
    epsilons = [0.0] * len(agents)
    for i in range(len(agents)):
        epsilons[i] = agents[i].epsilon
        L.log(str(agents[i].epsilon), end=",")
    epsilon_log.append(tuple(epsilons))


def log_score(scores_log, scores):
    global L
    scores_log.append(tuple(scores))
    for score in scores:
        L.log(str(score), end="")


def decay_agents(agents):
    for agent in agents:
        agent.epsilon_decay()


def train(agents, env, start_pos, goal_pos, num_episodes):
    global L
    epsilon_log = []
    score_log = []
    moves = ["U", "R", "D", "L", "N"]


    for t in range(num_episodes):
        states = env.reset(start_pos, goal_pos)
        done = False
        scores = [0] * env.num_agents
        while not done:
            # Get an action from each agent
            actions = step_agents(agents, states)
            # Apply the actions to the environment, returns new state and rewards
            next_states, rewards, done, info = env.step(actions)
            for i in range(env.num_agents):
                scores[i] += rewards[i]
            # Perform Q-Learning step on each agent
            update_agents(agents, states, actions, rewards, next_states)
            # Update the state
            states = copy.copy(next_states)
        log_epsilon(epsilon_log, agents)
        log_score(score_log, rewards)
        L.log("")
        print("Epoch: {t:4d}, Epsilons:".format(t=t), end="")
        for i in range(env.num_agents):
            print(" {:.4f}".format(epsilon_log[t][i]), end="")
        print(", Rewards:", end="")
        for i in range(env.num_agents):
            print(" {:.4f}".format(scores[i]), end="")
        print("")
        decay_agents(agents)
    plt.figure()
    plt.plot(epsilon_log)
    plt.title("Epsilon per Epoch")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.savefig("./results/epsilon.png")

    plt.figure()
    plt.plot(score_log)
    plt.title("Cumulative Reward per Epoch")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.savefig("./results/scores.png")

def test(agents, env, start_pos, goal_pos):
    # Plot the agents movements in tiles Left->Right Top->Bottom
    plt.figure()
    plt.suptitle("Multi-Agent Movements")
    plt_size = math.ceil(math.sqrt(env.size*2.0))

    index = 1
    for agent in agents:
        agent.set_epsilon(0.0)

    states = env.reset(start_pos, goal_pos)
    plt.subplot(plt_size, plt_size, index)
    plt.title("Step " + str(index))
    env.render()
    done = False
    while not done:
        # Get an action from each agent
        actions = step_agents(agents, states)
        # Apply the actions to the environment, returns new state and rewards
        next_states, rewards, done, info = env.step(actions)
        # Update the state
        states = copy.copy(next_states)
        # Render and plot
        index += 1
        plt.subplot(plt_size, plt_size, index)
        plt.title("Step " + str(index))
        env.render()
    env.render()
    plt.savefig("./results/movements.png")

def save_learned_path(tables, file_path):
    # TODO - Add second agent
    l = Logger(file_path=file_path, write_mode="w")
    action_map_abbrv = ['U', 'R', 'D', 'L', 'N']

    l.log("Learned optimal action for each state")
    l.log("U=Up, R=Right, D=Down, L=Left")
    l.log("")

    a = 0
    for table in tables:
        l.log("Agent {}".format(a))
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
        l.log("")
        a += 1


if __name__ == "__main__":
    num_agents = 2

    # create results directory and initialize log
    os.makedirs("results", exist_ok=True)
    global L
    L = Logger("./results/log.csv")
    for i in range(num_agents):
        L.log("Epsilon {}".format(i), end=",")
    for i in range(num_agents):
        L.log("Reward {}".format(i), end=",")
    L.log("")

    # Generate a Deterministic Envrionment
    env = environments.DeterministicEnvironment()
    # Create the agents
    agents = []
    for i in range(num_agents):
        agent = QAgent(env, decay_type=1, decay_amt=0.2)
        agents.append(agent)
    start_pos = [[0, env.size-1], [env.size-1, 0]]
    goal_pos = [env.size-1, env.size-1]
    train(agents, env, start_pos, goal_pos, 50)
    test(agents, env, start_pos, goal_pos)

    # Save the learned qtable paths
    tables = []
    for agent in agents:
        tables.append(agent.qtable)
    save_learned_path(tables, "./results/qtables.txt")

    print("See results folder for agent actions, stochastic transition probabilities, learned paths, and epsilon decay")
