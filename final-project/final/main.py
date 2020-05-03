import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import os
import gym
import environments
from agents import QAgent
from agents import DQAgent
from utilities import Logger


def step_agents(agents, states):
    actions = []
    for agent in agents:
        action = agent.step(states)
        actions.append(action)
    return actions

'''
    uses numpy cumulative sum to compute a simple moving average over
    a window of size n
'''
def moving_average(series, n):
    # given [1, 2, 3, 4]
    sums = np.cumsum(series, dtype=float) # [1, 1+2, 1+2+3, 1+2+3+4]
    sums[n:] = sums[n:] - sums[:-n] # [1+2+3, 1+2+3+4] - [1, 1+2] = [2+3, 3+4]
    return sums[n-1:] / n # skip first n-1 values and divide by n for average

def update_agents(agents, states, actions, rewards, next_states):
    for agent, action, reward in zip(agents, actions, rewards):
        agent.update(states, action, reward, next_states)

def remember_agents(agents, states, actions, rewards, next_states, done):
    for agent, action, reward in zip(agents, actions, rewards):
        agent.remember(states, action, reward, next_states, done)

def replay_agents(agents):
    for agent in agents:
        agent.replay()

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

def train_dqn(agents, env, start_pos, goal_pos, num_episodes):
    global L
    epsilon_log = []
    score_log = []

    for t in range(num_episodes):
        states = env.reset(start_pos, goal_pos)
        states = np.array(states).reshape(1,env.num_agents*2)
        done = False
        scores = [0] * env.num_agents
        while not done:
            # Get an action from each agent
            actions = step_agents(agents, states)
            # Apply the actions to the environment, returns new state and rewards
            next_states, rewards, done, info = env.step(actions)
            next_states = np.array(next_states).reshape(1,env.num_agents*2)
            for i in range(env.num_agents):
                scores[i] += rewards[i]
            # Store Transitions
            remember_agents(agents, states, actions, rewards, next_states, done)
            # Update the state
            s = copy.copy(next_states)
            # Replay agents
            replay_agents(agents)
        decay_agents(agents)
        log_epsilon(epsilon_log, agents)
        log_score(score_log, scores)
        L.log("")
        print("Epoch: {t:4d}, Epsilons:".format(t=t), end="")
        for i in range(env.num_agents):
            print(" {:.4f}".format(epsilon_log[t][i]), end="")
        print(", Rewards:", end="")
        for i in range(env.num_agents):
            print(" {:.4f}".format(scores[i]), end="")
        print("")
    
    decay_amt = 1 - agents[0].decay_amt
    lr = agents[0].lr
    
    plt.figure()
    plt.plot(epsilon_log)
    plt.title("Epsilon per Epoch")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.savefig("./results/lr={a:.4f}_decay_amt={e:.4f}_epsilon.png".format(a=lr, e=decay_amt))


    # Create subplots
    fig, axs = plt.subplots(env.num_agents, 1, sharex=True, sharey=True,)
    # Create and label big plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.suptitle("Cumulative Reward per Epoch")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    # Plot the score and moving average score for each agent on separate subplot
    score_log = np.array(score_log)
    window_size = np.max((int(score_log.shape[1]/1000), 10))
    if score_log.shape[0] < window_size: window_size = 1
    for i in range(score_log.shape[1]):
        scores_i = score_log.T[i] # pylint: disable=E1136  # pylint/issues/3139
        averages_i = moving_average(score_log.T[i], window_size) # pylint: disable=E1136  # pylint/issues/3139
        axs[i].plot(scores_i)
        axs[i].plot(averages_i)
    plt.savefig("./results/lr={a:.4f}_decay_amt={e:.4f}_scores.png".format(a=lr, e=decay_amt))

def train(agents, env, start_pos, goal_pos, num_episodes):
    global L
    epsilon_log = []
    score_log = []

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
        log_score(score_log, scores)
        L.log("")
        print("Epoch: {t:4d}, Epsilons:".format(t=t), end="")
        for i in range(env.num_agents):
            print(" {:.4f}".format(epsilon_log[t][i]), end="")
        print(", Rewards:", end="")
        for i in range(env.num_agents):
            print(" {:.4f}".format(scores[i]), end="")
        print("")
        decay_agents(agents)
    
    decay_amt = 1 - agents[0].decay_amt
    lr = agents[0].lr
    
    plt.figure()
    plt.plot(epsilon_log)
    plt.title("Epsilon per Epoch")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.savefig("./results/lr={a:.4f}_decay_amt={e:.4f}_epsilon.png".format(a=lr, e=decay_amt))


    # Create subplots
    fig, axs = plt.subplots(env.num_agents, 1, sharex=True, sharey=True,)
    # Create and label big plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.suptitle("Cumulative Reward per Epoch")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    # Plot the score and moving average score for each agent on separate subplot
    score_log = np.array(score_log)
    window_size = np.max((int(score_log.shape[1]/1000), 10))
    if score_log.shape[0] < window_size: window_size = 1
    for i in range(score_log.shape[1]):
        scores_i = score_log.T[i] # pylint: disable=E1136  # pylint/issues/3139
        averages_i = moving_average(score_log.T[i], window_size) # pylint: disable=E1136  # pylint/issues/3139
        axs[i].plot(scores_i)
        axs[i].plot(averages_i)
    plt.savefig("./results/lr={a:.4f}_decay_amt={e:.4f}_scores.png".format(a=lr, e=decay_amt))

def test(agents, env, start_pos, goal_pos, dqn=False):
    # Plot the agents movements in tiles Left->Right Top->Bottom
    plt.figure()
    plt.suptitle("Multi-Agent Movements")
    plt_size = math.ceil(math.sqrt(env.size*2.0))

    index = 1
    for agent in agents:
        agent.set_epsilon(0.0)

    states = env.reset(start_pos, goal_pos)
    if dqn: states = np.array(states).reshape(1,env.num_agents*2)
    plt.subplot(plt_size, plt_size, index)
    plt.title("Step " + str(index))
    boards = []
    boards.append(env.render())
    done = False
    while not done:
        # Get an action from each agent
        actions = step_agents(agents, states)
        # Apply the actions to the environment, returns new state and rewards
        next_states, rewards, done, info = env.step(actions)
        if dqn: next_states = np.array(next_states).reshape(1,env.num_agents*2)
        # Update the state
        states = copy.copy(next_states)
        # Render and plot
        index += 1
        boards.append(env.render())

    decay_amt = 1 - agents[0].decay_amt
    lr = agents[0].lr

    # Plot the movements
    plt.figure()
    plt.title("Multi-Agent Movements, lr={a:.4f}, decay_amt={e:.4f}".format(a=lr, e=decay_amt))
    max_idx = len(boards) + 1
    ncols = 4
    nrows = int(np.ceil(max_idx/ncols))
    for idx in range(1, max_idx):
        plt.subplot(nrows, ncols, idx)
        plt.imshow(boards[idx-1])
        plt.title("Step " + str(idx))
    plt.savefig("./results/lr={a:.4f}_decay_amt={e:.4f}_movements.png".format(a=lr, e=decay_amt))


if __name__ == "__main__":
    size = 6
    num_agents = 4
    alpha = 0.05
    epochs = 10**3
    epsilon_decay = 5 * 10**-3
    floor=0.03
    gamma=0.95
    dqn = True

    # create results directory and initialize log
    os.makedirs("results", exist_ok=True)
    global L
    file_path = "./results/lr={a:.4f}_decay_amt={e:.4f}_log.csv".format(a=alpha, e=epsilon_decay)
    L = Logger(file_path, write_mode="w")
    for i in range(num_agents):
        L.log("Epsilon {}".format(i), end=",")
    for i in range(num_agents):
        L.log("Reward {}".format(i), end=",")
    L.log("")

    # Generate a Deterministic Envrionment
    env = environments.MAEnvironment(size=size, num_agents=num_agents)
    
    # Create the agents
    shape = np.full((num_agents*2,), env.size)
    shape = np.append(shape, env.action_space.n)
    agents = []
    for i in range(num_agents):
        if not dqn: agent = QAgent(env, decay_type=1, decay_amt=epsilon_decay, lr=alpha, epsilon_floor=floor, gamma=gamma, shape=shape)
        else: agent = DQAgent(env, decay_type=1, decay_amt=epsilon_decay, lr=alpha, epsilon_floor=floor, gamma=gamma, shape=shape, size=(48,48))
        agents.append(agent)

    if num_agents == 1:
        start_pos = [[0,0]]
        goal_pos = [[env.size-1, env.size-1]]
    elif num_agents == 2:
        start_pos = [[0, env.size-1], [env.size-1, 0]]
        goal_pos = [env.size-1, env.size-1]
    elif num_agents == 3:
        start_pos = [[0,0], [int(env.size/2), 0], [env.size-1, 0]]
        goal_pos = [int(env.size/2), env.size-1]
    elif num_agents == 4:
        start_pos = [[0,0], [0, env.size-1], [env.size-1, 0], [env.size-1, env.size-1]]
        goal_pos = [int(env.size/2), int(env.size/2)]

    if not dqn: train(agents, env, start_pos, goal_pos, epochs)
    else: train_dqn(agents, env, start_pos, goal_pos, epochs)
    test(agents, env, start_pos, goal_pos, dqn=dqn)

    print("See results folder for agent actions, epsilon decay, and scores")
