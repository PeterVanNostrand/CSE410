import numpy as np
import matplotlib.pyplot as plt

# arr = [1, 2, 3, 4, 5, 6]
# p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# choice = np.random.choice(arr, p=p)
# print(choice)

# epsilon = 1
# epsilona = 1
# delta = 0.995
# delta_epsilon = 0.005
# log = [epsilon]
# loga = [epsilona]

# for i in range(0, 1000):
#     epsilona = epsilona * delta
#     val = epsilon * np.exp(-delta_epsilon * i)
#     log.append(val)
#     loga.append(epsilona)

# plt.plot(log)
# plt.plot(loga)
# plt.show()

# size = 4
# transition_matrix = []
# for x in range(size + 1):
#     state_x = []
#     for y in range(size + 1):
#         state_y = []
#         for a in range(4):
#             one_hot = np.zeros(4)
#             one_hot[a] = 1
#             state_y.append(one_hot)
#         print(state_y)
#         # state_x.append(state_y)
#     print(state_x)
#     transition_matrix.append(state_x)
# # print(transition_matrix)

from utilities import Logger

l = Logger(file_path="transition_probabilities.txt", write_mode="w")
print("See transition_probabilities.txt for transition probabilities")

transition_prob = np.zeros((4, 4, 4, 4))
acts = ['U', 'R', 'D', 'L']
idxs = [0, 1, 2, 3]
tab_row = "|  {R}  |  {C}  |  {A}  | {p0:5.2f} | {p1:5.2f} | {p2:5.2f} | {p3:5.2f} |"

l.log("| row | col | act |   U   |   R   |   D   |   L   |")
l.log("|-----|-----|-----|-------|-------|-------|-------|")

for row in range(4):
    for col in range(4):
        for act in range(4):
            p = transition_prob[row][col][act] # prob for actions given this row, col, act tuple
            p[act] = np.random.uniform(0.8, 1.0) # desired action is always >= 80% likely
            rem = 1.0 - p[act] # remaining probability
            for i in range(4): # randomly select other action probabilities, keeping total prob=1
                if i==act: continue
                p[i] = np.random.uniform(0, rem)
                rem -= p[i]
            p[np.random.choice(idxs)] += rem # add leftover probability to a random action
            op = p * 100 # output probability, for print formated to XX.XX %
            l.log(tab_row.format(R=row, C=col, A=acts[act], p0=op[0], p1=op[1], p2=op[2], p3=op[3]))