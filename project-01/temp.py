import numpy as np
import matplotlib.pyplot as plt

epsilon = 1
epsilona = 1
delta = 0.995
delta_epsilon = 0.005
log = [epsilon]
loga = [epsilona]

for i in range(0, 1000):
    epsilona = epsilona * delta
    val = epsilon * np.exp(-delta_epsilon * i)
    log.append(val)
    loga.append(epsilona)

plt.plot(log)
plt.plot(loga)
plt.show()

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
