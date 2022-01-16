import numpy as np
WT = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
B = [2, 3, 0.5]
inputs = [1, 2, 3, 2.5] # outputs from 3 neurons in the previous layer
op = np.dot(WT, inputs)
# final total
op1 = np.sum(op) + B[0]
print(op1)
#now layer of neurons
op_new = 0
op_new_list = []
for j in range(len(WT)):
    op_new += np.dot(WT[j], inputs) + B[j]
    op_new_list.append(op_new)
    op_new = 0
print(op_new_list)
# or
op_one_line = np.dot(WT, inputs) + B # shape 3X4 mult with shape 4X1 gives shape 3X1
print('In one line using dot', op_one_line)