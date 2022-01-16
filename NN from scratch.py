import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# coding 1 neuron
inputs = [1, 2, 3, 2.5] # outputs from 3 neurons in the previous layer
# adding one more input
wt = [0.2, 0.8, -0.5, 1.0] # we have 3 wts associated to every neuron
bias = 2 # every unique neuron has one bias
output = 0
for i in range(len(inputs)):
    print(inputs[i],",",wt[i])
    output += (inputs[i] * wt[i]) # add all of them together
output += bias
print(output)
'''
3 NEURONS AND 4 INPUTS
'''
# 3 neurons with 4 inputs
# input stays the same
wt1 = [0.2, 0.8, -0.5, 1.0]
wt2 = [0.5, -0.91, 0.26, -0.5]
wt3 = [-0.26, -0.27, 0.17, 0.87]
WT = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
b1 = 2
b2 = 3
b3 = 0.5
B = [2, 3, 0.5]
print('Here is the 3 neurons with 4 inputs')
op = []
op1 = 0
#print(len(WT))
for j in range(len(WT)):
    for i in range(len(inputs)):
        op1 += WT[j][i] * inputs[i]
    op1 += B[j]
    op.append(op1)
    op1 = 0 # reset op1 to clear it
print(op)
# you can also use zip