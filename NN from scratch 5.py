# a basic structure of an NN from scratch
import numpy as np
from nnfs.datasets import spiral_data
import nnfs
np.random.seed(0)
X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
# this here below is the RELU function basis
# for i in inputs:
#     if i > 0:
#         output.append(max(0,i))
# print(output)
#Now we apple the RELU on the LD
'''
Stuff
'''

X, y = spiral_data(100, 3) # data # here we are just creating 100 points per class
class LD:
    def __init__(self, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons
        self.wts = 0.10*np.random.randn(inputs, neurons)
        self.b = np.zeros((1, neurons)) # bias is dependent on the no of neurons
    def fwd(self, X):
        self.output = np.dot(X, self.wts) + self.b
        return self.output
class activation_RELU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # gets the max ones from the list of inputs
l1 = LD(2,5)
act1 = activation_RELU() # activation object
l1.fwd(X)
print(l1.output) # all the negatives can still be seen
act1.forward(l1.output)
print(act1.output) # all the negatives are gone

#l2 = LD(4,4)
#print(l1.wts)
#l2.fwd(l1.output)
#print(l2.output)