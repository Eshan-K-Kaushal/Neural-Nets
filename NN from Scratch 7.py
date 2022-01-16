# with the softmax function
# here we have modified the softmax function enough to prevent any overflow errors
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

X, y = spiral_data(samples =100, classes=3) # data # here we are just creating 100 points per class
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
class softmax:
    def forward(self,inputs):
        exp_vals = np.exp(inputs-np.max(inputs, axis=1, keepdims=True)) # the latter part gives us the max for each row
        probabilities  = exp_vals/np.sum(exp_vals, axis=1, keepdims=True)
        self.output = probabilities
        self.max_row_element = np.max(inputs, axis=1, keepdims=True)

dense1 = LD(2,3)
activation1 = activation_RELU()

dense2 = LD(3,3)
activation2 = softmax()

dense1.fwd(X)
activation1.forward(dense1.output)

dense2.fwd(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
print(activation2.max_row_element[:5])