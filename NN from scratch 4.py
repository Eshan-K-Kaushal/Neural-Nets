import numpy as np
np.random.seed(0)
X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
# training the dataset
# introducing the hidden layers

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.wts = 0.10*np.random.randn(n_inputs, n_neurons) # stops us from transposing later
        self.b = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.wts) + self.b
        return self.output
layer1 = Layer_Dense(4, 4) # you can have any number of neurons
layer2 = Layer_Dense(4, 2)
layer1.forward(X)
print('Layer 1', '\n',layer1.output)
layer2.forward(layer1.output)
print('Layer 2', '\n',layer2.output)
