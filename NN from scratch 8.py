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
        bounded_exp_vals = np.exp(inputs-np.max(inputs, axis=1, keepdims=True)) # the latter part gives us the max for each row
        probabilities  = bounded_exp_vals/np.sum(bounded_exp_vals,
        axis=1, keepdims=True) # divide every element of a row with the sum of that row

        self.output = probabilities
        self.max_row_element = np.max(inputs, axis=1, keepdims=True)

        return self.output
class Loss:
    def soln(self, output, y):
        sample_losses = self.forward(output, y) # we find the sample losses using our
        # categorical loss function
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_categorical_cross_entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred) # the matrix of the confidence vals of the batch
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1: # means they have passed scalar vals
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # a one hot encoded vetor
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            # everything comes to be zero except of the target classes
        negative_log_likes = -np.log(correct_confidences)
        return negative_log_likes




dense1 = LD(2,3)
activation1 = activation_RELU()

dense2 = LD(3,3)
activation2 = softmax()

dense1.fwd(X)
activation1.forward(dense1.fwd(X))

dense2.fwd(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
#print(activation2.max_row_element[:5])

loss_func = Loss_categorical_cross_entropy()
loss = loss_func.forward(activation2.output, y)
print('Loss', loss)

predictions = np.argmax(activation2.output, axis = 1)
accuracy  = np.mean(predictions == y)
print('accuracy - ', accuracy)