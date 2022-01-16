# Batches allow us to generalize
# These help in the fitment, everything can be done in 1 go
import numpy as np
WT = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
B = [2, 3, 0.5]
inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

# second layer
WT2 = [[0.1, -0.14, -0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
B2 = [-1, 2, -0.5]

# multilayer neurons
layer1_outputs = np.dot(inputs, np.array(WT).T) + B
# output of the previous layer is the input of the next
layer2_outputs = np.dot(layer1_outputs, np.array(WT2).T) + B2

'''
Below is basic, I have updated the code above to be a bit more technical 
'''
# now the input has a batch size of 3
#inputs = np.array(inputs) # convert this into an array
#WT = np.array(WT)
#op = np.dot(inputs, WT.T) + B # 3 neurons with 4 inputs
# we reversed the order of multiplication since we need better readability
'''
above is the mat mult of 3X4 with 4X3
'''
print(layer2_outputs)