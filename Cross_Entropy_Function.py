import numpy as np
import math
# we try to solve for x given an input b
# e**x = b

b = 5.2 # example
print(np.log(b))
print(math.e ** np.log(b))

soft_max_output = [0.7, 0.1, 0.2] # after the neuron had done the processing
# the vals have been put through the softmax for better fitting and better vals
# we get vals in between 0 and 1
# kinda like probability
target_op = [1,0,0]

loss = -(math.log(soft_max_output[0]) * target_op[0] +
         math.log(soft_max_output[1]) * target_op[1]+
         math.log(soft_max_output[2]) * target_op[2])
print('Loss is- ', loss)
# same as doing
#loss = -math.log(softmax_output[0]) since target op was 1, we dont write/consider it
#loss increases confidence decreases

# working on the batch now
# we also have many targets
