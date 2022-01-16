import numpy as np

soft_op = [[0.7, 0.1, 0.2],
           [0.1, 0.5, 0.4],
           [0.02, 0.9, 0.08]]
soft_max_op = np.array([[0.7, 0.1, 0.2],
           [0.1, 0.5, 0.4],
           [0.02, 0.9, 0.08]]) # these are the predicted vals
class_targets  = [0,1,1] # lets say [cat, dog, human]

for target_idx, distribution in zip(class_targets, soft_op): # if not array
    print(distribution[target_idx])
# or
print(soft_max_op[[0,1,2], class_targets])
# for if we take soft_max_op as an np array here softmax is 2d
# so the first set of entries show the rows and
# the class_targets holding the labels we desire, show the column
# vals we have to take from the respective rows to fill up the
# soft_max_ouptut list
'''
now inorder to make it a bit more flexible and not hardcode it we just put len
of soft_max output as range 
'''
print('By using the range (len(soft_max), class_targets) \n- ',
      soft_max_op[range(len(soft_max_op)), class_targets])

# now we add the negative log to get the loss
print('By using the range (len(soft_max), class_targets) and then '
      'applying log to it to make it into ' 
      'loss\n- ',-np.log(soft_max_op[range(len(soft_max_op)), class_targets]))

'''
in the end, we just get the avg loss
'''
avg_loss = np.mean(-np.log(soft_max_op[range(len(soft_max_op)), class_targets]))
print('avg loss - ', avg_loss)
# but we have to take care for 0 since -log(0) since that gives an inf on -log
# and gives infinity on mean/loss
'''
So we decide to clip the vals 
'''
clpped_vals = np.clip(soft_max_op, 1e-7, 1-1e-7)
print('Clipped Vals are: \n', clpped_vals)
