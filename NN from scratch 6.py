# here we will take a look at the softmax activation function
# exponentiating helps us to get rid of the negative signs without losing there essence
import numpy as np
list1 = [4.8, 1.21, 2.385]
list11 = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]
array_of_list11 = np.array(list11)
def softmax(list1):
    def exp(x):
        return np.exp(x)
    exp_list = list(map(exp, list1))
    sum_exp_list = np.sum(exp_list) # axis = 1 for rows and 0 for cols
    exp_vals = []
    for i in exp_list:
        exp_vals.append(i/sum_exp_list) # all the vals after softmax has been applied to them
    return exp_vals
print(softmax(list1))

exp_vals = np.exp(list11)
print('exp_vals is - \t', exp_vals)
print(np.sum(list11, axis=1, keepdims=True))
normalization_const = exp_vals/np.sum(exp_vals, axis=1, keepdims=True)
print(normalization_const)