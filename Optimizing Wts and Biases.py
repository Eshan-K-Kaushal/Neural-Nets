# the focus of this module is to make one learn about the NNs way of
#updating the wts and biases 
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x ** 2
x = np.arange(0, 50, 0.001)
y = f(x)
#print(len(x))

plt.plot(x,y)
plt.show()

#approx_derivative = (y2-y1)/(x2-x1)
colors = ['k', 'g', 'r', 'b', 'c']
def approx_tangent_line(x, approximate_derivative, b):
    return approx_derivative*x + b

for i in range(5):

    p2_delta = 0.0001
    x1 = i
    x2 = x1+p2_delta # where the p2_delta stands asa small step / the derivative val
    y1 = f(x1)
    y2 = f(x2)
    print((x1, y1), (x2, y2))
    approx_derivative = (y2-y1)/(x2-x1)
    b = y2 - approx_derivative*x2

    to_plot = [x1-0.9, x1, x1+0,9]
    plt.scatter(x1, y1, c = colors[i])
    plt.plot(to_plot, [approx_tangent_line(point, approx_derivative, b)for point in to_plot],
             c=colors[i])
    print('Approximate derivative for f(x)', f, f'where x = {x1} is {approx_derivative}')
plt.show()

