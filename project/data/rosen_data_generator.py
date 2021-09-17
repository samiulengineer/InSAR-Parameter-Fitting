import numpy as np
import pandas as pd
from scipy.optimize import minimize


def rosen(a, b, x):
    """The Rosenbrock function"""
    return (sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (b-x[:-1])**2.0))**2


def input_y(stack_size):

    x0 = np.array([10, 10])
    data = np.zeros([stack_size, 4])
    baseline_a = np.zeros([stack_size, 1])
    baseline_b = np.zeros([stack_size, 1])
    x1_orginal = np.zeros([stack_size, 1])
    x2_orginal = np.zeros([stack_size, 1])

    count = 0
    while (count < stack_size):
        a = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)

        def f(x):
            return rosen(a, b, x)
        res = minimize(f, x0, method='nelder-mead',
                       options={'xatol': 1e-8, 'disp': False})
        x1, x2 = res.x
        data[count] = [a, b, x1, x2]
        baseline_a[count] = [a]
        baseline_b[count] = [b]
        x1_orginal[count] = [x1]
        x2_orginal[count] = [x2]
        count += 1

    return baseline_a, baseline_b, x1_orginal, x2_orginal, data


a, b, x1, x2, data = input_y(10)

x = pd.DataFrame(data, columns=["a", "b", "x1", "x2"])
x.to_csv('train_data_10000_check.csv', index=False)
