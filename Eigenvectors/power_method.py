# iterative method이다. -> 굳이 jax를 사용할 이유가 없음.

import numpy as np
from Solving_Linear_System.LU_decomposition import LU_inverse

def power_method(a, iteration, isinverse=False): # a : n, n
    if isinverse :
        a = LU_inverse(a)
    n, m = a.shape
    assert n == m, "Please Input Square Matrix"
    a = a.reshape(n, 1, n)
    x = np.ones((1, n, 1)) # n
    for i in range(iteration):
        x = np.matmul(a, x.reshape(1, n, 1))
        x = x / x.max() # normalized eigenvector
    return x.reshape(n)


if __name__ == '__main__':
    a = np.array([[4, 2, -2], [-2, 8, 1], [2, 4, -4]])
    print(power_method(a, 8, isinverse=False))