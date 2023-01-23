import numpy as np
import jax.numpy as jnp
from linear_eq_functional import _cal_LU, forward_sub, back_sub

def LU_decomposition(mat):
    n, m = mat.shape
    assert n == m, "Please Input Square Matrix"
    L, U = np.zeros((n, n)), np.eye(n)
    L[:, 0] = mat[:, 0]
    U[0, :] = mat[0, :] / L[0, 0]
    L, U = _cal_LU(L, U, mat)
    return L, U

def LU_solution(mat, b):
    L, U = LU_decomposition(mat)
    y = forward_sub(L, b)
    x = back_sub(U, y)
    return x

def LU_inverse(mat):
    n, m = mat.shape
    assert n == m, "Please Input Square Matrix"
    solutions = np.eye(n)
    ans = []
    for i in range(0, n):
        ans.append(LU_solution(mat, solutions[i]))
    ans = jnp.stack(ans)
    return ans

if __name__ == '__main__':
    mat = np.array([[4, -2, -3, 6], [-6, 7, 6.5, -6], [1, 7.5, 6.25, 5.5], [-12, 22, 15.5, -1]], np.float64)
    b = np.array([12, -6.5, 16, 17], np.float64)
    L, U = LU_decomposition(mat)
    print("LU Decomposition.")
    print("L : ", L)
    print("U : ", U)
    print("Linear Equation Solution : ",LU_solution(mat, b))
    # Correct !!