
import functools
import jax.numpy as jnp
import jax;
import numpy as np

jax.config.update("jax_platform_name", 'cpu') # cpu 로 돌리는게 빠르다 그냥.
"""
    이러한 Numerical Method는 matrix의 크기에 따라 control flow (for문)의 iteration이
    비례해서 커지게 된다. 물론 inner product 와 같은 수학적인 computation의 시간은 줄여줄 수 있지만,
    그냥 cpu로 돌리는게 더 빠르다. 그리고 그냥 numpy로 돌리는게 더 빠를 것 같다.
    (내 코드가 효율적이지 못할수도 있음.)
"""


# Back Substitution
@jax.jit
def _check_U(mat):
    return ((jnp.triu(mat) != mat).sum() == 0)  # triu라면 tri upper 취한거랑 비교해서 같다.


@functools.partial(jax.jit, static_argnums=(3,))
def _compute_back_xs_value(x, mat, b, idx):
    n, _ = mat.shape
    if idx == n - 1:
        return b[idx] / mat[idx, idx]
    else:
        # mat_prime = jax.lax.dynamic_slice(mat, (idx, idx+1), (idx, n-1))
        mat_prime = mat[idx, idx + 1:]
        x_prime = x[idx + 1:]
        return (b[idx] - jnp.dot(mat_prime, x_prime)) / mat[idx, idx]

# Back Substitution
@jax.jit
def _check_L(mat):
    return ((jnp.tril(mat) != mat).sum() == 0)  # triu라면 tri upper 취한거랑 비교해서 같다.


@functools.partial(jax.jit, static_argnums=(3,))
def _compute_forward_xs_value(x, mat, b, idx):
    n, _ = mat.shape
    if idx == 0:
        return b[idx] / mat[idx, idx]
    else:
        # mat_prime = jax.lax.dynamic_slice(mat, (idx, idx+1), (idx, n-1))
        mat_prime = mat[idx, :idx]
        x_prime = x[:idx]
        return (b[idx] - jnp.dot(mat_prime, x_prime)) / mat[idx, idx]



"""
    fori_loop으로 control flow 돌릴떄 내부 function에 indexing하는 부분이 있는 경우.
    idx가 shapedarray로 정의된다. -> static argnums로 바뀌게 되는데
    이런 경우에는 그냥 python for 문으로 작성해야한다.
"""


# from jax.experimental.host_callback import call
# call ("") function으로 shaped array value를 Print 할 수 있다.

@functools.partial(jax.jit, static_argnums=(0,))
def _back_sub_func(i, params):  # params : mat, b
    lxs, mat, b = params
    n = lxs.shape[0]

    comp_value = _compute_back_xs_value(lxs, mat, b, i)
    lxs = lxs.at[i].set(comp_value)

    return (lxs, mat, b)

@functools.partial(jax.jit, static_argnums=(0,))
def _forward_sub_func(i, params):  # params : mat, b
    lxs, mat, b = params
    n = lxs.shape[0]

    comp_value = _compute_forward_xs_value(lxs, mat, b, i)
    lxs = lxs.at[i].set(comp_value)

    return (lxs, mat, b)


def back_sub(mat, b):
    n, m = mat.shape
    assert n == m, "Please Input Square Matrix"
    assert _check_U(mat), "Please Input Upper Triangular Matrix"

    xs = jnp.ones((n,), jnp.float32)

    val = (xs, mat, b)

    for i in range(n - 1, -1, -1):
        val = _back_sub_func(i, val)

    xs, _, _ = val
    return xs


def forward_sub(mat, b):
    n, m = mat.shape
    assert n == m, "Please Input Square Matrix"
    assert _check_L(mat), "Please Input Lower Triangular Matrix"

    xs = jnp.ones((n,), jnp.float32)

    val = (xs, mat, b)

    for i in range(n):
        val = _forward_sub_func(i, val)

    xs, _, _ = val
    return xs


def _cal_LU(L, U, mat):
    n, _ = L.shape
    for i in range(1, n):
        for j in range(1, i + 1):
            L[i, j] = mat[i, j] - np.dot(L[i, :j], U[:j, j])

        for j in range(i + 1, n):
            U[i, j] = (mat[i, j] - np.dot(L[i, :j], U[:j, j])) / L[i, i]
    return L, U