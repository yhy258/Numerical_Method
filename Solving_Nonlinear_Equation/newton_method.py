from nonlinear_case import OpenMethod

import math

import jax.numpy as jnp
from jax import grad
import jax; jax.config.update("jax_platform_name", 'cpu')


class NewtonMethod(OpenMethod):
    def __init__(self, func, fd_func):
        super().__init__()
        self.func = func
        self.fd_func = fd_func

    def _update_func(self, x):
        return -(self.func(x)/self.fd_func(x))

    def newtonmethod(self, iteration, x, stop_error=0.001):
        self._history_save(x)
        for i in range(iteration):
            new_x = x + self._update_func(x)
            x = new_x
            self._history_save(x)

            if self._estimate_error() < stop_error :
                break
        return self.history[-1], self.func(self.history[-1])


class JaxNewtonMethod(OpenMethod):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self.derivative_fn = grad(func)

    def _update_func(self, x):
        inv_part = jnp.linalg.pinv(self.derivative_fn(x).reshape(-1, 1))[0]

        return -(self.func(x).block_until_ready() * inv_part)

    def newtonmethod(self, iteration, x, stop_error=0.001):
        self._history_save(x)
        for i in range(iteration):
            new_x = x + self._update_func(x)

            self._history_save(new_x)
            x = new_x
            # estimate error가 Vector form이라 고려필요.
            # if self._estimate_error() < stop_error :
            #     break

        return self.history[-1], self.func(self.history[-1])
if __name__ == '__main__':
    def nonlinear_function(x):
        return 8 - 4.5*(x - math.sin(x))
    def fd_nonlinear_function(x):
        return -4.5*(1 - math.cos(x))

    num_solver = NewtonMethod(nonlinear_function, fd_nonlinear_function)
    print(num_solver.newtonmethod(20, 2, 0.0001))


    # def nonlinear_function(x):
    #     return 3 * jnp.dot(x, x.T)  # 만들어질때 jnp vector는 row vector이다.
    #
    #
    # num_solver = JaxNewtonMethod(nonlinear_function)
    # x_small = jnp.arange(10.)
    # print(num_solver.newtonmethod(10, x_small, 0.0001))