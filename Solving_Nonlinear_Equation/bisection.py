from nonlinear_case import BracketingMethod

import math
# bisec func 의 image는 상수.

class BIS_Num(BracketingMethod) :
    def __init__(self, bisec_func, a, b):
        super().__init__()
        self.bisec_func = bisec_func
        self.a = a
        self.b = b


    def _condition_test(self, a, b):
        return self.bisec_func(a) * self.bisec_func(b) < 0

    def bisection(self, iteration, stop_error=0.001):
        assert self._condition_test(self.a, self.b), "Boundary Condition 위배"

        for i in range(iteration):
            new_x = (self.a + self.b) / 2
            if self._condition_test(new_x, self.b) :
                self.a = new_x
            else :
                self.b = new_x

            self._history_save(new_x)
            if self._estimate_error() < stop_error:
                break

        return new_x, self.bisec_func(new_x), self.history

if __name__ == '__main__':
    def nonlinear_function(x):
        return 8 - 4.5*(x - math.sin(x))

    num_solver = BIS_Num(nonlinear_function, 2, 3)
    sol, sol_func, _ = num_solver.bisection(20)
    print(sol, sol_func)

