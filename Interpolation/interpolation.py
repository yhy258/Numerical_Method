from typing import TypeVar, Generic, Tuple, Union, Optional
import numpy as np

# numpy array type hint
Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Array(np.ndarray, Generic[Shape, DType]):
    pass

def _x_sort_func(x, y): # x domain에 대해 정렬.
    indices = np.lexsort((y, x))
    return x[indices], y[indices]

class LinearLagrangeInterpolation: # Spline을 위한 LagrangeInterpolation
    def __init__(self, x: Array['N', float], y: Array['N', float]):
        assert len(x) == 2, "2개의 데이터 전용."
        assert len(x) == len(y), "동일한 수의 x, y 데이터를 넣어주세요"
        self.data_num = 2
        self.x = x
        self.y = y

    def _lagrange_func(self, new_x):
        lower = self.x.reshape(-1, 1) - self.x  # n by n
        lower[np.diag_indices_from(lower)] = 1 # zero divide 예방
        factor = (new_x - self.x).reshape(1, -1).repeat(self.data_num, 0) / lower
        factor[np.diag_indices_from(factor)] = 1
        factor = factor.prod(-1)
        return (self.y * factor).sum()


    def play(self, x):
        return self._lagrange_func(x)


class LagrangeInterpolation: # Non pararmeteric
    mode_dict = {'linear' : 2,'quadratic' : 3,'cubic' : 4}
    def __init__(self, x: Array['N', float], y: Array['N', float], mode : str):
        assert mode in ["linear", "quadratic", "cubic"], "please select in [linear, quadratic, cubic]"
        assert len(x) == len(y), "동일한 수의 x, y 데이터를 넣어주세요"
        assert self.mode_dict[mode] == len(x), "모드에 알맞는 개수의 데이터를 넣어주세요"
        self.data_num = self.mode_dict[mode]
        self.x = x
        self.y = y

    def _lagrange_func(self, new_x):
        lower = self.x.reshape(-1, 1) - self.x  # n by n
        lower[np.diag_indices_from(lower)] = 1 # zero divide 예방
        factor = (new_x - self.x).reshape(1, -1).repeat(self.data_num, 0) / lower
        factor[np.diag_indices_from(factor)] = 1
        factor = factor.prod(-1)
        return (self.y * factor).sum()


    def play(self, x):
        return self._lagrange_func(x)

class LinearSplines:
    def __init__(self, x, y): # _x_sort_func으로 미리 x domain 기준 정렬되어있음.
        self.x = x
        self.y = y
        self.interval_nums = len(self.x) - 1
        self.splines = []

    def _get_linear_lagrange_interp(self, x, y):
        return LinearLagrangeInterpolation(x, y)

    def train(self):
        for i in range(self.interval_nums):
            self.splines.append(self._get_linear_lagrange_interp(self.x[i:i+2], self.y[i:i+2]))
        # interval 별로 spline 저장.

    def play(self, x): # 들어온 요소가 어떤 interval에 속하는지.
        interval_idx = np.where(self.x > x)[0][0] - 1
        this_interval = self.splines[interval_idx]
        return this_interval.play(x)

if __name__=="__main__":
    # problem : https://coast.nd.edu/jjwteach/www/www/30125/pdfnotes/lecture3_6v13.pdf
    x = np.array([0.40, 0.50, 0.70, 0.80])
    y = np.array([-0.916291, -0.693147, -0.356675, -0.223144])
    model = LagrangeInterpolation(x[:4], y[:4], mode='cubic')
    print("Lagrange Interpolation Result : ",model.play(0.60))

    # problem : Numerical method book.
    x = np.array([11, 8, 15, 18])
    y = np.array([9, 5, 10, 8])
    x, y = _x_sort_func(x, y)
    model = LinearSplines(x, y)
    model.train()
    print("Linear Spline Result : ",model.play(12.7))