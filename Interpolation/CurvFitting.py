from typing import TypeVar, Generic, Tuple, Union, Optional
import numpy as np

# One Dimension..
# numpy array type hint
Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    pass


class LinearRegression:
    def __init__(self):
        self.a1 = None
        self.a0 = None

    def train(self, x: Array['N', float], y: Array['N', float]):
        n, ny = len(x), len(y)
        assert n == ny, "Please same number data & target"
        Sx = np.sum(x)
        Sy = np.sum(y)
        Sxy = np.sum(x * y)
        Sxx = np.sum(np.square(x))
        self.a1 = (n * Sxy - Sx * Sy) / (n * Sxx - np.square(Sx))
        self.a0 = (Sxx * Sy - Sxy * Sx) / (n * Sxx - np.square(Sx))

    def test(self, x: Array['N', float]):
        assert self.a1 != None and self.a0 != None, "우선 훈련을 진행해주세요"
        return self.a0 + self.a1 * x

    def print_params(self):
        print(f"Equation__ : y = {self.a0} + {self.a1} * x ")


if __name__ == '__main__':
    x = np.linspace(0, 100, 11)
    y = np.array([0.94, 0.96, 1.0, 1.05, 1.07, 1.09, 1.14, 1.17, 1.21, 1.24, 1.28])

    model = LinearRegression()
    model.train(x, y)
    model.print_params()

