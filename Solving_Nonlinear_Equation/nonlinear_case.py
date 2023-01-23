class BracketingMethod(): # 구간을 정해서 numerical method를 풀어나가는.
    def __init__(self):
        self.history = {'a': [], 'b': [], 'new_x': []}

    def _condition_test(self, a, b):
        pass

    def _estimate_error(self): # Tolerance
        return abs((self.b - self.a)/2.)

    def _history_save(self, x):
        self.history['a'] += [self.a]
        self.history['b'] += [self.b]
        self.history['new_x'] += [x]

class OpenMethod():
    def __init__(self):
        self.history = [] # zero finding 대상인 domain value
        pass

    def _update_func(self):
        pass

    def _estimate_error(self):
        return abs((self.history[-1] - self.history[-2]) / self.history[-2])

    def _history_save(self, x):
        self.history.append(x)