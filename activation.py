import numpy as np


class ReLU:
    def forward(self, X: int):
        vec_max = np.vectorize(max)
        self.output = vec_max(0, X)

    def backward(self, X: int):
        pass
