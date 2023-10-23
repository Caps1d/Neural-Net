import numpy as np

from activation import ReLU


np.random.seed(0)


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        # n_inputs x n_neurons = T(n_neurons x n_inputs)
        # Hence we take care of transpose for the dot product
        # 0.10 * to ensure we're between 0.1 and -.1
        self.size = n_neurons
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: list):
        self.output = np.dot(inputs, self.weights) + self.biases


def main():
    X = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]
    layer1 = LayerDense(len(X[0]), 5)
    layer2 = LayerDense(layer1.size, 2)
    layer3 = LayerDense(layer2.size, 2)

    activation = ReLU()

    layer1.forward(X)
    activation.forward(layer1.output)

    layer2.forward(activation.output)
    activation.forward(layer2.output)
    print(activation.output)

    layer3.forward(activation.output)
    activation.forward(layer3.output)

    print(activation.output)
    pass


if __name__ == "__main__":
    main()
