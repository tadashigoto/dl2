import numpy as np
from p036sigmoid import Sigmoid
// from p039
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        W1 = 0.01 * np.random.randn(I,H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params = []
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

