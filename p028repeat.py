import numpy as np
D, N = 8,7
x = np.random.randn(1, D)
y = np.repeat(x, N, axis=0)

dy = np.random.randn(N, D)
dx = np.sum(dy, axis=0, keepdims=True)
print('x:', x)
print('y:', y)
print('dy:', dy)
print('dx:', dx)