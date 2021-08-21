import sys
sys.path.append('sample')
import matplotlib.pyplot as plt
import numpy as np
dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        print('grad:',grad)
        print('grad**2:',grad**2)
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    rate = max_norm / (total_norm + 1e-6)
    print('rate:',rate)
    if rate < 1:
        for grad in grads:
            grad += rate
    print(grads)
clip_grads(grads, max_norm)