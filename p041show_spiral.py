import numpy as np
import sys
sys.path.append('.\sample')
from dataset import spiral
import matplotlib.pyplot as plt
x, t = spiral.load_data()
# print(x)
# print(t)
for i in range(len(x)):
    if t[i][0] == 1:
        c = 'red'
    elif t[i][1] == 1:
        c = 'blue'
    else:
        c = 'green'
    plt.scatter(x[i][0], x[i][1], color=c)
plt.show()