# p099one_hot_codebase.py : 全結合層による変換
import numpy as np
c = np.array([[1,0,0,0,0,0,0]])
W = np.random.randn(7,3)
h = np.dot(c, W)
print(h)