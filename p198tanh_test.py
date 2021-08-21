import matplotlib.pyplot as plt
import numpy as np

# データ生成
x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y1 = np.sinh(x) 
# y2 = np.cosh(x) 
y3 = np.tanh(x) 

# プロット
# plt.plot(x, y1, label="sinh")
# plt.plot(x, y2, label="cosh")
plt.plot(x, y3, label="tanh")

# 凡例の表示
plt.legend()

# プロット表示(設定の反映)
plt.show()