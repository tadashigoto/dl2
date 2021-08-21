# p201impletemnt_rnn.py : Time RNNレイヤの実装
import numpy as np
def set_status(*arg):
    print(arg)
    print(len(arg))
a=np.arange(12).reshape(2,3,-1)
print(a)
print(a[:,0,:])