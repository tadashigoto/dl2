# p201impletemnt_rnn.py : Time RNNレイヤの実装
import numpy as np
def set_status(*arg):
    print(arg)
    print(len(arg))
a=[1,2,3]
set_status(a[0])
set_status(a)
set_status(a[0],a[1])
set_status(*a)
