import numpy
import cupy
import cupyx
a = cupy.zeros((6,), dtype=numpy.float32)
i = cupy.array([1, 0, 1])
v = cupy.array([1., 1., 1.])
cupyx.scatter_add(a, i, v)
print(a)
