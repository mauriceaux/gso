from numba import cuda
import numpy
import math

@cuda.jit('void(int32[:], int32[:], int32[:])')
def use_bar(aryA, aryB, aryOut):
    i = cuda.grid(1) # global position of the thread for a 1D grid.
    aryOut[i] = bar(aryA[i], aryB[i])

@cuda.jit('int32(int32, int32)', device=True)
def bar(a, b):
    return a+b+10

a = numpy.ones(256)
b = numpy.ones(256)
c = numpy.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(a.shape[0] / threadsperblock)
use_bar[blockspergrid, threadsperblock](a,b,c)
print(c)