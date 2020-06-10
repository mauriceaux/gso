from numba import cuda, float32
import numpy
import math


from problemas.knapsack.knapsack import KP

problema = KP(f'problemas/instances/knapPI_1_500_1000_1')