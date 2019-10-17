#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:09:41 2019

@author: mauri
"""

from GSO import GSO
from Problem import Problem
import numpy as np
import multiprocessing as mp
from scipy.cluster.vq import kmeans,vq
import read_instance as r_instance



gso = GSO()
problem = Problem("instances/mscp41.txt")
gso.setEvalEnc(problem.evalEnc)
gso.UNIVERSE = gso.genRandomSwarm(50, problem.instance.get_columns())
gso.LEVELS = 2
gso.numIter = [50,250]
gso.numSubSwarms = [10]

EPOCHS = 3
gso.optimize(maximize=False, epochs = EPOCHS)


print(f'best particle {gso.bestParticle}')

def func1(pos,costo):
    print(f'pos {pos} \ncosto {costo}')
    print(f'funcion obj {np.sum(np.array(pos) * np.array(costo))}')
    return np.sum(np.array(pos) * np.array(costo))

instance = r_instance.Read(f'instances/mscp41.txt')
err = func1(gso.bestParticleBin, instance.get_c())
#print(f'best particle sum {np.sum(gso.bestParticle)}')
#print(f'best particle sum sin {np.sin(np.sum(gso.bestParticle))}')
print(f'best obj {gso.globalBest}')
