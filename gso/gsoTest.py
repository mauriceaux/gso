#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:09:41 2019

@author: mauri
"""

from GSO import GSO
from Problem import Problem
from ProblemTest import ProblemTest
import numpy as np
import multiprocessing as mp
from scipy.cluster.vq import kmeans,vq
import read_instance as r_instance
from datetime import datetime


gso = GSO()
gso.min = -5
gso.max = 5

gso.minVel = -1
gso.maxVel= 1

gso.accel = 3
#gso.min = -500
#gso.max = 500
#problem = ProblemTest()

problem = Problem("instances/mscp41.txt")
#problem = Problem("instances/mscpnrh5.txt")
gso.setEvalEnc(problem.evalEnc)
#
#gso.UNIVERSE = gso.genRandomSwarm(50, problem.get_columns())
gso.UNIVERSE = gso.genRandomSwarm(50, problem.instance.get_columns())

gso.LEVELS = 2
gso.numIter = [50,250]
gso.numSubSwarms = [10]


#gso.LEVELS = 3
#gso.numIter = [50,250,50]
#gso.numSubSwarms = [10,3]


#gso.UNIVERSE = gso.genRandomSwarm(500, problem.instance.get_columns())
#gso.LEVELS = 3
#gso.numIter = [10,30, 40]
#gso.numSubSwarms = [100,10]

EPOCHS = 1
start = datetime.now()
print(f'START {start.strftime("%H:%M:%S")}')
gso.optimize(maximize=False, epochs = EPOCHS)
end = datetime.now()



print(f'best particle {gso.bestParticle}')
print(f'best bin {gso.bestParticleBin}')
print(f'best obj {gso.globalBest}')

def func1(pos,costo):
#    print(f'pos {pos} \ncosto {costo}')
    print(f'funcion obj {np.sum(np.array(pos) * np.array(costo))}')
    return np.sum(np.array(pos) * np.array(costo))

#err = func1(gso.bestParticleBin, problem.instance.get_c())
#print(f'best particle sum {np.sum(gso.bestParticle)}')
#print(f'best particle sum sin {np.sin(np.sum(gso.bestParticle))}')
#
print(f'END   {end.strftime("%H:%M:%S")}')
print(f'TOTAL {(end-start)}')