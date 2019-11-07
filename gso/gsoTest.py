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
import sys


path = []

import os
#directory = 'instances/delProfe/'
#directory = 'instances/'
directory = 'instancesFinal/'
for filename in os.listdir(directory):
    if not filename.endswith(".txt"): continue
#    print(filename)
#    exit()
    path.append(filename)
generalStart = datetime.now()
#path.append('rail2536.txt')
for f in path:
    f = f'{directory}{f}'
    try:
        print(f'PROCESANDO {f}')
        problem = Problem(f)
        
        gso = GSO()
        gso.onlineAdjust = True
        gso.min = -5
        gso.max = 5
        
        gso.minVel = -3
        gso.maxVel= 3
        gso.setScaler(1,10)
        gso.accel = 1
        gso.accelPer  = 0.1
        gso.accelBest = 0.1
        #problem = Problem("instances/off/scpnrh5.txt")
        
        #problem = Problem("instances/mscp41.txt")
        #gso.decode = problem.binarizeMod
        #gso.repair = problem.repara
        #gso.repair = problem.reparaMod
        #gso.encode = problem.encodeInstance
        #gso.evalDecoded = problem.evalInstance
        
        #gso.setEvalEnc(problem.evalEncMod)
        gso.setEvalEnc(problem.evalEnc)
        #print(gso.evalEnc)
        #exit()
        #gso.UNIVERSE = gso.genRandomSwarm(50, problem.get_columns())
        gso.UNIVERSE = gso.genRandomSwarm(50, problem.instance.get_columns())
        
        gso.LEVELS = 2
        gso.numIter = [50,50]
        #gso.numIter = [50,250]
        #gso.numIter = [1,1]
        gso.numSubSwarms = [12]
        
        
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
    except Exception as e:
        print(f'Error al procesar archivo {f}: {e.args}')
    end = datetime.now()
    
    
    
    print(f'best particle {gso.bestParticle}')
    print(f'best bin {gso.bestParticleBin}')
    print(f'best obj {gso.globalBest}')
    np.set_printoptions(threshold=sys.maxsize)
#    np.savetxt(f"resultadosFinal/globalBest{f.replace('/','.')}.csv", np.array([gso.bestParticleBin,end-start,EPOCHS,gso.LEVELS,gso.numIter]), delimiter=",")
    with open(f"resultadosFinal/globalBest{f.replace('/','.')}.csv", "a") as myfile:
        bestBinStr = np.array2string(gso.bestParticleBin, max_line_width=1000000000000000, precision=1, separator=',', suppress_small=False)
        numIterStr = np.array2string(np.array(gso.numIter), max_line_width=1000000000000000, precision=1, separator=",", suppress_small=False)
        myfile.write(f'{gso.globalBest},{bestBinStr},{start},{end},{end-start},{EPOCHS},{gso.LEVELS},{numIterStr}\n')
#    def func1(pos,costo):
#    #    print(f'pos {pos} \ncosto {costo}')
#        print(f'funcion obj {np.sum(np.array(pos) * np.array(costo))}')
#        return np.sum(np.array(pos) * np.array(costo))
    
    #err = func1(gso.bestParticleBin, problem.instance.get_c())
    #print(f'best particle sum {np.sum(gso.bestParticle)}')
    #print(f'best particle sum sin {np.sin(np.sum(gso.bestParticle))}')
    #
    print(f'END   {end.strftime("%H:%M:%S")}')
    print(f'TOTAL {(end-start)}')
    
generalStop = datetime.now()
print(f'FIN, DEMORO {generalStop - generalStart}')