#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:09:41 2019

@author: mauri
"""

from GSO import GSO
from KPProblem.KPProblem import Problem
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
directory = 'KPProblem/instances/'
#directory = 'instanciasReducidas/'
for filename in os.listdir(directory):
    if not filename.endswith(".txt"): continue
#    print(filename)
#    exit()
#    path.append(filename)
    
path.append('knapPI_1_100_1000_1')
generalStart = datetime.now()

for iteracion in range(1):
    for f in path:
        nombreArchivo = f + ""
        f = f'{directory}{f}'
        try:
            print(f'PROCESANDO {f}')
            problem = Problem(f)
            
            gso = GSO()
    #        gso.onlineAdjust = True
            gso.min = -3
            gso.max = 3
            
            gso.minVel = -3
            gso.maxVel= 3
            gso.setScaler(1,10)
            gso.accel = 1
            gso.accelPer  = 2.05 * np.random.uniform()
            gso.accelBest = 2.05 * np.random.uniform()
    #        gso.accelPer  = 0.5
    #        gso.accelBest = 0.3
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
            gso.UNIVERSE = gso.genRandomSwarm(50, problem.instance.numItems)
#                print(gso.UNIVERSE[0])
#                exit()
            
            gso.LEVELS = 2
            gso.numIter = [50,250]
    #        gso.numIter = [150,150]
            #gso.numIter = [1,1]
            gso.numSubSwarms = [12]
            
            
            #gso.LEVELS = 3
            #gso.numIter = [50,250,50]
            #gso.numSubSwarms = [10,3]
            
            
            #gso.UNIVERSE = gso.genRandomSwarm(500, problem.instance.get_columns())
            #gso.LEVELS = 3
            #gso.numIter = [10,30, 40]
            #gso.numSubSwarms = [100,10]
            
            EPOCHS = 3
            start = datetime.now()
            print(f'START {start.strftime("%H:%M:%S")}')

            gso.optimize(maximize=False, epochs = EPOCHS)
        except Exception as e:
            print(f'Error al procesar archivo {f}: {e.args}')
            raise e
        end = datetime.now()
        
        
        
        print(f'best particle {gso.bestParticle}')
        print(f'best bin {gso.bestParticleBin}')
        print(f'best obj {gso.globalBest}')
        np.set_printoptions(threshold=sys.maxsize)
    #    np.savetxt(f"resultadosFinal/globalBest{f.replace('/','.')}.csv", np.array([gso.bestParticleBin,end-start,EPOCHS,gso.LEVELS,gso.numIter]), delimiter=",")
        
        
        with open(f"resultadosFinalKP/globalBest{f.replace('/','.')}.csv", "a") as myfile:
            bestBinStr = np.array2string(gso.bestParticleBin, max_line_width=1000000000000000, precision=1, separator=',', suppress_small=False)
            numIterStr = np.array2string(np.array(gso.numIter), max_line_width=1000000000000000, precision=1, separator=",", suppress_small=False)
            myfile.write(f'{gso.globalBest},{bestBinStr},{start},{end},{end-start},{EPOCHS},{gso.LEVELS},{numIterStr}\n')
        
        with open(f"mejoresResultados/{end.strftime('%H%M%S')}.{nombreArchivo}", "w") as myfile:
            myfile.write(gso.mejoresResultados)
            
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
        del gso
        del bestBinStr
        del numIterStr
    
    
generalStop = datetime.now()
print(f'FIN, DEMORO {generalStop - generalStart}')