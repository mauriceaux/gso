#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:41:52 2019

@author: mauri
"""
import numpy as np
class GSO:
    def __init__(self, levelNum=2, totIter = 10):
        self.TOT_ITER = totIter
        self.NUM_ITER = 0
        self.accelBest = np.random.uniform(size=(50,2000))
        self.accelPer = np.random.uniform(size=(50,2000))
#        self.accelBest = np.ones((50,2000))
#        self.accelPer = np.ones((50,2000))
        self.randPer = np.random.uniform(size=(50,2000))
        self.randBest = np.random.uniform(size=(50,2000))
        for level in range(levelNum):
            pass
        
    def moveSwarm(self, swarm, velocity, personalBest, bestFound):
        self.NUM_ITER += 1
        self.accelPer  = np.random.uniform(size=(swarm.shape))
        self.randPer   = np.random.uniform(size=(swarm.shape))
        self.accelBest = np.random.uniform(size=(swarm.shape))
        self.randBest  = np.random.uniform(size=(swarm.shape))
#        print(bestFound)
        
        personalDif = personalBest - swarm
#        print(personalDif.shape)
        personalAccel = self.accelPer * self.randPer * personalDif
        bestDif = bestFound - swarm
        bestAccel = self.accelBest * self.randBest * bestDif
        acceleration =  personalAccel + bestAccel
#        iW = 1 - (self.NUM_ITER/(1+self.TOT_ITER))
        iW = 1
        nextVel = (iW*velocity) + acceleration
        return swarm+nextVel, nextVel
        