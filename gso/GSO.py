#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:41:52 2019

@author: mauri
"""
import math
import numpy as np
import multiprocessing as mp
from multiprocessing import Value
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.preprocessing import normalize
from matplotlib import pyplot
#from statsmodels.tsa.seasonal import seasonal_decompose
#import line_profiler
from threading import Lock
lock = Lock()
class GSO:
    def __init__(self, globalBest=None):
#        np.random.seed(0)
        if globalBest is not None:
            global gBest
            gBest = globalBest
        self.evalEnc = None
        self.accel=1
        self.min = -5
        self.max = 5
        self.TOT_ITER = 10
        self.NUM_ITER = 0
        self.swarmSize = 50
        self.featureSize = 2000
        self.numIter = [10,20,30]
        self.numSubSwarms = [10,5]
        self.globalBest = None
        self.bestParticle = None
        self.bestParticleBin = None
        self.LEVELS = 3
        self.accelPer  = 0.3
        self.accelBest = 0.5
        self.randPer   = 2.05 * np.random.uniform()
        self.randBest  = np.random.uniform()
        self.minVel = -1
        self.maxVel = 1
        self.scaler = MinMaxScaler(feature_range=(self.minVel,self.maxVel))
        self.decode = None
        self.repair = None
        self.evalDecoded = None
        self.onlineAdjust = False
        
    def setScaler(self, minVal, maxVal):
        self.scaler = MinMaxScaler(feature_range=(minVal,maxVal))
        
#    @profile
    def setEvalEnc(self, evalEnc):
        self.evalEnc = evalEnc
        
#    @profile
    def moveSwarm(self, swarm, velocity, personalBest, bestFound):        
        personalDif = personalBest - swarm
        personalAccel = self.accelPer * self.randPer * personalDif
        bestDif = bestFound - swarm
        bestAccel = self.accelBest * self.randBest * bestDif
        acceleration =  personalAccel + bestAccel
        iW = self.accel
        nextVel = (iW*velocity) + acceleration
        nextVel[nextVel > self.maxVel]  = self.maxVel
        nextVel[nextVel < self.minVel]  = self.minVel
        ret = swarm+nextVel
        ret[ret > self.max]  = self.max
        ret[ret < self.min] = self.min
        return ret, nextVel


#    @profile
    def updateSwarmData(self, swarmData, iterations, globalBest):
        global gBest
        exitCount = 0
        start = datetime.now()
        swarm         = np.vstack(np.array(swarmData)[:,0])
        velocity      = np.vstack(np.array(swarmData)[:,1])
        personalBest  = np.vstack(np.array(swarmData)[:,2])
        evals         = np.array(swarmData)[:,3]
        bestFound     = np.vstack(np.array(swarmData)[:,4])
        bestEval      = np.array(swarmData)[0,5]
        bestParticleBin = np.vstack(np.array(swarmData)[:,6])
        evaluationsCsv = []
        for i in range(iterations):
            if i >0 and i % 4 == 0 and self.onlineAdjust:
                
                self.updateAccelParams(evaluationsCsv[-4:])
            startIter = datetime.now()
            if len(swarm) == 1:
                nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, gBest.value)
            else:
                nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, bestFound)
            returning = [self.evalEnc(item) for item in list(nswarm)]
            binParticle = [item[1] for item in returning]
            binParticle = np.vstack(binParticle)
            evaluations = [item[0] for item in returning]
            evaluations = np.array(np.vstack(evaluations))
            evaluations = evaluations.reshape((swarm.shape[0]))
            evaluationsCsv.append(evaluations)
            bestidx = evaluations > evals
            
            personalBest[bestidx] = nswarm[bestidx]
            idx = np.argmax(evaluations)
            
            if bestEval is None or (evaluations[idx] > bestEval):
                bestEval =  evaluations[idx]
                with gBest.get_lock():
                    gBest.value = evaluations[idx]
                bestFound = np.tile(nswarm[idx], (nswarm.shape[0],1))
                bestParticleBin = binParticle[idx]
            else:
                exitCount += 1
            evals = evaluations
            swarm = nswarm
            endIter = datetime.now()
            print(f'best obj {bestEval} duracion iteracion {i} {endIter-startIter}')
            if exitCount >= 10:
                break
#            
        ret = []
        
        for i in range(swarm.shape[0]):
            ret.append([swarm[i], velocity[i], personalBest[i], evaluations[i], bestFound[i], bestEval])
        end = datetime.now()
        print(f'tiempo actualizacion enjambre {end-start}')
        return np.array(ret), evaluationsCsv, bestParticleBin
    
    
#    @profile
    def genSubSwarms(self, universe, currLevel, levels, swarmsPerLevel):
        centroids = None
        idx = None
        if currLevel < len(swarmsPerLevel) and len(universe[0]) > swarmsPerLevel[currLevel]:
            centroids,_ = kmeans(np.array(universe[0], dtype=np.float),swarmsPerLevel[currLevel])
            
            idx,_ = vq(universe[0],centroids)
        ret = {}
        for i in range(len(universe[0])):
            if currLevel >= len(swarmsPerLevel) or len(universe[0]) <= swarmsPerLevel[currLevel]:
                if i not in ret.keys(): ret[i] = []
                ret[i].append([universe[0][i]
                    , universe[1][i]
                    , universe[2][i]
                    , universe[3][i]
                    , universe[4]
                    , universe[5]
                    , universe[6]])
                
            else:
                if idx[i] not in ret.keys(): ret[idx[i]] = []
                ret[idx[i]].append([universe[0][idx[i]]
                    , universe[1][idx[i]]
                    , universe[2][idx[i]]
                    , universe[3][idx[i]]
                    , universe[4]
                    , universe[5]
                    , universe[6]])
        return ret
    
#    @profile
    def updateAccelParams(self, intervalo):
#        intervalo = evaluationsCsv[-10:]
        grupoInicio = intervalo[:2]
        grupoFin = intervalo[-2:]
        dif = np.mean(grupoFin) - np.mean(grupoInicio)
        if dif < 100:
            self.accelBest = np.random.uniform(low=-3,high=3)
            self.accelPer = np.random.uniform(low=-3,high=3)
            self.accel = np.random.uniform(low=-3,high=3)
            
#        print(f'update params {grupoFin-grupoInicio}')
#        exit()
        
    
#    @profile
    def genRandomSwarm(self, swarmSize = 50, featureSize = 2000):    
        swarm =        np.random.uniform(low=self.min, high=self.max, size=(swarmSize, featureSize))
        velocity =     np.random.uniform(size=(swarmSize, featureSize))
        personalBest = np.zeros((swarmSize, featureSize))
        returning = [self.evalEnc(item) for item in list(personalBest)]
        personalBest = [item[1] for item in returning]
        
        personalBest = np.vstack(personalBest)
        bestFound = personalBest[0]
        evals =[item[0] for item in returning]
        bestEval = returning[0][0]
        personalBestBin = np.zeros((swarmSize, featureSize))
        return [swarm, velocity, personalBest, evals, bestFound, bestEval, personalBestBin]
    
#    @profile
    def getNextLvlSwarms(self, swarms, globalBest, globalBestEval):
        bestParticles = []
        bestEvals = []
        for i in range(len(swarms)):
            bestParticles.append(swarms[i][0][4])
            bestEvals.append(swarms[i][0][5])
        nxtSwarm = self.genRandomSwarm(len(bestParticles), swarms[0][0][4].shape[0])
        nxtSwarm[0] = bestParticles
        nxtSwarm[3] = bestEvals
        nxtSwarm[4] = globalBest
        nxtSwarm[5] = globalBestEval
        return nxtSwarm
            
#    @profile
    def optimize(self, maximize, epochs):
        swarms = []
        universes = []
        universes.append(self.genSubSwarms(self.UNIVERSE, 0, self.LEVELS, self.numSubSwarms))
        for level in range (self.LEVELS):
            if level >= len(self.numSubSwarms):
                np.savetxt(f"resultados/swarmL{level}S{0}.csv", np.array([]), delimiter=",")
            else:
                for i in range (self.numSubSwarms[level]):
                    np.savetxt(f"resultados/swarmL{level}S{i}.csv", np.array([]), delimiter=",")
        gBest = Value('f', -math.pow(10,6))
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            for level in range(self.LEVELS):
                swarms = universes[level]
                print(f'LEVEL {level}')        
                swarmsList = [item for item in swarms.values()]  
                startSwarm = datetime.now()
                args = [list((swarm, self.numIter[level], swarm[0][5])) for swarm in swarmsList]
                
                pool = mp.Pool(8, initializer = self.__init__,  initargs = (gBest, ))
                ret = pool.starmap(self.updateSwarmData, args)
                pool.close()
                endSwarm = datetime.now()
                print(f'Optimization for level {level} completed in: {endSwarm - startSwarm}')
                swarms = [data[0] for data in ret]
                evals = [data[1] for data in ret]
                for i in range(len(swarms)):
                    print(f'global best {np.array(swarms[i][0][5])} swarm {i} level {level}')
                    if self.globalBest is None or swarms[i][0][5] > self.globalBest:
                        self.globalBest = swarms[i][0][5]
                        self.bestParticle = swarms[i][0][4]
                        self.bestParticleBin = ret[i][2]
                    with open(f"resultados/swarmL{level}S{i}.csv", "ab") as f:
                        np.savetxt(f, np.array(evals[i]), delimiter=",")
                    nxtSwarm = self.getNextLvlSwarms(swarms, self.bestParticle, self.globalBest)
                if level+1 >= len(universes):
                    universes.append(self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms))
                else:
                    universes[level+1] = self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms)
                
