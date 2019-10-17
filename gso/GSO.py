#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:41:52 2019

@author: mauri
"""
import numpy as np
import multiprocessing as mp
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import MinMaxScaler

class GSO:
    def __init__(self):
        self.TOT_ITER = 10
        self.NUM_ITER = 0
        self.swarmSize = 50
        self.featureSize = 2000
        self.UNIVERSE = self.genRandomSwarm(self.swarmSize, self.featureSize)
        self.numIter = [10,20,30]
        self.numSubSwarms = [10,5]
        self.globalBest = None
        self.bestParticle = None
        self.bestParticleBin = None
        self.LEVELS = 3
        self.scaler = MinMaxScaler(feature_range=(-5,5))
        np.random.seed(0)
            
    def setEvalEnc(self, evalEnc):
        self.evalEnc = evalEnc
        
    def moveSwarm(self, swarm, velocity, personalBest, bestFound):
        self.NUM_ITER += 1
        self.accelPer  = 2.05 * np.random.uniform(size=(swarm.shape))
        self.randPer   = np.random.uniform(size=(swarm.shape))
        self.accelBest = 2.05 * np.random.uniform(size=(swarm.shape))
        self.randBest  = np.random.uniform(size=(swarm.shape))
        
        personalDif = personalBest - swarm
        personalAccel = self.accelPer * self.randPer * personalDif
        bestDif = bestFound - swarm
        bestAccel = self.accelBest * self.randBest * bestDif
        acceleration =  personalAccel + bestAccel
#        iW = 1 - (self.NUM_ITER/(1+self.TOT_ITER))
        iW = 0.5
        nextVel = (iW*velocity) + acceleration
        ret = swarm+nextVel
        ret[ret > 5]  = 5
        ret[ret < -5] = -5
#        shape = ret.shape
#        ret = np.reshape(ret, (1,np.prod(ret.shape)))
#        ret = self.scaler.fit_transform(ret)
#        ret = np.reshape(ret, shape)
        return ret, nextVel

    def updateSwarmData(self, swarmData, iterations):
        swarm         = np.vstack(np.array(swarmData)[:,0])
        velocity      = np.vstack(np.array(swarmData)[:,1])
        personalBest  = np.vstack(np.array(swarmData)[:,2])
        evals         = np.array(swarmData)[:,3]
        bestFound     = np.vstack(np.array(swarmData)[:,4])
        bestEval      = np.array(swarmData)[0,5]
#        print(f'bestEval {bestEval}')
#        exit()
        for i in range(iterations):
            nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, bestFound)
            args = nswarm
            
            pool = mp.Pool(8)
#            evaluations, binParticle = pool.map(self.evalEnc, list(args))
            returning = pool.map(self.evalEnc, list(args))
            returning = np.array(returning)
            binParticle = returning[:,1]
            evaluations = returning[:,0]
            pool.close()
#            print(np.array(returning).shape)
#            exit()
#            evaluations = []
#            binParticle = []
#            for s in args:
#                fitness, decoded = self.evalEnc(s)
#                evaluations.append(fitness)
#                binParticle.append(decoded)
                
            


            evaluations = np.array(np.vstack(evaluations))
#            print(evaluations.shape)
#            exit()
            
            self.globalBest is None
        
            evaluations = evaluations.reshape((swarm.shape[0]))
            bestidx = evaluations > evals
            for j in range(bestidx.shape[0]):
                if bestidx[j]:
                    personalBest[j] = nswarm[j]
                    
                    if self.globalBest is None or evaluations[j] > self.globalBest:
#                    if evaluations[j] > bestEval:
#                        print(f'{evaluations[j]} > {bestEval} {evaluations[j] > bestEval}')
                        bestFound = np.tile(nswarm[j], (nswarm.shape[0],1))
#                        print(f'bestFound {bestFound.shape}')
#                        exit()
                        bestEval = evaluations[j]
                        self.globalBest = evaluations[j]
                        self.bestParticle = bestFound[0]
                        self.bestParticleBin = binParticle[j]
                        print(f'best obj {self.globalBest}')
            swarm = nswarm
        ret = []
        for i in range(swarm.shape[0]):
            ret.append([swarm[i], velocity[i], personalBest[i], evaluations[i], bestFound[i], bestEval])
        return np.array(ret)
    
    
    def genSubSwarms(self, universe, currLevel, levels, swarmsPerLevel):
        centroids = None
        idx = None
        if currLevel < len(swarmsPerLevel) and len(universe[0]) > swarmsPerLevel[currLevel]:
            
#            try:
#            print(f'np.array(universe[0]).shape {np.array(universe[0]).shape} level {currLevel}')
            centroids,_ = kmeans(universe[0],swarmsPerLevel[currLevel])
            
            idx,_ = vq(universe[0],centroids)
        ret = {}
        for i in range(len(universe[0])):
            if currLevel >= len(swarmsPerLevel) or len(universe[0]) <= swarmsPerLevel[currLevel]:
                if 0 not in ret.keys(): ret[0] = []
                ret[0].append([universe[0][i]
                    , universe[1][i]
                    , universe[2][i]
                    , universe[3][i]
                    , universe[4]
                    , universe[5]])
                
            else:
                if idx[i] not in ret.keys(): ret[idx[i]] = []
                ret[idx[i]].append([universe[0][idx[i]]
                    , universe[1][idx[i]]
                    , universe[2][idx[i]]
                    , universe[3][idx[i]]
                    , universe[4]
                    , universe[5]])
        return ret
    
    
    def genRandomSwarm(self, swarmSize = 50, featureSize = 2000):    
        swarm =        np.random.uniform(low=-5, high=5, size=(swarmSize, featureSize))
        velocity =     np.random.uniform(size=(swarmSize, featureSize))
        personalBest = np.random.uniform(low=-5, high=5, size=(swarmSize, featureSize))
        bestFound =    np.random.uniform(low=-5, high=5, size=(featureSize))
        evals =        np.ones((swarmSize)) * -9999
#        np.random.uniform(size=(swarmSize))
        bestEval =     -9999
        return [swarm, velocity, personalBest, evals, bestFound, bestEval]
    
    def getNextLvlSwarms(self, swarms, globalBest, globalBestEval):
        bestParticles = []
        bestEvals = []
        
        for swarmIdx in swarms.keys():
#            print(f'np.array(swarms[swarmIdx]).shape {np.array(swarms[swarmIdx]).shape}')
            bestParticles.append(swarms[swarmIdx][0][4])
            bestEvals.append(swarms[swarmIdx][0][5])
        nxtSwarm = self.genRandomSwarm(len(bestParticles), swarms[swarmIdx][0][4].shape[0])
        nxtSwarm[0] = bestParticles
        nxtSwarm[3] = bestEvals
        nxtSwarm[4] = globalBest
        nxtSwarm[5] = globalBestEval
        return nxtSwarm
            
    def optimize(self, maximize, epochs):
        
        swarms = []
        universes = []
        universes.append(self.genSubSwarms(self.UNIVERSE, 0, self.LEVELS, self.numSubSwarms))
        
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            for level in range(self.LEVELS):
                swarms = universes[level]
                print(f'LEVEL {level}')        
                for swarmIdx in swarms.keys():
                    swarmData = np.array(swarms[swarmIdx])
                    nextSwarmData = self.updateSwarmData(swarmData, self.numIter[level])
#                    if self.globalBest is None or nextSwarmData[0,5] > self.globalBest:
#                        self.globalBest = nextSwarmData[0,5]
#                        self.bestParticle = nextSwarmData[0,4]
#                        print(f'best obj {self.globalBest}')
                    swarms[swarmIdx] = nextSwarmData
                    if (swarmData[0] == nextSwarmData[0]):
                        print(f'swarms[{swarmIdx}] {swarms[swarmIdx]} \n no actualizado')
                        exit()
                
#                print(self.bestParticle)
                nxtSwarm = self.getNextLvlSwarms(swarms, self.bestParticle, self.globalBest)
                universes.append(self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms))
        