#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:41:52 2019

@author: mauri
"""
import numpy as np
import multiprocessing as mp
from scipy.cluster.vq import kmeans,vq

class GSO:
    def __init__(self):
        self.TOT_ITER = 10
        self.NUM_ITER = 0
#        self.accelBest = np.random.uniform(size=(50,2000))
#        self.accelPer = np.random.uniform(size=(50,2000))
##        self.accelBest = np.ones((50,2000))
##        self.accelPer = np.ones((50,2000))
#        self.randPer = np.random.uniform(size=(50,2000))
#        self.randBest = np.random.uniform(size=(50,2000))
#        self.evalEnc = None
#        for level in range(levelNum):
#            pass
        self.swarmSize = 50
        self.featureSize = 2000
        self.UNIVERSE = self.genRandomSwarm(self.swarmSize, self.featureSize)
        self.numIter = [10,20,30]
        self.numSubSwarms = [10,5]
        self.globalBest = None
        self.LEVELS = 3
        
    def setEvalEnc(self, evalEnc):
        self.evalEnc = evalEnc
        
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
        iW = 1 - (self.NUM_ITER/(1+self.TOT_ITER))
#        iW = 1
        nextVel = (iW*velocity) + acceleration
        return swarm+nextVel, nextVel

    def updateSwarmData(self, swarmData, iterations):
        swarm         = np.vstack(np.array(swarmData)[:,0])
        velocity      = np.vstack(np.array(swarmData)[:,1])
        personalBest  = np.vstack(np.array(swarmData)[:,2])
        evals         = np.array(swarmData)[:,3]
        bestFound     = np.vstack(np.array(swarmData)[:,4])
        bestEval      = np.array(swarmData)[0,5]
#        print(f'bestFound {bestFound.shape}')
#        print(evals)
#        exit()
#        swarmSize = swarm.shape[0]
        for i in range(iterations):
#            print(f'ITER              {i}')
#            for i in range(swarms.shape[0]):
#            swarm = swarms[i]
#            velocity = velocities[i]
#            personalBest = personalBests[i]
#            bestFound = bestsFound[i]
#                bestEval = bestEvals[i]
#            print(f'swarm {swarm.shape}')
            nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, bestFound)
            
#            print(f'nswarm {nswarm.shape}')
            args = nswarm
            pool = mp.Pool()
            evaluations = pool.map(self.evalEnc, list(args))
#                print(f'evaluations {len(evaluations)}')
            pool.close()
#            print(f'evaluations {len(evaluations)}')
#            print(f'evals {evals.shape}')
            evaluations = np.array(np.vstack(evaluations))
            
        
            evaluations = evaluations.reshape((swarm.shape[0]))
#            evals = evals.reshape((swarmSize))
            
#                print(f'evaluations {evaluations.shape}')
            bestidx = evaluations > evals
#            print(f'bestidx {bestidx}')
            for j in range(bestidx.shape[0]):
                if bestidx[j]:
                    personalBest[j] = nswarm[j]
                    if evaluations[j] > bestEval:
#                        print(f'bestFound antes {bestFound.shape}')
#                        print(f'nswarm[{j}] antes {np.tile(nswarm[j], (nswarm.shape[0],1)).shape}')
                        bestFound = np.tile(nswarm[j], (nswarm.shape[0],1))
#                        print(f'bestFound despues {bestFound.shape}')
#                        exit()
                        bestEval = evaluations[j]
#                        bestEval = np.tile(evaluations[j], (evaluations.shape[0], 1))
#            swarms[i] = swarm
            swarm = nswarm
#        print(f'swarm.shape {swarm.shape}')
#        print(f'velocity.shape {velocity.shape}')
#        print(f'personalBest.shape {personalBest.shape}')
#        print(f'evaluations.shape {evaluations.shape}')
#        print(f'bestFound.shape {bestFound.shape}')
#        print(f'bestEval.shape {np.tile(bestEval,(swarm.shape[0],1)).shape}')
#        print(f'np.tile(bestEval,(swarm.shape[0],1)) {np.tile(bestEval,(swarm.shape[0],1))}')
#        print(swarm.shape[0])
        ret = []
        for i in range(swarm.shape[0]):
            ret.append([swarm[i], velocity[i], personalBest[i], evaluations[i], bestFound[i], bestEval])
        return np.array(ret)
    
    
    def genSubSwarms(self, universe, currLevel, levels, swarmsPerLevel):
        centroids = None
        idx = None
        if currLevel < len(swarmsPerLevel) and len(universe[0]) > swarmsPerLevel[currLevel]:
            
#            try:
            print(f'np.array(universe[0]).shape {np.array(universe[0]).shape} level {currLevel}')
            centroids,_ = kmeans(universe[0],swarmsPerLevel[currLevel])
            
            idx,mmm = vq(universe[0],centroids)
#            except:
#                print(f'kmeans(universe[0],swarmsPerLevel[currLevel]) kmeans({len(universe[0])},{swarmsPerLevel[currLevel]})')
#                exit()
    #    print(mmm.shape)
        ret = {}
    #    print(np.array(universe[4]).shape)
    #    exit()
        for i in range(len(universe[0])):
            
            
    #        print(f'current level {currLevel} universe shape {np.array(universe).shape} len(swarmsPerLevel) {len(swarmsPerLevel)}')
    #        exit()
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
        swarm =        np.random.uniform(size=(swarmSize, featureSize))
        velocity =     np.random.uniform(size=(swarmSize, featureSize))
        personalBest = np.random.uniform(size=(swarmSize, featureSize))
        bestFound =    np.random.uniform(size=(featureSize))
        evals =        np.random.uniform(size=(swarmSize))
        bestEval =     np.random.uniform()
    #    print(f'bestEval {bestEval}')
    #    exit()
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
#        print(f'np.array(nxtSwarm).shape {np.array(nxtSwarm).shape}')
#        print(f'len(nxtSwarm[0]) {len(nxtSwarm[0])}')
        return nxtSwarm
            
    def optimize(self, maximize, epochs):
        
        swarms = []
        
        #for i in range(10):
        #    swarms.append(genRandomSwarm())
        
        
        
        universes = []
        #universes.append(UNIVERSE)
        
        #print(len(swarms[3]))
        
    #    globatBestParticle = None
        
        universes.append(self.genSubSwarms(self.UNIVERSE, 0, self.LEVELS, self.numSubSwarms))
        
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            for level in range(self.LEVELS):
                swarms = universes[level]
                print(f'LEVEL {level}')
            
        #        print(swarms.keys())
                
                for swarmIdx in swarms.keys():
                #    print(f'swarmData pre processed {np.array(swarms[swarmIdx]).shape}')
                    swarmData = np.array(swarms[swarmIdx])
                #    print(f'swarms[{swarmIdx}] es {len(swarms[swarmIdx])}')
                #    print(f'swarmData es {swarmData.shape}')
                #    print(f'swarm {np.vstack(swarmData[:,0]).shape}')
                #    print(f'velocity {np.vstack(swarmData[:,1]).shape}')
                #    print(f'personalBest {np.vstack(swarmData[:,2]).shape}')
                #    print(f'evals {np.vstack(swarmData[:,3]).shape}')
                #    print(f'bestFound {np.vstack(swarmData[:,4]).shape}')
                #    print(f'bestEval {np.vstack(swarmData[:,5]).shape}')
                #    exit()
                #    print(f'swarmData {swarmData[3]}')
                    nextSwarmData = self.updateSwarmData(swarmData, self.numIter[level])
                    if self.globalBest is None or nextSwarmData[0,5] > self.globalBest:
                        self.globalBest = nextSwarmData[0,5]
                        self.bestParticle = nextSwarmData[0,4]
                        print(f'best obj {self.globalBest}')
                    swarms[swarmIdx] = nextSwarmData
                #    print(f'best founds {globalBestParticle.shape}')
                #    print(f'nextSwarmData es {np.array(nextSwarmData).shape}')
                    if (swarmData[0] == nextSwarmData[0]):
                        print(f'swarms[{swarmIdx}] {swarms[swarmIdx]} \n no actualizado')
                        exit()
                
                nxtSwarm = self.getNextLvlSwarms(swarms, self.bestParticle, self.globalBest)
                universes.append(self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms))
        #    exit()
        