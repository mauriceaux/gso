#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:41:52 2019

@author: mauri
"""
import math
import numpy as np
import multiprocessing as mp
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
#import line_profiler
from threading import Lock
lock = Lock()
class GSO:
    def __init__(self):
        self.accel=1
        self.min = -5
        self.max = 5
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
        
        np.random.seed(0)
#        self.accelPer  = 2.05 * np.random.uniform()
#        self.accelBest = 2.05 * np.random.uniform()
        self.accelPer  = 0.1
        self.accelBest = 0.1
        
        self.randPer   = 2.05 * np.random.uniform()
        self.randBest  = np.random.uniform()
        self.minVel = -1
        self.maxVel = 1
        self.scaler = MinMaxScaler(feature_range=(self.minVel,self.maxVel))
        self.decode = None
        self.repair = None
        self.evalDecoded = None
    def setScaler(self, minVal, maxVal):
        self.scaler = MinMaxScaler(feature_range=(minVal,maxVal))
#    @profile
    def setEvalEnc(self, evalEnc):
        self.evalEnc = evalEnc
        
#    @profile
    def moveSwarm(self, swarm, velocity, personalBest, bestFound):
#        self.accelBest = 2.05 * np.random.uniform()
#        self.randPer   = 2.05 * np.random.uniform()
#        self.NUM_ITER += 1
#        accelPer = self.accelPer  * np.ones((swarm.shape))
#        accelBest = self.accelBest * np.ones((swarm.shape))
#        randPer  = self.randPer
#        randBest = self.randBest
        
#        randPer  = self.randPer   * np.ones((swarm.shape))
#        randBest = self.randBest  * np.ones((swarm.shape))
        
#        accelPer = self.accelPer  * np.ones((swarm.shape))
#        accelBest = self.accelBest * np.ones((swarm.shape))
        accelPer = self.accelPer  
        accelBest = self.accelBest 
        
        #randPer = np.random.uniform(low=-1, high=1)
        #randBest = np.random.uniform(low=-1, high=1)
        randPer = 1
        randBest = 1
        
        personalDif = personalBest - swarm
        personalAccel = accelPer * randPer * personalDif
        bestDif = bestFound - swarm
        bestAccel = accelBest * randBest * bestDif
        acceleration =  personalAccel + bestAccel
#        iW = 1 - (self.NUM_ITER/(1+self.TOT_ITER))
        iW = self.accel
        nextVel = (iW*velocity) + acceleration
        
#        print(ret)
#        print(f'nextVel inicio {nextVel}')
#        shape = nextVel.shape
#        nextVel = np.reshape(nextVel, (1,np.prod(nextVel.shape)))
#        nextVel = self.scaler.fit_transform(nextVel)
#        nextVel = np.reshape(nextVel, shape)
        nextVel[nextVel > self.maxVel]  = self.maxVel
        nextVel[nextVel < self.minVel]  = self.minVel
        ret = swarm+nextVel
#        print(f'nextVel fin {nextVel}')
        
        ret[ret > self.max]  = self.max
        ret[ret < self.min] = self.min
        
#        ret[ret > 5]  = 0.8
#        ret[ret < -5] = 0.2
        
#        exit()
#        shape = ret.shape
#        ret = np.reshape(ret, (1,np.prod(ret.shape)))
#        ret = self.scaler.fit_transform(ret)
#        ret = np.reshape(ret, shape)
        return ret, nextVel


#    @profile
    def updateSwarmData(self, swarmData, iterations, globalBest):
        start = datetime.now()
#        print(f'swarmData {swarmData} iterations {iterations}')
        swarm         = np.vstack(np.array(swarmData)[:,0])
        velocity      = np.vstack(np.array(swarmData)[:,1])
        personalBest  = np.vstack(np.array(swarmData)[:,2])
        evals         = np.array(swarmData)[:,3]
        bestFound     = np.vstack(np.array(swarmData)[:,4])
        bestEval      = np.array(swarmData)[0,5]
        evaluationsCsv = []
#        print(f'updateSwarmData bestFound.shape {np.array(bestFound).shape}')
#        print(f'updateSwarmData personalBest.shape {np.array(personalBest)}')
        for i in range(iterations):
            startIter = datetime.now()
            nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, bestFound)
#            pool = mp.Pool(4)
#            returning = pool.map(self.evalEnc, list(nswarm))
#            pool.close()
            returning = [self.evalEnc(item) for item in list(nswarm)]
            binParticle = [item[1] for item in returning]
            binParticle = np.vstack(binParticle)
#            nswarm = np.copy(binParticle)
            evaluations = [item[0] for item in returning]
            evaluations = np.array(np.vstack(evaluations))
            evaluations = evaluations.reshape((swarm.shape[0]))
            evaluationsCsv.append(evaluations)
            bestidx = evaluations > evals
            
            personalBest[bestidx] = nswarm[bestidx]
            lock.acquire()
            bestEval = self.globalBest
            lock.release()
            idx = np.argmax(evaluations)
            with lock:
                if self.globalBest is None or evaluations[idx] > self.globalBest:
                
                    bestEval =  evaluations[idx]
                    self.globalBest = evaluations[idx]
                    bestFound = np.tile(nswarm[idx], (nswarm.shape[0],1))
                    self.bestParticle = bestFound[0]
                    self.bestParticleBin = binParticle[idx]
                    print(f'best obj {bestEval} iter {i}')
#                lock.release()
            evals = evaluations
            swarm = nswarm
            
        ret = []
        
        for i in range(swarm.shape[0]):
            ret.append([swarm[i], velocity[i], personalBest[i], evaluations[i], bestFound[i], bestEval])
#        print(np.array(particles).shape)
        
#        np.savetxt(f"resultados/swarmMovementL{level}S{swarmIdx}.csv", np.array(particles), delimiter=",")
#        np.savetxt(f"resultados/swarmVelocityL{level}S{swarmIdx}.csv", np.array(velocities), delimiter=",")
        end = datetime.now()
        print(f'tiempo actualizacion enjambre {end-start}')
        return np.array(ret), evaluationsCsv, self.bestParticleBin
    
    
#    @profile
    def genSubSwarms(self, universe, currLevel, levels, swarmsPerLevel):
        centroids = None
        idx = None
        if currLevel < len(swarmsPerLevel) and len(universe[0]) > swarmsPerLevel[currLevel]:
            
#            try:
#            print(f'np.array(universe[0]).shape {np.array(universe[0]).shape} level {currLevel}')
            centroids,_ = kmeans(universe[0],swarmsPerLevel[currLevel])
            
            idx,_ = vq(universe[0],centroids)
        ret = {}
        print(f'universe.shape {np.array(universe).shape}')
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
    
#    @profile
    def updateAccelParams(self):



#        print(evals-evaluations.reshape(np.prod(evaluations.shape)))
#        evaluations = evaluations.reshape(np.prod(evaluations.shape))
##        print(evals)
##        print(evaluations)
#        
#        self.accelPer[evals<evaluations]  -= 1.5
#        self.accelPer[evals>=evaluations]   += 1.5
#        
#        self.accelBest[evals<evaluations] -= 1.5
#        self.accelBest[evals>=evaluations]  += 1.5
#        self.accelPer[evals>evaluations.reshape(np.prod(evaluations.shape))] += 1
#        print(self.accelPer)
#        print(evals<evaluations.reshape(np.prod(evaluations.shape)))
#        exit()
        #binBest - binSwarm
        #print (f'binswarm {binSwarm}')
        #print (f'binBests {binBests}')
        #print (f'binBest - binSwarm {binBests - binSwarm}')
        #print (f'binglobalBest - binSwarm {self.bestParticleBin - binSwarm}')
        #binBests - binSwarm
        #exit()
#        self.accelBest = np.mean(self.bestParticleBin - binSwarm)
#        self.accelPer = np.mean(binBests - binSwarm)
        self.accelBest = 0.3#2.05*np.random.uniform() 
        self.accelPer = 0.5#2.05*np.random.uniform() 
        """if 0 < tendencia < 20: 
                self.accelBest += 0.1
                self.accelPer += 0.1
                self.accel += 0.1
        if 0 > tendencia:
                self.accelBest += -0.1
                self.accelPer += -0.1
                self.accel +=-0.1
        if 20 < tendencia:
                #self.accel -= 0.1
        #        self.accelBest += -1
                self.accelPer += -1
        return [self.accelPer
            ,self.accelBest
            ,self.randPer
            ,self.randBest]"""
    
#    @profile
    def genRandomSwarm(self, swarmSize = 50, featureSize = 2000):    
        swarm =        np.random.uniform(low=self.min, high=self.max, size=(swarmSize, featureSize))
        velocity =     np.random.uniform(size=(swarmSize, featureSize))
#        personalBest = np.random.uniform(low=self.min, high=self.max, size=(swarmSize, featureSize))
        personalBest = np.zeros((swarmSize, featureSize))
#        bestFound =    np.random.uniform(low=self.min, high=self.max, size=(featureSize))
        bestFound =    np.zeros((featureSize))
        evals =        np.ones((swarmSize)) * -math.pow(10,6)
#        np.random.uniform(size=(swarmSize))
        bestEval =     -math.pow(10,6)
        #bins = np.random.uniform(size=(swarmSize, featureSize))
#        bins = np.ones((swarmSize, featureSize))
#        bestBins = np.zeros((swarmSize, featureSize))
        return [swarm, velocity, personalBest, evals, bestFound, bestEval]
    
#    @profile
    def getNextLvlSwarms(self, swarms, globalBest, globalBestEval):
        bestParticles = []
        bestEvals = []
#        swarmsL = swarms.values()
        print(f'getNextLvlSwarms np.array(swarms).shape {np.array(swarms).shape}')
        for i in range(len(swarms)):
            print(f'getNextLvlSwarms np.array(swarms[i]).shape {np.array(swarms[i]).shape}')
            bestParticles.append(swarms[i][0][4])
            bestEvals.append(swarms[i][0][5])
        nxtSwarm = self.genRandomSwarm(len(bestParticles), swarms[0][0][4].shape[0])
        nxtSwarm[0] = bestParticles
        nxtSwarm[3] = bestEvals
        nxtSwarm[4] = globalBest
        nxtSwarm[5] = globalBestEval
        print(f'getNextLvlSwarms np.array(nxtSwarm[2][0]).shape {np.array(nxtSwarm[2][0]).shape}')
        return nxtSwarm
            
#    @profile
    def optimize(self, maximize, epochs):
        
        swarms = []
        universes = []
        universes.append(self.genSubSwarms(self.UNIVERSE, 0, self.LEVELS, self.numSubSwarms))
        
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            for level in range(self.LEVELS):
                swarms = universes[level]
                print(f'LEVEL {level}')        
                i=0
                
                pool = mp.Pool(3)
                swarmsList = [item for item in swarms.values()]  
#                print(f'swarmsList[0][5] {np.array(swarmsList[0][0][5])}')
#                exit()
                args = [list((swarm, self.numIter[level], swarm[0][5])) for swarm in swarmsList]
#                keys = swarms.keys()
#                print(f'args {np.array(args).shape}')
                ret = pool.starmap(self.updateSwarmData, args)
                pool.close()
                swarms = [data[0] for data in ret]
                evals = [data[1] for data in ret]
                
#                evals = np.array(ret)[:,1]
#                print(swarms[0])
#                print(evals[0])
#                print(f'global best!! {np.array(ret[0][2])}')
#                exit()
                for i in range(len(swarms)):
#                    print(f'global best!! {np.array(swarms[i][0][4]).shape}')
                    if self.globalBest is None or swarms[i][0][5] > self.globalBest:
                        self.globalBest = swarms[i][0][5]
                        self.bestParticle = swarms[i][0][4]
                        self.bestParticleBin = ret[i][2]
                    np.savetxt(f"resultados/swarmL{level}S{i}.csv", np.array(evals[i]), delimiter=",")
#                exit()
                
                
#                for swarmIdx in swarms.keys():
#                    i+=1
##                    print(f'level {level} swarm {i} of {len(swarms)}')
#                    swarmData = np.array(swarms[swarmIdx])
#                    print(f'swarmData {swarmData.shape} num iteraciones {self.numIter[level]}')
#                    nextSwarmData, evals = self.updateSwarmData(swarmData, self.numIter[level])
#                    np.savetxt(f"resultados/swarmL{level}S{swarmIdx}.csv", np.array(evals), delimiter=",")
##                    print(f'nextSwarmData {np.array(nextSwarmData)[:,3]}')
##                    exit()
##                    np.savetxt(f"resultados/swarmL{level}S{swarmIdx}.csv", np.array(nextSwarmData)[:,3], delimiter=",")
##                    if self.globalBest is None or nextSwarmData[0,5] > self.globalBest:
##                        self.globalBest = nextSwarmData[0,5]
##                        self.bestParticle = nextSwarmData[0,4]
##                        print(f'best obj {self.globalBest}')
#                    swarms[swarmIdx] = nextSwarmData
#                    if (swarmData[0] == nextSwarmData[0]):
#                        print(f'swarms[{swarmIdx}] {swarms[swarmIdx]} \n no actualizado')
#                        exit()
                
#                print(self.bestParticle)
#                print(f'optimize swarms {np.array(self.globalBest)}')
#                exit()
                nxtSwarm = self.getNextLvlSwarms(swarms, self.bestParticle, self.globalBest)
                
#                print(f'level+1 >= len(universes) {level+1} >= {len(universes)} {level+1 >= len(universes)}')
#                print(f'!!np.array(nxtSwarm[2]).shape {np.array(nxtSwarm[2]).shape}')
                if level+1 >= len(universes):
                    universes.append(self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms))
                else:
                    universes[level+1] = self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms)
                
