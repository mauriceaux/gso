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
        self.scaler = MinMaxScaler(feature_range=(0,1))
        np.random.seed(0)
#        self.accelPer  = 2.05 * np.random.uniform()
#        self.accelBest = 2.05 * np.random.uniform()
        self.accelPer  = 0.1
        self.accelBest = 0.1
        
        self.randPer   = 2.05 * np.random.uniform()
        self.randBest  = np.random.uniform()
        self.minVel = -1
        self.maxVel = 1
        self.decode = None
        self.repair = None
        self.evalDecoded = None
            
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
        
        randPer = np.random.uniform(low=-1, high=1)
        randBest = np.random.uniform(low=-1, high=1)
#        randPer = 1
#        randBest = 1
        
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
    def updateSwarmData(self, swarmData, iterations, swarmIdx, level):
        swarm         = np.vstack(np.array(swarmData)[:,0])
        velocity      = np.vstack(np.array(swarmData)[:,1])
        personalBest  = np.vstack(np.array(swarmData)[:,2])
        evals         = np.array(swarmData)[:,3]
        bestFound     = np.vstack(np.array(swarmData)[:,4])
        bestEval      = np.array(swarmData)[0,5]
#        print(f'bestEval {bestEval}')
#        exit()
        evaluationsCsv = []
        particles = []
        velocities = []
#        self.accelPer  = np.ones((swarm.shape))
#        self.accelBest = np.ones((swarm.shape))
        for i in range(iterations):
            self.accelPer  = 2.05 * np.random.uniform()
            self.accelBest = 2.05 * np.random.uniform()
            nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, bestFound)
#            print(f'nswarm {nswarm}')
#            print(f'nswarm {nswarm.shape}')
#            exit()
            particles.append(swarm[0])
            velocities.append(velocity[0])
#            norm = (nswarm-velocity)
#            norm[norm>self.max] = self.max
#            norm[norm<self.min] = self.min
#            if (norm != swarm).all(): 
#                print(f'velocidad mal aplicada')
#                print(f"swarm \n{swarm} \nvelocity {velocity}\nnorm {norm}\nnswarm {nswarm}")
#                exit()
            args = nswarm
            start = datetime.now()
            
#            pool = mp.Pool(8)
#            binParticle = pool.map(self.decode, list(args))
#            pool.close()
##            print(bins)
##            print(np.array(bins).shape)
##            exit()
#            pool = mp.Pool(5)
#            binParticle = pool.map(self.repair, list(binParticle))
#            pool.close()
##            print(repaired)
##            print(np.array(repaired).shape)
##            exit()            
#            pool = mp.Pool(5)
#            evals = pool.map(self.evalDecoded, list(binParticle))
#            pool.close()
#            evaluations = np.array(evals)
#            print(evals)
#            print(np.array(evals).shape)
#            exit()                        
            
            pool = mp.Pool(4)
#            evaluations, binParticle = pool.map(self.evalEnc, list(args))
            returning = pool.map(self.evalEnc, list(args))
            pool.close()
            returning = np.array(returning)
            binParticle = returning[:,1]
            evaluations = returning[:,0]
            end = datetime.now()
#            print(f'tiempo evaluacion {end-start}')
            

            evaluations = np.array(np.vstack(evaluations))
#            print(evaluations)
            
            velParams = self.getVelParams(swarm, nswarm, evals, evaluations)
            self.accelPer  = velParams[0]
            self.accelBest = velParams[1]            
            self.randPer   = velParams[2]
            self.randBest  = velParams[3]
#            print(evaluations.shape)
#            exit()
            
            self.globalBest is None
#            print(evaluations)
            evaluations = evaluations.reshape((swarm.shape[0]))
            evaluationsCsv.append(evaluations)
#            print(f'evaluations {evaluations}')
            bestidx = evaluations > evals
#            dif = personalBest
            
#            bestEval1 = evaluations[np.argmax(evaluations)] > self.globalBest ? evaluations[np.argmax(evaluations)] : self.globalBest
            
            personalBest[bestidx] = nswarm[bestidx]
#            for j in range(bestidx.shape[0]):
##                print(f'evaluations[j] {evaluations[j]}')
#                if bestidx[j]:
##                    personalBest[j] = nswarm[j]
#                    
#                    if self.globalBest is None or evaluations[j] > self.globalBest:
##                    if evaluations[j] > bestEval:
##                        print(f'{evaluations[j]} > {bestEval} {evaluations[j] > bestEval}')
#                        bestFound = np.tile(nswarm[j], (nswarm.shape[0],1))
##                        print(f'bestFound {bestFound.shape}')
##                        exit()
##                        bestEval = evaluations[j]
##                        self.globalBest = evaluations[j]
#                        self.bestParticle = bestFound[0]
#                        self.bestParticleBin = binParticle[j]
#                        print(f'best obj {self.globalBest} iter {i}')
#            if (bestEval1 != bestEval).any(): 
#                print(f'no iguales\n{bestEval1} {bestEval}')
#                exit()
            
            bestEval = self.globalBest
            idx = np.argmax(evaluations)
            if self.globalBest is None or evaluations[idx] > self.globalBest:
                bestEval =  evaluations[idx]
                self.globalBest = evaluations[idx]
                bestFound = np.tile(nswarm[idx], (nswarm.shape[0],1))
                self.bestParticle = bestFound[0]
                self.bestParticleBin = binParticle[idx]
                print(f'best obj {self.globalBest} iter {i}')
#            print(f'bestFound \n{bestFound.shape}\n{bestFound1.shape}')
#            exit()
            evals = evaluations
            swarm = nswarm
        ret = []
#        exit()
        
#        print(self.accelPer)
#        print(self.accelBest)
        
        for i in range(swarm.shape[0]):
            ret.append([swarm[i], velocity[i], personalBest[i], evaluations[i], bestFound[i], bestEval])
#        print(np.array(particles).shape)
        np.savetxt(f"resultados/swarmL{level}S{swarmIdx}.csv", np.array(evaluationsCsv), delimiter=",")
        np.savetxt(f"resultados/swarmMovementL{level}S{swarmIdx}.csv", np.array(particles), delimiter=",")
        np.savetxt(f"resultados/swarmVelocityL{level}S{swarmIdx}.csv", np.array(velocities), delimiter=",")
        return np.array(ret)
    
    
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
    def getVelParams(self,swarm, nswarm, evals, evaluations):
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
        return [self.accelPer
            ,self.accelBest
            ,self.randPer
            ,self.randBest]
    
#    @profile
    def genRandomSwarm(self, swarmSize = 50, featureSize = 2000):    
        swarm =        np.random.uniform(low=self.min, high=self.max, size=(swarmSize, featureSize))
        velocity =     np.random.uniform(size=(swarmSize, featureSize))
        personalBest = np.random.uniform(low=self.min, high=self.max, size=(swarmSize, featureSize))
        bestFound =    np.random.uniform(low=self.min, high=self.max, size=(featureSize))
        evals =        np.ones((swarmSize)) * -math.pow(10,6)
#        np.random.uniform(size=(swarmSize))
        bestEval =     -math.pow(10,6)
        return [swarm, velocity, personalBest, evals, bestFound, bestEval]
    
#    @profile
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
                
                for swarmIdx in swarms.keys():
                    i+=1
#                    print(f'level {level} swarm {i} of {len(swarms)}')
                    swarmData = np.array(swarms[swarmIdx])
#                    print(f'swarmData {swarmData.shape}')
                    nextSwarmData = self.updateSwarmData(swarmData, self.numIter[level], swarmIdx, level)
#                    print(f'nextSwarmData {np.array(nextSwarmData)[:,3]}')
#                    exit()
#                    np.savetxt(f"resultados/swarmL{level}S{swarmIdx}.csv", np.array(nextSwarmData)[:,3], delimiter=",")
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
#                print(f'level+1 >= len(universes) {level+1} >= {len(universes)} {level+1 >= len(universes)}')
                if level+1 >= len(universes):
                    universes.append(self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms))
                else:
                    universes[level+1] = self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms)
        
