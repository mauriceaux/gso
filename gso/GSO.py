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
import ctypes
#from statsmodels.tsa.seasonal import seasonal_decompose
#import line_profiler
from threading import Lock
lock = Lock()
class GSO:
    def __init__(self, globalBest=None, gBestParticle = None):
#        np.random.seed(0)
        if globalBest is not None:
            global gBest
        
            gBest = globalBest
        if gBestParticle is not None:
            global gBestP
            gBestP = gBestParticle
        self.evalEnc = None
        self.inertia=None
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
        self.randPer   = 1
        self.randBest  = 1
        self.minVel = -1
        self.maxVel = 1
        self.scaler = MinMaxScaler(feature_range=(self.minVel,self.maxVel))
        self.decode = None
        self.repair = None
        self.evalDecoded = None
        self.onlineAdjust = False
        self.mejoresResultados = ""
    def setScaler(self, minVal, maxVal):
        self.scaler = MinMaxScaler(feature_range=(minVal,maxVal))
        
#    @profile
    def setEvalEnc(self, evalEnc):
        self.evalEnc = evalEnc
        
#    @profile
    def moveSwarm(self, swarm, velocity, personalBest, bestFound):        
        self.accelPer = 2.05*np.random.uniform()
        self.accelBest = 2.05*np.random.uniform()
        self.randPer = np.random.uniform(low=-1, high=1)
        self.randBest = np.random.uniform(low=-1, high=1)
#        self.accelPer = 0.1
#        self.accelBest = 0.1
#        self.randPer = 1
#        self.randBest = 1
        personalDif = personalBest - swarm
        personalAccel = self.accelPer * self.randPer * personalDif
        bestDif = bestFound - swarm
        bestAccel = self.accelBest * self.randBest * bestDif
        acceleration =  personalAccel + bestAccel
        
        nextVel = (self.inertia*velocity) + acceleration
        nextVel[nextVel > self.maxVel]  = self.maxVel
        nextVel[nextVel < self.minVel]  = self.minVel
        ret = swarm+nextVel
        ret[ret > self.max]  = self.max
        ret[ret < self.min] = self.min
        return ret, nextVel


#    @profile
    def updateSwarmData(self, swarmData, iterations, globalBest):
        global gBest
        global gBestP
        exitCount = 0
        start = datetime.now()
        swarm         = np.vstack(np.array(swarmData)[:,0])
        velocity      = np.vstack(np.array(swarmData)[:,1])
        personalBest  = np.vstack(np.array(swarmData)[:,2])
        evals         = np.array(swarmData)[:,3]
        
        bestFound     = np.vstack(np.array(swarmData)[:,4])
#        bestEval      = np.array(swarmData)[0,5]
        bestEval      = None
        
        
        
        bestParticleBin = np.vstack(np.array(swarmData)[:,6])[0]
#        print(bestParticleBin.shape)
#        exit()
        evaluationsCsv = []
        mejoresResultados = ""
        for i in range(iterations):
#            if self.onlineAdjust:
#                self.updateAccelParams(evaluationsCsv[-4:])
            startIter = datetime.now()
#            print(np.array(swarm).shape)
#            exit()
            if exitCount >= 5:
                velocity = np.ones((swarm.shape))*self.minVel
            self.inertia = 1 - (i/(iterations + 1))
#            print(velocity)
            if len(swarm) == 1:
                #print(f'out {np.frombuffer(gBestP.get_obj(), float).shape}')
                #exit()
                bests = np.tile(np.frombuffer(gBestP.get_obj(), dtype=ctypes.c_float), (swarm.shape[0], 1))
                nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, bests)
            else:
                nswarm, velocity = self.moveSwarm(swarm, velocity, personalBest, bestFound)
            returning = [self.evalEnc(item) for item in list(nswarm)]
            binSwarm = [item[1] for item in returning]
            binSwarm = np.vstack(binSwarm)
#            nswarm = binParticle.copy()
#            nswarm[nswarm == 1] = self.max
#            nswarm[nswarm == 0] = self.min
#            print(nswarm)
#            exit()
            evaluations = [item[0] for item in returning]
            evaluations = np.array(np.vstack(evaluations))
            
            evaluations = evaluations.reshape((swarm.shape[0]))
#            print(evaluations)
#            exit()
            evaluationsCsv.append(evaluations)
            bestidx = evaluations > evals
            
            
            
            if bestidx.any(): personalBest[bestidx] = nswarm[bestidx]
            idx = np.argmax(evaluations)
            
            if bestEval is None or (evaluations[idx] > bestEval):
                bestEval =  evaluations[idx]
                
                bestFound = np.tile(nswarm[idx], (nswarm.shape[0],1))
#                print(f'binParticle[idx] {binParticle[idx]}')
                bestParticleBin = np.copy(binSwarm[idx])
            
            if gBest.value is None or evaluations[idx] > gBest.value:
                with gBest.get_lock():
                    gBest.value = evaluations[idx]
                    
                with gBestP.get_lock():
                    
                    gBestP[:]=np.copy(nswarm[idx][:])
#                if exitCount > 0: exitCount -= 1
#            else:
#                exitCount += 1
            mejoresResultados+=f'{datetime.timestamp(datetime.now()),gBest.value}\n'
            
            intervalo = evaluationsCsv[-4:]
            if self.alza(intervalo):
                if exitCount > 0: exitCount -= 1
            else:
                self.updateAccelParams()
                exitCount += 1
            
            evals = evaluations
            swarm = np.copy(nswarm)
            endIter = datetime.now()
            
            print(f'best obj {bestEval} {gBest.value} duracion iteracion {i} {endIter-startIter}')
#            if exitCount >= 5:
#                newBestFound = bestFound[0]
#                indices0 = np.where(newBestFound > 0)
#                if len(indices0[0]) > 0:
#                    newBestFound[np.random.choice(indices0[0])] = self.min
#                else:
#                    newBestFound[np.argmax(newBestFound)] = self.min
#                bestFound = np.tile(newBestFound, (nswarm.shape[0],1))
                
#                break
#            
              
#        print(bestFound)
#        exit()
        
#        print(np.random.choice(indices0[0]))
        
#        print(np.random.choice(np.where(bestFound[0] == 0)))
#        exit()
        ret = []
        
        for i in range(swarm.shape[0]):
            ret.append([swarm[i], velocity[i], personalBest[i], evaluations[i], bestFound[i], bestEval])
        end = datetime.now()
        print(f'tiempo actualizacion enjambre {end-start}')
#        print(np.array(mejoresResultados))
#        exit()
        return np.array(ret), evaluationsCsv, bestParticleBin, mejoresResultados
    
    
#    @profile
    def genSubSwarms(self, universe, currLevel, levels, swarmsPerLevel):
        centroids = None
        idx = None
        if currLevel < len(swarmsPerLevel) and len(universe[0]) > swarmsPerLevel[currLevel]:
            centroids,_ = kmeans(np.array(universe[0], dtype=np.float),swarmsPerLevel[currLevel])
            
            idx,_ = vq(universe[0],centroids)
        ret = {}
#        print(f'agregando {swarmsPerLevel[currLevel]} swarms')
        for i in range(len(universe[0])):
            
            if currLevel < len(swarmsPerLevel) and len(universe[0]) > swarmsPerLevel[currLevel]:
                if idx[i] not in ret.keys(): ret[idx[i]] = []
                ret[idx[i]].append([universe[0][idx[i]]
                    , universe[1][idx[i]]
                    , universe[2][idx[i]]
                    , universe[3][idx[i]]
                    , universe[4]
                    , universe[5]
                    , universe[6]])
                
            else:
                if i not in ret.keys(): ret[i] = []
                ret[i].append([universe[0][i]
                    , universe[1][i]
                    , universe[2][i]
                    , universe[3][i]
                    , universe[4]
                    , universe[5]
                    , universe[6]])
#        print(f'agregado {ret} swarms')
        return ret
    
    def alza(self, intervalo):
        scaler = MinMaxScaler()
        intervalo = scaler.fit_transform(intervalo)
        grupoInicio = intervalo[:2]
        grupoFin = intervalo[-2:]
        dif = np.mean(grupoFin) - np.mean(grupoInicio)
        return dif >= 0
        
#    @profile
    def updateAccelParams(self):
#        intervalo = evaluationsCsv[-10:]
#        grupoInicio = intervalo[:2]
#        grupoFin = intervalo[-2:]
#        dif = np.mean(grupoFin) - np.mean(grupoInicio)
#        if dif < 50:
#            self.accelBest = np.random.uniform(low=-3,high=3)
#            self.accelPer = np.random.uniform(low=-3,high=3)
#            self.accel = np.random.uniform(low=-3,high=3)
        if self.accelBest > 0:
            self.accelBest += -1
        else:
            self.accelBest += 1
#        self.accelBest *= -1
        if self.accelPer > 0:
            self.accelPer += -1
        else:
            self.accelPer += 1
        if self.inertia > 0:
            self.inertia += -1
        else:
            self.inertia += 1
#        print(f'update params {grupoFin-grupoInicio}')
#        exit()
        
    
#    @profile
    def genRandomSwarm(self, swarmSize = 50, featureSize = 2000):    
#        swarm =        np.random.uniform(low=self.min, high=self.max, size=(swarmSize, featureSize))
        swarm =        np.ones((swarmSize, featureSize)) * self.min
#        print(len(swarm.tolist()))
#        exit()
#        swarm[0] = np.ones((featureSize)) * self.min
#        swarm[1] = np.ones((featureSize)) * self.max
        velocity =     np.random.uniform(size=(swarmSize, featureSize))
#        personalBest = np.ones((featureSize)) * self.min
#        personalBest = np.zeros((featureSize)) 
#        personalBest = np.random.uniform(low=self.min, high=self.max, size=(featureSize)) 
#        pool = mp.Pool()
        pool = mp.Pool(4)
        r = pool.map(self.evalEnc, swarm.tolist())
        pool.close()
#        r = [self.evalEnc(item) for item in swarm]
        evals = [item[0] for item in r]
        swarm = [item[1] for item in r]
        swarm = np.array(swarm)
        personalBestBin = [item[1] for item in r]
        swarm[swarm == 1] = self.max
#        print(swarm)
#        exit()
        swarm[swarm == 0] = self.min
#        returning = self.evalEnc(personalBest)
        personalBest = np.copy(swarm)
        bestIdx = np.argmax(evals)
#        print(returning[0])
#        exit()
#        exit()
#        personalBest = [returning[1] for item in range(swarmSize)]
#        swarm[0] = personalBest[0].copy()
#        personalBest = np.vstack(personalBest)
#        bestFound = personalBest[0]
        bestFound = swarm[bestIdx]
#        evals =[returning[0] for item in range(swarmSize)]
#        bestEval = evals[np.argmax(evals)]
#        bestEval = returning[0]
        bestEval = evals[bestIdx]
#        personalBestBin = np.zeros((swarmSize, featureSize))
        return [swarm, velocity, personalBest, evals, bestFound, bestEval, personalBestBin]
    
#    @profile
    def getNextLvlSwarms(self, swarms, globalBest, globalBestEval):
        bestParticles = []
        bestEvals = []
#        print(type(swarms))
#        exit()
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
        """
        for level in range (self.LEVELS):
            if level >= len(self.numSubSwarms):
                np.savetxt(f"resultados/swarmL{level}S{0}.csv", np.array([]), delimiter=",")
            else:
                for i in range (self.numSubSwarms[level]):
                    np.savetxt(f"resultados/swarmL{level}S{i}.csv", np.array([]), delimiter=",")
        """
        gBest = Value('f', -math.pow(10,6))
        
        #print(f'inicio in {self.UNIVERSE[4].dtype}')
        #print(f'inicio in {self.UNIVERSE[4].shape}')
        #print(f'inicio in {self.UNIVERSE[4]}')
        gBestP = mp.Array(ctypes.c_float, self.UNIVERSE[4])
        #print(np.frombuffer(gBestP.get_obj(), dtype=ctypes.c_int).shape)
        #print(np.frombuffer(gBestP.get_obj(), dtype=ctypes.c_int))
        #exit()
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            for level in range(self.LEVELS):
                swarms = universes[level]
                print(f'LEVEL {level}, swarms {len(swarms)}')        
                swarmsList = [item for item in swarms.values()]  
                startSwarm = datetime.now()
                args = [list((swarm, self.numIter[level], swarm[0][5])) for swarm in swarmsList]
                
                pool = mp.Pool(4, initializer = self.__init__,  initargs = (gBest, gBestP))
                ret = pool.starmap(self.updateSwarmData, args)
                pool.close()
                endSwarm = datetime.now()
                print(f'Optimization for level {level} completed in: {endSwarm - startSwarm}')
                swarms = {i:ret[i][0] for i in range(len(ret))}
#                swarms = [data[0].tolist() for data in ret]
                evals = [data[1] for data in ret]
                universes[level] = swarms.copy()
                mr = ""
                for item in ret:
                    mr += item[3].replace("(","").replace(")","")
#                mr = [np.array(item[3]) for item in ret]
#                print(mr)
#                exit()
                self.mejoresResultados += mr
                for i in range(len(swarms)):
                    #print(evals)
                    #exit()
#                    np.savetxt(f"resultados/swarmL{level}S{i}.csv", np.array(evals[i]), delimiter=",")
                    print(f'global best {np.array(swarms[i][0][5])} swarm {i} level {level} gBest.value {gBest.value}')
#                    start = datetime.now()
                    if self.globalBest is None or swarms[i][0][5] > self.globalBest:
#                        gBest.value = swarms[i][0][5]
                        self.globalBest = swarms[i][0][5]
                        self.bestParticle = swarms[i][0][4]
                        self.bestParticleBin = ret[i][2]
#                    end = datetime.now()
#                    print(f'if demoro {end-start}')
#                    start = datetime.now()
#                    with open(f"resultados/swarmL{level}S{i}.csv", "ab") as f:
#                        np.savetxt(f, np.array(evals[i]), delimiter=",")
#                    end = datetime.now()
#                    print(f'archivo demoro {end-start}')
#                    start = datetime.now()
                    if level < self.LEVELS -1:
                        nxtSwarm = self.getNextLvlSwarms(swarms, self.bestParticle, self.globalBest)
#                    end = datetime.now()
#                    print(f'next level demoro {end-start}')
                if level < self.LEVELS -1:
                    if level+1 >= len(universes):
                        universes.append(self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms))
                    else:
                        universes[level+1] = self.genSubSwarms(nxtSwarm, level+1, self.LEVELS, self.numSubSwarms)
                
