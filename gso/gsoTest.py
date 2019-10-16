#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:09:41 2019

@author: mauri
"""

from GSO import GSO
from Problem import Problem
import numpy as np
import multiprocessing as mp
from scipy.cluster.vq import kmeans,vq




    
#        
#    
#def updateEvaluations(swarmData, evaluations, globalBest):
##    print(np.array(swarmData[0]).shape)
##    print(np.array(swarmData).shape)
##    print(swarmData[0])
##    exit()
#    swarms        = np.array(swarmData[0])
#    velocities    = np.array(swarmData[1])
##    personalBests = swarmData[2]
#    personalBests = np.array(swarmData[2])
#    evals         = np.array(swarmData[3])
#    bestFound     = np.array(swarmData[4])
#    bestEval      = np.array(swarmData[5])
#    
##    print(f'np.where {np.where(evaluations>np.reshape(evals,(-1)))}')
##    print(f'evals {evals}')
#    evals = np.array(evals).reshape(-1)
##    print(f'evals {np.array(evals).reshape(-1)}')
##    print(f'evaluation {evaluations}')
##    exit()
##    print(f'evaluation>evals {evaluations>evals}')
##    bestidx = np.where(evaluations>evals)
#    bestidx = evaluations>evals
##    print(f'swarms {swarms.shape}')
##    print(f'swarmData[2] {swarmData[2].shape}')
#    
#    for i in range(bestidx.shape[0]):
#        if bestidx[i]:
#            swarmData[2][i] = personalBests[i]
#            swarmData[3][i] = evaluations[i]
#        else:
#            swarmData[2][i] = swarms[i]
#            swarmData[3][i] = evals[i]
##    swarmData[3] = np.where(evaluations>evals, evaluations, evals)
#    bestEvalIdx  = np.argmax(swarmData[3])
##    bestEvalGIdx  = np.argmax(swarmData[5])
#    swarmData[4] = swarmData[2][bestEvalIdx]
##    print(f'swarmData[5][bestEvalIdx] {swarmData[5][bestEvalIdx]}')
##    print(f'bestEval {bestEval}')
#    if bestEval[bestEvalIdx] > globalBest:
#        globalBest = swarmData[5]
##    bestidx = np.where(evaluations>np.reshape(evals,(-1)), evaluations, np.reshape(evals,(-1)))
##    print(f'bestidx {bestidx}')
##    print(f'evaluations {evaluations[bestidx]}')
##    print(f'evals {np.argmax(np.reshape(evals,(-1)))} {np.reshape(evals,(-1))}')
##    exit()
##    
##    maxIdx = np.argmax(evaluations)
#    bestIdx = np.argmax(swarmData[5])
##    bestEvals = np.reshape(swarmData[5], (-1))
#    globalBest = swarmData[5][bestIdx]
##    print(np.reshape(bestEvals, (-1)))
##    print(np.reshape(bestEvals, (-1)).shape)
##    print(f'1 {evaluations[maxIdx]} bestEvals {bestEvals}')
##    exit()
##    if evaluations[maxIdx] > globalBest:
##        swarmData[2]=swarmData[2][maxIdx]
##        swarmData[4]=swarmData[0][maxIdx]
##        swarmData[5]=evaluations[maxIdx]
##        globalBest = evaluations[maxIdx]
#        
##        if evaluations[maxIdx] > globalBest:
##            globalBest = evaluations[maxIdx]
#    return swarmData, globalBest
#
#def genSubSwarms(universe, currLevel, levels, swarmsPerLevel):
#    centroids = None
#    idx = None
#    if currLevel < len(swarmsPerLevel):
#        centroids,_ = kmeans(universe[0],swarmsPerLevel[currLevel])
#        idx,mmm = vq(universe[0],centroids)
##    print(mmm.shape)
#    ret = {}
##    print(np.array(universe[4]).shape)
##    exit()
#    for i in range(len(universe[0])):
#        
#        
##        print(f'current level {currLevel} universe shape {np.array(universe).shape} len(swarmsPerLevel) {len(swarmsPerLevel)}')
##        exit()
#        if currLevel >= len(swarmsPerLevel):
#            if 0 not in ret.keys(): ret[0] = []
#            ret[0].append([universe[0][i]
#                , universe[1][i]
#                , universe[2][i]
#                , universe[3][i]
#                , universe[4]
#                , universe[5]])
#            
#        else:
#            if idx[i] not in ret.keys(): ret[idx[i]] = []
#            ret[idx[i]].append([universe[0][idx[i]]
#                , universe[1][idx[i]]
#                , universe[2][idx[i]]
#                , universe[3][idx[i]]
#                , universe[4]
#                , universe[5]])
##        print (f'agregando universe[{idx[i]}] a ret[{i}] ')
##    print(f'len {len(ret[0])}')
##    print(f'len[0] {len(ret[0])}')
##    exit()
#    return ret

gso = GSO()
problem = Problem()
gso.setEvalEnc(problem.evalEnc)
gso.UNIVERSE = gso.genRandomSwarm(500, 100)
gso.LEVELS = 4
gso.numIter = [5,10,15,20,25]
gso.numSubSwarms = [100,10,5]

EPOCHS = 10
gso.optimize(maximize=False, epochs = EPOCHS)


print(f'best particle sum {np.sum(gso.bestParticle)}')
print(f'best particle sum sin {np.sin(np.sum(gso.bestParticle))}')
print(f'best obj {gso.globalBest}')
exit()



#
#swarms = []
#
##for i in range(10):
##    swarms.append(genRandomSwarm())
#
#swarmSize = 50
#featureSize = 2000
#UNIVERSE = gso.genRandomSwarm(swarmSize, featureSize)
#
#universes = []
##universes.append(UNIVERSE)
#
##print(len(swarms[3]))
#numIter = [10,20,30]
#globalBest = 0
#globatBestParticle = None
#LEVELS = 3
#universes.append(gso.genSubSwarms(UNIVERSE, 0, LEVELS, [10,5]))
#
#for epoch in range(EPOCHS):
#    print(f'epoch {epoch}')
#    for level in range(LEVELS):
#        swarms = universes[level]
#        print(f'LEVEL {level}')
#    
##        print(swarms.keys())
#        
#        for swarmIdx in swarms.keys():
#        #    print(f'swarmData pre processed {np.array(swarms[swarmIdx]).shape}')
#            swarmData = np.array(swarms[swarmIdx])
#        #    print(f'swarms[{swarmIdx}] es {len(swarms[swarmIdx])}')
#        #    print(f'swarmData es {swarmData.shape}')
#        #    print(f'swarm {np.vstack(swarmData[:,0]).shape}')
#        #    print(f'velocity {np.vstack(swarmData[:,1]).shape}')
#        #    print(f'personalBest {np.vstack(swarmData[:,2]).shape}')
#        #    print(f'evals {np.vstack(swarmData[:,3]).shape}')
#        #    print(f'bestFound {np.vstack(swarmData[:,4]).shape}')
#        #    print(f'bestEval {np.vstack(swarmData[:,5]).shape}')
#        #    exit()
#        #    print(f'swarmData {swarmData[3]}')
#            nextSwarmData = gso.updateSwarmData(swarmData, numIter[level])
#            if nextSwarmData[0,5] > globalBest:
#                globalBest = nextSwarmData[0,5]
#                globalBestParticle = nextSwarmData[0,4]
#                print(f'best obj {globalBest}')
#            swarms[swarmIdx] = nextSwarmData
#        #    print(f'best founds {globalBestParticle.shape}')
#        #    print(f'nextSwarmData es {np.array(nextSwarmData).shape}')
#            if (swarmData[0] == nextSwarmData[0]):
#                print(f'swarms[{swarmIdx}] {swarms[swarmIdx]} \n no actualizado')
#                exit()
#        
#        nxtSwarm = gso.getNextLvlSwarms(swarms, globalBestParticle, globalBest)
#        universes.append(gso.genSubSwarms(nxtSwarm, level+1, LEVELS, [10,5]))
##    exit()
#    
    
    
print(f'best particle sum {np.sum(globalBestParticle)}')
print(f'best particle sum sin {np.sin(np.sum(globalBestParticle))}')
print(f'best obj {globalBest}')
exit()











#print(np.array(UNIVERSE[5]))
#exit()
LEVELS = 2
swarmsPerLevel = [10] # swarmsPerLevel[0] swarm number on level 1, swarmsPerLevel[1] swarm number on level 2...





globalBest = 0
epochs = 3
iterations   = 50
universes = []
universes.append(UNIVERSE)
#print(f'universes[0][5] {universes[0][5]}')
#exit()
#print(f"universes[0] shape {np.array(universes[0]).shape}")
for _ in range(epochs):
    for i in range(LEVELS):
#        print(f"universes[i] shape {np.array(universes[i]).shape}")
        swarms = genSubSwarms(universes[i], i, LEVELS, swarmsPerLevel)
        
#        print(f"swarms shape {np.array(swarms).shape}")
        for swarmIdx in swarms:
#            print(f'subswarm len(swarms[swarmIdx]) {len(swarms[swarmIdx])} ')
#            print(f'swarmIdx {swarmIdx} swarm shape {np.array(swarms[swarmIdx]).shape}')
            swarmData = swarms[swarmIdx]
#            print(f'swarms {np.array(swarmData).shape}')
#            exit()
#            print(len(swarmData[0][4]))
#            print(len(swarmData[1]))
#            exit()
            
            
            swarmData, evaluations = gso.updateSwarmData(swarmData, iterations, swarmSize)
            swarmData, globalBest  = updateEvaluations(swarmData, evaluations, globalBest)
            universes.append(swarmData)
            print(f'globalBest {globalBest}')
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#def updateSwarmData(swarmData, iterations, swarmSize):
##    print(np.array(swarmData).shape)
##    exit()
#    swarms       = np.vstack(np.array(swarmData)[:,0])
##    print(np.vstack(swarm).shape)
##    exit()
#    velocities     = np.vstack(np.array(swarmData)[:,1])
#    personalBests = np.vstack(np.array(swarmData)[:,2])
#    evals        = np.vstack(np.array(swarmData)[:,3])
#
#    bestsFound    = np.vstack(np.array(swarmData)[:,4])
##    print(f'bestEval {np.vstack(np.array(swarmData)[:,5])}')
#    bestEvals     = np.array(swarmData)[:,5]
##    print(f'bestEval {bestEval}')
##    exit()
##    print(np.array(swarmData)[:,4])
##    exit()
#    
#    swarmSize = swarms.shape[0]
##    print(swarmSize)
#    for _ in range(iterations):
#        for i in range(swarms.shape[0]):
#            swarm = swarms[i]
#            
#            velocity = velocities[i]
#            personalBest = personalBests[i]
#            bestFound = bestsFound[i]
#            bestEval = bestEvals[i]
#            nswarm, velocities = gso.moveSwarm(swarm, velocity, personalBest, bestFound)
##            print(f'swarm  {swarm}')
#            print(f'nswarm {nswarm.shape}')
#            args = nswarm
#            pool = mp.Pool()
#            evaluations = pool.map(problem.evalEnc, list(args))
##        print(len(evaluations))
##        exit()
#            pool.close()
#            evaluations = np.array(np.vstack(evaluations))
#            
#        
#            evaluations = evaluations.reshape((swarm.shape[0]))
#            evals = evals.reshape((swarmSize))
#            print(f'evals {evals.shape}')
#            print(f'evaluations {evaluations.shape}')
##            exit()
##        print(evals)
###        exit()
##        print(bestEval)
##        exit()
#            bestidx = evaluations>evals
##        print(bestidx)
##        exit()
#            for j in range(bestidx.shape[0]):
#                if bestidx[j]:
#                    personalBest[j] = nswarm[i]
#                    if evaluations[j] > bestEval:
#                        bestFound = nswarm[j]
#                        bestEval = evaluations[j]
#            swarms[i] = swarm
##            evals = evaluations
##        personalBest[bestidx] = nswarm[bestidx]
##        swarmData[0] = nswarm
#    
#    return [nswarm, velocities, personalBest, evaluations, bestFound, bestEval], evaluations
            
            
#def genRandomSwarm(swarmSize = 50, featureSize = 2000):    
#    swarm =        np.random.uniform(size=(swarmSize, featureSize))
#    velocity =     np.random.uniform(size=(swarmSize, featureSize))
#    personalBest = np.random.uniform(size=(swarmSize, featureSize))
#    bestFound =    np.random.uniform(size=(featureSize))
#    evals =        np.random.uniform(size=(swarmSize))
#    bestEval =     np.random.uniform()
##    print(f'bestEval {bestEval}')
##    exit()
#    return [swarm, velocity, personalBest, evals, bestFound, bestEval]