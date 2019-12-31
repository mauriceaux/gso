#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:02:37 2019

@author: mauri
"""
from datetime import datetime
import numpy as np
from . import read_instance_kp as instance_reader
from . import binarizationstrategy as _binarization
from .kp_repairStrategy import ReparaStrategy as repairStrategy
import multiprocessing as mp

class KP():
    def __init__(self, instancePath):
        print(f'LEYENDO INSTANCIA {instancePath}')
        self.instancia = instancePath
        self.instance = instance_reader.Read(instancePath)
        print(f'self.instance.numItems {self.instance.numItems} ')
        if(self.instance.numItems != self.instance.itemWeights.shape[0]):
            raise Exception(f'El número de items {self.instance.numItems} es distinto al número de pesos {self.instance.itemWeights.shape[0]}')
        self.tTransferencia = "sShape1"
        self.tBinary = "Standar"
        self.minimize = False
        self.repairStrategy = repairStrategy(self.instance.numItems, self.instance.capacidad, self.instance.itemValues, self.instance.itemWeights)
        self.binarizationStrategy = _binarization.BinarizationStrategy(self.tTransferencia, self.tBinary)
        self.paralelo = False

    def getNombre(self):
        return 'knapsack'
    
    def getNumDim(self):
        return self.instance.numItems

    def getRangoSolucion(self):
        return {'max': 3.0, 'min':-3.0}
    
    def evalEnc(self, encodedInstance):
        decoded = self.decodeInstance(encodedInstance)
        repaired, numReparaciones = self.repairStrategy.repara(decoded)
        #repaired, numReparaciones = self.repairStrategy.reparaBatch(np.array([decoded]))
        fitness = self.evalInstance(repaired)
        return fitness, repaired, numReparaciones

    def evalEncBatch(self, encodedInstances, mejorSol):
        if mejorSol is None:
            mejorSol = encodedInstances[0]
        decoded = self.decodeInstanceBatch(encodedInstances, mejorSol)
        #for enc in encodedInstances:
        #    decoded.append(self.decodeInstance(encodedInstance))
        #repaired, numReparaciones = self.repairStrategy.repara(decoded)
        start = datetime.now()
        numReparaciones = 0
        repaired = self.repairStrategy.reparaBatch(np.array(decoded))
        fitness = self.evalInstanceBatch(repaired)
        end = datetime.now()
        #print(f'evalEncBatch demoro {end-start}')
        return fitness, repaired, numReparaciones
    
    def decodeInstance(self, encodedInstance):
        start = datetime.now()
        encodedInstance = self.binarizationStrategy.binarize(list(encodedInstance))
        end = datetime.now()
        binTime = end-start
        return np.array(encodedInstance)

    def decodeInstanceBatch(self, encodedInstances, mejorSol):
        start = datetime.now()
        encodedInstance = self.binarizationStrategy.binarizeBatch(encodedInstances, mejorSol)
        end = datetime.now()
        binTime = end-start
        #print(f'decodeInstanceBatch demoro {binTime}')
        return np.array(encodedInstance)
           
    def evalInstance(self, decoded):
        return self.fObj(decoded)

    def evalInstanceBatch(self, decoded):
        start = datetime.now()
        #ret = np.apply_along_axis(self.fObj, -1, decoded)
        ret = np.sum(self.instance.itemValues*decoded, axis=1)
        #print(ret)
        #print(ret.shape)
        #exit()
        end = datetime.now()
        #print(f'evalInstanceBatch demoro {end-start}')
        return ret
        #return np.array([self.fObj(d) for d in decoded])
    
    def fObj(self, solution):
        return np.sum(self.instance.itemValues*solution)
    
    def encode(self, decoded):
        decoded = np.array(decoded)
        decoded[decoded == 1] = self.getRangoSolucion()['max']
        decoded[decoded == 0] = self.getRangoSolucion()['min']
        return decoded
    
    def generarSolsAlAzar(self, numSols):
        start = datetime.now()
        args = np.ones((numSols, self.getNumDim())) * self.getRangoSolucion()['max']
#        args = np.random.uniform(size=(numSols, self.getNumDim()))
        _,sol,_ = self.evalEncBatch(args, args[0])
        #if self.paralelo:
        #    pool = mp.Pool(4)
        #    ret = pool.map(self.evalEnc, args)
        #    pool.close()
        #    sol = np.array([self.encode(item[1]) for item in ret])
        #else:
        #    sol = []
        #    for arg in args:
        #        _,bin,_ = self.evalEnc(arg)
        #        sol.append(bin)
        #    sol = np.array(sol)
        end = datetime.now()
        #print(f'generarSolsAlAzar demoro {end-start}')
        return sol