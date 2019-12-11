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

    def getNombre(self):
        return 'knapsack'
    
    def getNumDim(self):
        return self.instance.numItems

    def getRangoSolucion(self):
        return {'max': 5.0, 'min':-5.0}
    
    def evalEnc(self, encodedInstance):
        decoded = self.decodeInstance(encodedInstance)
        repaired, numReparaciones = self.repairStrategy.repara(decoded)
        fitness = self.evalInstance(repaired)
        return fitness, decoded, numReparaciones
    
    def decodeInstance(self, encodedInstance):
        start = datetime.now()
        encodedInstance = self.binarizationStrategy.binarize(list(encodedInstance))
        end = datetime.now()
        binTime = end-start
        return np.array(encodedInstance)
           
    def evalInstance(self, decoded):
        return self.fObj(decoded)
    
    def fObj(self, solution):
        return np.sum(self.instance.itemValues*solution)
    
    def generarSolsAlAzar(self, numSols):
#        args = np.zeros((numSols, self.getNumDim()), dtype=np.float)
        args = np.random.uniform(size=(numSols, self.getNumDim()))
        args = np.ones((numSols, self.getNumDim()), dtype=np.float) * self.getRangoSolucion()['max']
        pool = mp.Pool(4)
        ret = pool.map(self.evalEnc, args)
        pool.close()
        sol = np.array([item[1] for item in ret])
        return sol