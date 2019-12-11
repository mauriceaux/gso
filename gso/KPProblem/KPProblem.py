#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:02:44 2019

@author: mauri
"""
import numpy as np
from . import read_instance_kp as instance_reader
from . import binarizationstrategy as _binarization
from .kp_repairStrategy import ReparaStrategy as repairStrategy
from datetime import datetime

class Problem():
    def __init__(self, instancePath = None):
        print(f'LEYENDO INSTANCIA {instancePath}')
        self.instance = instance_reader.Read(instancePath)
        if(self.instance.numItems != self.instance.itemWeights.shape[0]):
            raise Exception(f'El número de items {self.instance.numItems} es distinto al número de pesos {self.instance.itemWeights.shape[0]}')
        self.tTransferencia = "sShape1"
        self.tBinary = "Standar"
        self.minimize = False
        self.repairStrategy = repairStrategy(self.instance.numItems, self.instance.capacidad, self.instance.itemValues, self.instance.itemWeights)
        self.binarizationStrategy = _binarization.BinarizationStrategy(self.tTransferencia, self.tBinary)
   
    
    
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