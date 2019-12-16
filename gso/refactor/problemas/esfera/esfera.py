#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:02:37 2019

@author: mauri
"""
from datetime import datetime
import numpy as np
from . import binarizationstrategy as _binarization
import multiprocessing as mp
#from .kp_repairStrategy import ReparaStrategy as repairStrategy

class Esfera():
    def __init__(self):
        self.centro = [-5,5]
        self.radio = 200
        self.instancia = f'esfera centro {self.centro} radio {self.radio}'

    def getNombre(self):
        return 'esfera'

    
    
    def getNumDim(self):
        return 2
    def getRangoSolucion(self):
        return {'max': 100, 'min':-100}

    def evalEnc(self, encodedInstance):
        repaired, numReparaciones = self.repara(encodedInstance)
        fitness = self.evalInstance(repaired)
        return fitness, encodedInstance, numReparaciones
               
    def evalInstance(self, decoded):
        return -np.sqrt(np.power(decoded[0]-self.centro[0],2)
                       +np.power(decoded[1]-self.centro[1],2))
    
    def repara(self, solution):
        valido = self.evalInstance(solution) <= self.radio
        numReparaciones = 0
        while not valido:
            solution = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(1, self.getNumDim()))
            valido = self.evalInstance(solution) <= self.radio
            numReparaciones += 1
        return solution, numReparaciones
    
    def generarSolsAlAzar(self, numSols):
        args = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(numSols, self.getNumDim()))
        pool = mp.Pool(4)
        ret = pool.map(self.evalEnc, args)
        pool.close()
        sol = np.array([item[1] for item in ret])
        return sol