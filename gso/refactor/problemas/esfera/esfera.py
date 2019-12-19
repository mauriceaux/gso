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
        self.centro = [-5,5,-30]
        self.radio = 200
        self.instancia = f'esfera centro {self.centro} radio {self.radio}'
        self.paralelo = False

    def getNombre(self):
        return 'esfera'

    
    
    def getNumDim(self):
        return 3
        
    def getRangoSolucion(self):
        return {'max': 1000, 'min':-1000}

    def evalEnc(self, encodedInstance):
        repaired, numReparaciones = self.repara(encodedInstance)
        fitness = self.evalInstance(repaired)
        return fitness, encodedInstance, numReparaciones
               
    def evalInstance(self, decoded):
        suma = 0
        for i in range(len(decoded)):
            suma += (decoded[i]-self.centro[i])**2
        return -np.sqrt(suma)
    
    def repara(self, solution):
        valido = -self.evalInstance(solution) <= self.radio
        #print(valido)
        #exit()
        numReparaciones = 0
        while not valido:
            idx = np.random.choice(np.arange(self.getNumDim()))
            exp = 1 if solution[idx] > 0 else -1
            solution[idx] = (abs(solution[idx]) - 1) * exp
            #solution = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(self.getNumDim()))
            valido = -self.evalInstance(solution) <= self.radio
            numReparaciones += 1
        return solution, numReparaciones
    
    def generarSolsAlAzar(self, numSols):
        args = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(numSols, self.getNumDim()))
        sol = None
        if self.paralelo:
            pool = mp.Pool(4)
            ret = pool.map(self.evalEnc, args)
            pool.close()
            sol = np.array([item[1] for item in ret])
        else:
            sol = []
            for arg in args:
                sol.append(self.evalEnc(arg)[1])
            sol = np.array(sol)
        return sol