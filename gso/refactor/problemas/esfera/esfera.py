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
        self.centro = [0,0]
        self.centro1 = [50,50]
        self.centro2 = [-50,50]
        self.radio = 1000
        self.instancia = f'esfera centro {self.centro} radio {self.radio}'
        self.paralelo = False

    def getNombre(self):
        return 'esfera'

    
    
    def getNumDim(self):
        return 2
        
    def getRangoSolucion(self):
        return {'max': 1000, 'min':-1000}

    def evalEnc(self, encodedInstance):
        repaired, numReparaciones = self.repara(encodedInstance)
        fitness = self.evalInstance(repaired)
        return fitness, encodedInstance, numReparaciones

    def evalEncBatch(self, encodedInstances,mejorSol):
        fitness = []
        encodedInstance = []
        numReparaciones = []
        for encodedInstance in encodedInstances:

            a, b = self.repara(encodedInstance)
            c = self.evalInstance(a)
            fitness.append(c)
            numReparaciones.append(b)
        return np.array(fitness), encodedInstances, numReparaciones
               
    def evalInstance(self, decoded):
        suma = 0
#        print(f"decoded {decoded}")
        for i in range(len(decoded)):
            suma += (decoded[i]-self.centro[i])**2
#            suma += (decoded[i]-self.centro1[i])**2
#            suma += (decoded[i]-self.centro2[i])**2
        distancia = np.sqrt(suma)
        if distancia <100 and distancia > 30:
            return -100
        if distancia <250 and distancia > 150:
            return -310
        if distancia <450 and distancia > 350:
            return -450
        return -distancia
    
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
    
    def generarSolsAlAzar(self, numSols, mejorSol=None):
#        args = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(numSols, self.getNumDim()))
        args= np.ones((numSols, self.getNumDim()))
        
#        coord1 = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max']) if mejorSol is None else mejorSol[0] -40
#                                    ,high=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max']) if mejorSol is None else mejorSol[0] -40
        if mejorSol is None:
            args = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(numSols, self.getNumDim()))
        else:
#            print(mejorSol)
#            print(f"np.repeat(np.array([0., 1., 0.])[None, :], n, axis=0) {np.repeat(np.array(mejorSol)[None, :], numSols, axis=0)}")
#            exit()
            args = np.repeat(np.array(mejorSol)[None, :], numSols, axis=0) * (np.random.uniform(low=0.7, high=1.3))
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
#        print(f"sol {sol.shape}")
#        exit()
        
        evals = [self.evalInstance(sol[i]) for i in range(sol.shape[0])]
#        print(f"evals {np.array(evals)}")
#        print(f"evals {np.array(evals).reshape((-1,1))}")
#        exit()
        return sol, np.array(evals)