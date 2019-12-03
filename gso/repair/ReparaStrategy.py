#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:00:31 2019

@author: mauri
"""

#import readOrProblems as rOP
from . import solution as sl
from . import heuristic as he
from . import matrixUtility as mu
import numpy as np

class ReparaStrategy:
    
    def __init__(self, matrix, pesos, row, cols):
        
#        pesos, matrix = rOP.generaMatrix(instanceFile) #Se generan los pesos y la matrix desde la instancia a ejecutar
        matrix = np.array(matrix)
#        print(matrix.shape)
#        exit()
        self.rows = row
        self.cols = cols
        self.pesos = np.array(pesos)
        self.matrix = matrix
        self.rHeuristic = he.getRowHeuristics(matrix)
        self.dictCol = he.getColumnRow(matrix)
#        row, cols = matrix.shape #Se extrae el tamaño del problema
        self.dictcHeuristics = {}
        self.cHeuristic = []
        self.lSolution = []
        self.dict = he.getRowColumn(matrix)
    def repara_one(self, solution):    
        return self.repara(solution)
    
    def repara(self, solution):
        lSolution = [i for i in solution if i == 1] 
#        print(lSolution)
#        exit()
        lSolution, numReparaciones = sl.generaSolucion(lSolution,self.matrix,self.pesos,self.rHeuristic,self.dictcHeuristics,self.dict,self.cHeuristic,self.dictCol)
        sol = np.zeros(self.cols, dtype=np.int)
        sol[lSolution] = 1
        return sol.tolist(), numReparaciones
    
    def cumple(self, solucion):
        for i in range(self.rows): 
            if np.sum(self.matrix[i]*solucion) < 1: return 0
        return 1