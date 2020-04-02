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
#        print(f'solution {len(solution)}')
        lSolution = [i for i in range(len(solution)) if solution[i] == 1] 
#        print(f'lSolution {len(lSolution)}')
#        exit()
        lSolution, numReparaciones = sl.generaSolucion(lSolution,self.matrix,self.pesos,self.rHeuristic,self.dictcHeuristics,self.dict,self.cHeuristic,self.dictCol)
        sol = np.zeros(self.cols, dtype=np.float)
        sol[lSolution] = 1
        return sol.tolist(), numReparaciones
    
    def reparatwo(self,solucion):
        cont = 0
        while True:
            incumplidas = np.zeros(np.array(solucion).shape)
            for i in range(self.rows): 
                if np.sum(self.matrix[i]*solucion) < 1: 
                    incumplidas += self.matrix[i]-(self.matrix[i]*solucion)
            if np.sum(incumplidas) < 1: 
                return solucion, cont
            solucion[np.argmax(incumplidas/self.pesos)] = 1
            cont += 1
        return solucion, cont
    
    def cumple(self, solucion):
        cont = 0
        for i in range(self.rows): 
            if np.sum(self.matrix[i]*solucion) < 1: 
                cont += 1
        return cont