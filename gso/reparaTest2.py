#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:19:40 2019

@author: mauri
"""
import repair.readOrProblems as rOP
import repair.solution as sl
import repair.heuristic as he
import numpy as np

file = '/home/mauri/proyectos/mg/semestre1/autSearch/project/gso/instancesFinal/scpnre1.txt'
pesos, matrix = rOP.generaMatrix(file) #Se generan los pesos y la matrix desde la instancia a ejecutar
#print(matrix.shape)
#exit()
rHeuristic = he.getRowHeuristics(matrix)
dictCol = he.getColumnRow(matrix)
row, cols = matrix.shape #Se extrae el tamaño del problema
dictcHeuristics = {}
cHeuristic = []
lSolution = np.array([])
dict = he.getRowColumn(matrix)
lSolution = sl.generaSolucion(lSolution,matrix,pesos,rHeuristic,dictcHeuristics,dict,cHeuristic,dictCol)
print(lSolution)
sol = np.zeros(cols)
sol[lSolution] = 1
print(sol)