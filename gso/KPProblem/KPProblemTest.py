#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:16:09 2019

@author: mauri
"""

from KPProblem import Problem
import numpy as np

problem = Problem("instances/knapPI_1_100_1000_1")
print(f'numero items: {problem.instance.numItems}')

print(f'solucion shape {solucion.shape} {problem.instance.itemWeights.shape}')
solucion = problem.repairStrategy.repara(solucion)
print(problem.fObj(solucion))