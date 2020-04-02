#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:35:28 2019

@author: mauri
"""

from solver.MHSolver import Solver
from algoritmos.Genetic import Genetic
from problemas.scp.SCPProblem import SCPProblem
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import json
from datetime import datetime

if __name__ == '__main__':
    carpeta = 'problemas/scp/instances'
    carpetaResultados = 'resultados/scp'
    for _ in range(31):
        for archivo in os.listdir(carpeta):
            path = os.path.join(carpeta, archivo)
            if os.path.isdir(path):
                # skip directories
                continue
            problema = SCPProblem(f'{carpeta}/{archivo}')
            genetic = Genetic(problema, maximize=False, n=20)
#            gso = GSO(niveles=2, numParticulas=50, iterPorNivel={1:50, 2:250}, gruposPorNivel={1:12,2:12})
#            gso.mostrarGraficoParticulas = False
#            gso.procesoParalelo = True
#            gso.setProblema(problema)
        
#            solver = Solver()
#            solver.autonomo = True
#            solver.setAlgoritmo(gso)
            
            inicio = datetime.now()
            genetic.optimize()
            fin = datetime.now()
            totalCostG = genetic.getBestCost()
            execTimeG = genetic.execTime
            iterationsG = genetic.iterations
            print(f"mejor resultado {totalCostG} demoro {execTimeG} num iteraciones {iterationsG}")
#    solver.graficarConvergencia()

