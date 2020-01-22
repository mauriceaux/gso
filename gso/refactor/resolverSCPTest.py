#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:35:28 2019

@author: mauri
"""

from solver.MHSolver import Solver
from algoritmos.gso import GSO
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
            gso = GSO(niveles=2, numParticulas=50, iterPorNivel={1:50, 2:250}, gruposPorNivel={1:12,2:12})
            gso.procesoParalelo = False
            gso.geometric=True
            gso.setProblema(problema)
        
            solver = Solver()
            solver.autonomo = True
            solver.setAlgoritmo(gso)
            
            inicio = datetime.now()
            solver.resolverProblema()
            fin = datetime.now()
            with open(f"{carpetaResultados}{'/autonomo' if solver.autonomo else ''}/{archivo}.csv", "a") as myfile:
                mejorSolStr = np.array2string(solver.algoritmo.indicadores["mejorSolucion"], max_line_width=10000000000000000000000, precision=1, separator=",", suppress_small=False)
                myfile.write(f'{solver.algoritmo.indicadores["mejorObjetivo"]},{inicio}, {fin}, {fin-inicio}, {mejorSolStr}\n')
            with open(f"{carpetaResultados}/algoritmos/gso/{archivo}GSO.csv", "a") as myfile:
                myfile.write(json.dumps(solver.algoritmo.indicadores["tiempos"]))
            with open(f"{carpetaResultados}/algoritmos/gso/{archivo}-evalsTodas.csv", "a") as myfile:
                myfile.write(json.dumps(solver.algoritmo.dataEvals))
    print(f'mejor resultado  {solver.getMejorResultado()}')
    print(f'mejor solucion   {solver.getMejorSolucion()}')
    print(f'tiempo ejecución {solver.getTiempoEjecucion()}')
#    solver.graficarConvergencia()

