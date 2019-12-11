#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:35:28 2019

@author: mauri
"""

from solver.MHSolver import Solver
from algoritmos.gso import GSO
from problemas.esfera.esfera import Esfera
if __name__ == '__main__':
    problema = Esfera()
    gso = GSO(niveles=2, numParticulas=50, iterPorNivel={1:500, 2:500}, gruposPorNivel={1:12,2:12})
    gso.procesoParalelo = False
    gso.setProblema(problema)

    solver = Solver()
    solver.setAlgoritmo(gso)

    solver.resolverProblema()
    print(f'mejor resultado  {solver.getMejorResultado()}')
    print(f'mejor solucion   {solver.getMejorSolucion()}')
    print(f'tiempo ejecución {solver.getTiempoEjecucion()}')
    solver.graficarConvergencia()

