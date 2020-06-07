#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:35:28 2019

@author: mauri
"""

from solver.MHSolver import Solver
from algoritmos.gso import GSO
from problemas.rosenbrock.rosenbrock import Rosenbrock
if __name__ == '__main__':
    carpetaResultados = 'resultados/esfera'
    problema = Rosenbrock()
    gso = GSO(niveles=2, numParticulas=50, iterPorNivel={1:50, 2:250}, gruposPorNivel={1:10,2:10})
    gso.carpetaResultados = carpetaResultados
    gso.procesoParalelo = True
    gso.mostrarGraficoParticulas = False
    gso.setProblema(problema)
    gso.instancia = 'rosenbrock'

    solver = Solver()
    solver.autonomo = True
    solver.setAlgoritmo(gso)

    solver.resolverProblema()
    print(f'mejor resultado  {solver.getMejorResultado()}')
    print(f'mejor solucion   {solver.getMejorSolucion()}')
    print(f'tiempo ejecución {solver.getTiempoEjecucion()}')
    print(f'num llamadas funcion objetivo {solver.algoritmo.indicadores["numLlamadasFnObj"]}')
#    print(f'solucion promedio {solver.algoritmo.solPromedio}')
#    input("Press Enter to continue...")
    solver.graficarConvergencia()

