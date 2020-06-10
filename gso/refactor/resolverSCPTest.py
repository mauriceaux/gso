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
import sqlalchemy as db
import json

if __name__ == '__main__':
    carpeta = 'problemas/scp/instances'
    carpetaResultados = 'resultados/scp'
    engine = db.create_engine('postgresql://mh:mh@localhost:5432/resultados_mh')
    metadata = db.MetaData()
    connection = engine.connect()
    datosEjecucion = db.Table('datos_ejecucion', metadata, autoload=True, autoload_with=engine)
    resultadoEjecucion = db.Table('resultado_ejecucion', metadata, autoload=True, autoload_with=engine)
    insertDatosEjecucion = datosEjecucion.insert().returning(datosEjecucion.c.id)
    insertResultadoEjecucion =resultadoEjecucion.insert()
    for _ in range(31):
        for archivo in os.listdir(carpeta):
            path = os.path.join(carpeta, archivo)
            if os.path.isdir(path):
                # skip directories
                continue
            data = {
                'nombre_algoritmo' : 'GSO',
                'parametros': json.dumps({
                    'instancia' : archivo
                    #,'niveles'=2, 
                    #'numParticulas'=50, 
                    #'iterPorNivel'={1:50, 2:250}, 
                    #'gruposPorNivel'={1:12,2:12}
                }),
                'inicio' : datetime.now(),
                'estado' : 'ejecucion'
            }
            ResultProxy = connection.execute(insertDatosEjecucion,data)
            idEjecucion = ResultProxy.fetchone()[0]
            problema = SCPProblem(f'{carpeta}/{archivo}')
            gso = GSO(niveles=2, idInstancia=idEjecucion, numParticulas=50, iterPorNivel={1:50, 2:250}, gruposPorNivel={1:12,2:12})
            gso.carpetaResultados = carpetaResultados
            gso.instancia = archivo
            gso.mostrarGraficoParticulas = False
            gso.procesoParalelo = False
            gso.setProblema(problema)
        
            solver = Solver()
            solver.autonomo = True
            solver.setAlgoritmo(gso)
            
            inicio = datetime.now()
            solver.resolverProblema()
            fin = datetime.now()
            updateDatosEjecucion = datosEjecucion.update().where(datosEjecucion.c.id == idEjecucion)
            connection.execute(updateDatosEjecucion, {'fin':fin, 'estado' : 'terminado'})
            connection.execute(insertResultadoEjecucion, {
                'id_ejecucion':idEjecucion
                ,'fitness' : -solver.algoritmo.indicadores["mejorObjetivo"]
                ,'inicio': inicio 
                ,'fin': fin
                ,'mejor_solucion' : json.dumps(solver.algoritmo.indicadores["mejorSolucion"].tolist())
                })
             
            #with open(f"{carpetaResultados}{'/autonomo' if solver.autonomo else ''}/{archivo}inercia.csv", "a") as myfile:
            #    mejorSolStr = np.array2string(solver.algoritmo.indicadores["mejorSolucion"], max_line_width=10000000000000000000000, precision=1, separator=",", suppress_small=False)
            #    myfile.write(f'{solver.algoritmo.indicadores["mejorObjetivo"]},{inicio}, {fin}, {fin-inicio}, {mejorSolStr}\n')
            #with open(f"{carpetaResultados}/algoritmos/gso/{archivo}GSO.csv", "a") as myfile:
            #    myfile.write(json.dumps(solver.algoritmo.indicadores["tiempos"]))
            #with open(f"{carpetaResultados}/algoritmos/gso/{archivo}-evalsTodas.csv", "a") as myfile:
            #    myfile.write(json.dumps(solver.algoritmo.dataEvals))
    print(f'mejor resultado  {solver.getMejorResultado()}')
    print(f'mejor solucion   {solver.getMejorSolucion()}')
    print(f'tiempo ejecución {solver.getTiempoEjecucion()}')
    print(f'solucion promedio {solver.algoritmo.solPromedio}')
#    solver.graficarConvergencia()

