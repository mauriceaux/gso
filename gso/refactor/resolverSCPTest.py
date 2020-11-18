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
import configparser
from sqlalchemy.sql import text

if __name__ == '__main__':
    carpeta = 'problemas/scp/instances'
    carpetaResultados = 'resultados/scp'
    config = configparser.ConfigParser()
    config.read('db_config.ini')
    host = config['postgres']['host']
    db_name = config['postgres']['db_name']
    port = config['postgres']['port']
    user = config['postgres']['user']
    pwd = config['postgres']['pass']

    engine = db.create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{db_name}')
    metadata = db.MetaData()
    connection = engine.connect()
    datosEjecucion = db.Table('datos_ejecucion', metadata, autoload=True, autoload_with=engine)
    resultadoEjecucion = db.Table('resultado_ejecucion', metadata, autoload=True, autoload_with=engine)
    insertDatosEjecucion = datosEjecucion.insert().returning(datosEjecucion.c.id)
    insertResultadoEjecucion =resultadoEjecucion.insert()
    sql = text("""update datos_ejecucion set estado = 'ejecucion', inicio = :inicio
                    where id = 
                    (select id from datos_ejecucion
                        where estado = 'pendiente'
                        and nombre_algoritmo = 'GSO-ORIGINAL'
                        order by id asc
                        limit 1) returning id, parametros;""")
    
    while True:
        inicio = datetime.now()
        arrResult = connection.execute(sql,**{"inicio":inicio}).fetchone()
        if arrResult is None: 
            break
        idEjecucion = arrResult[0]
        
        param = json.loads(arrResult[1])
        print(param)
        try :
            archivo = param['instancia']
            paramOptimizar = param['paramOptimizar']
            path = os.path.join(carpeta, param['instancia'])
            problema = SCPProblem(path)
            gso = GSO(niveles=2, idInstancia=idEjecucion, numParticulas=50, iterPorNivel={1:50, 2:250}, gruposPorNivel={1:10,2:10}, dbEngine=engine)
            gso.carpetaResultados = carpetaResultados
            gso.instancia = archivo
            gso.mostrarGraficoParticulas = False
            gso.procesoParalelo = False
            gso.setProblema(problema)
        
            solver = Solver()
            solver.autonomo = False
            solver.setAlgoritmo(gso)
            solver.setParamOptimizar(paramOptimizar)
            
            inicio = datetime.now()
            solver.resolverProblema()
            fin = datetime.now()
            updateDatosEjecucion = datosEjecucion.update().where(datosEjecucion.c.id == idEjecucion)
            connection.execute(updateDatosEjecucion, {'fin':fin, 'estado' : 'terminado'})
            connection.execute(insertResultadoEjecucion, {
                'id_ejecucion':idEjecucion
                ,'fitness' : int(-solver.algoritmo.indicadores["mejorObjetivo"])
                ,'inicio': inicio 
                ,'fin': fin
                ,'mejor_solucion' : json.dumps(solver.algoritmo.indicadores["mejorSolucion"].astype('B').tolist())
                })
        except Exception as error:
            updateDatosEjecucion = datosEjecucion.update().where(datosEjecucion.c.id == idEjecucion)
            connection.execute(updateDatosEjecucion, {'inicio':None,'fin':None, 'estado' : 'pendiente'})
            raise error

    print("fin")    
    exit()
    
    

