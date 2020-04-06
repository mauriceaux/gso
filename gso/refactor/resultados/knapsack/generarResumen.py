#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:38:08 2019

@author: mauri
"""

import pandas as pd
import os
import numpy as np
import datetime as dt

directory = './'
res = []
instancias = []

#mejores = []
#peores = []
#promedios = []
#std = []
#indices=[]
#ejecuciones = []

optimos = []
diferencias = []
mejores = []
peores = []
promedios = []
std = []
indices=[]
ejecuciones = []

orden = {
        'knapPI_1_100_1000_1.csv':[0,9147]
        ,'knapPI_1_200_1000_1.csv':[1,11238]
        ,'knapPI_1_500_1000_1.csv':[2,28857]
        ,'knapPI_1_1000_1000_1.csv':[3,54503]
        ,'knapPI_1_2000_1000_1.csv':[4,110625]
        ,'knapPI_1_5000_1000_1.csv':[5,276457]
        ,'knapPI_1_10000_1000_1.csv':[6,563647]
        ,'knapPI_2_100_1000_1.csv':[7,1514]
        ,'knapPI_2_200_1000_1.csv':[8,1634]
        ,'knapPI_2_500_1000_1.csv':[9,4566]
        ,'knapPI_2_1000_1000_1.csv':[10,9052]
        ,'knapPI_2_2000_1000_1.csv':[11,18051]
        ,'knapPI_2_5000_1000_1.csv':[12,44356]
        ,'knapPI_2_10000_1000_1.csv':[13,90204]
        ,'knapPI_3_100_1000_1.csv':[14,2397]
        ,'knapPI_3_200_1000_1.csv':[15,2697]
        ,'knapPI_3_500_1000_1.csv':[16,7117]
        ,'knapPI_3_1000_1000_1.csv':[17,14390]
        ,'knapPI_3_2000_1000_1.csv':[18,28919]
        ,'knapPI_3_5000_1000_1.csv':[19,72505]
        ,'knapPI_3_10000_1000_1.csv':[20,146919]
        }

for filename in os.listdir(directory):
    print(filename)
    if filename in orden:
        
        data = pd.read_csv(directory+filename, header=None)
        mejor = np.max(data.iloc[:,0].values)
        peor = np.min(data.iloc[:,0].values)
        promedio = np.mean(data.iloc[:,0].values)
        standar = np.std(data.iloc[:,0].values)
        tiempoMaximo = np.max(pd.to_timedelta(data.iloc[:,3].values).total_seconds())
        tiempoMinimo = np.min(pd.to_timedelta(data.iloc[:,3].values).total_seconds())
        tiempoMedio = np.mean(pd.to_timedelta(data.iloc[:,3].values).total_seconds())
        devStdTiempo = np.std(pd.to_timedelta(data.iloc[:,3].values).total_seconds())
#        fila = []
        indices.append(orden[filename][0])
        optimos.append(orden[filename][1])
        diferencias.append((mejor-orden[filename][1])*100/orden[filename][1])
        instancias.append(filename)
        mejores.append(mejor)
        peores.append(peor)
        promedios.append(promedio)
        std.append(standar)
        ejecuciones.append(len(data.iloc[:,3].values))
        
#        fila.append(filename)
#        fila.append(mejor)
#        fila.append(peor)
#        fila.append(promedio)
#        fila.append(standar)
        
#        fila.append(tiempoMinimo)
#        fila.append(tiempoMaximo)
#        fila.append(tiempoMedio)
#        fila.append(devStdTiempo)
#        fila.append(len(data.iloc[:,3].values))
#        res.append(fila)
print('***************************************')
frame = pd.DataFrame({'ID': indices,
                      'INST': instancias,
                      'OPTIMOS': optimos,
                      '% DIFERENCIA': diferencias,
                      'MEJORES' : mejores,
                      'PEORES' : peores,
                      'PROM': promedios,
                      'STD': std,
                      'NUM EJEC': ejecuciones})
    
frame.sort_values(by=['ID'], inplace=True)

frame.to_csv('KPresumen.csv', sep=';', decimal=',', index=False)
#with open(f"./resumen.csv", "w") as file:
#    for item in res:
#        file.write(f"{item}\n")