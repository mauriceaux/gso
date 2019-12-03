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

directory = 'resultadosFinal/'
res = []
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".csv"):
        
        data = pd.read_csv(directory+filename, header=None)
#        print(np.mean(pd.to_timedelta(data.iloc[:,-5].values).total_seconds()))
        
        
#        print(datetime.strptime(data.iloc[:,-5].values[0], '%H:%M:%S'))
#        exit()
        mejor = np.max(data.iloc[:,0].values)
#        print(mejor)
        
        peor = np.min(data.iloc[:,0].values)
#        print(peor)
        
        promedio = np.mean(data.iloc[:,0])
        standar = np.std(data.iloc[:,0].values)
#        print(promedio)
#        exit()
        tiempoMaximo = np.max(pd.to_timedelta(data.iloc[:,-5].values).total_seconds())
        tiempoMinimo = np.min(pd.to_timedelta(data.iloc[:,-5].values).total_seconds())
        tiempoMedio = np.mean(pd.to_timedelta(data.iloc[:,-5].values).total_seconds())
        devStdTiempo = np.std(pd.to_timedelta(data.iloc[:,-5].values).total_seconds())
        fila = []
        fila.append(filename)
        fila.append(mejor)
        fila.append(peor)
        fila.append(promedio)
        fila.append(standar)
        
        fila.append(tiempoMinimo)
        fila.append(tiempoMaximo)
        fila.append(tiempoMedio)
        fila.append(devStdTiempo)
        fila.append(len(data.iloc[:,-5].values))
#        print(fila)
        res.append(fila)
#        print(np.max(data.iloc[:,0].values))
#        exit()
print('***************************************')
with open(f"resultadosFinal/resumen.csv", "w") as file:
    for item in res:
        file.write(f"{item}\n")
#        print(f'{item}')