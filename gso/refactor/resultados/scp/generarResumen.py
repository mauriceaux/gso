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

optimos = []
diferencias = []
mejores = []
peores = []
promedios = []
std = []
indices=[]
ejecuciones = []
#listaArchivos = os.listdir(directory)
#listaArchivos.sort()
#print(listaArchivos)
#exit()

orden = {
        'scp41.txt.csv':[0,429]
        ,'scp42.txt.csv':[1,512]
        ,'scp43.txt.csv':[2,516]
        ,'scp44.txt.csv':[3,494]
        ,'scp45.txt.csv':[4,512]
        ,'scp46.txt.csv':[5,560]
        ,'scp47.txt.csv':[6,430]
        ,'scp48.txt.csv':[7,492]
        ,'scp49.txt.csv':[8,641]
        ,'scp410.txt.csv':[9,514]
        ,'scp51.txt.csv':[10,253]
        ,'scp52.txt.csv':[11,302]
        ,'scp53.txt.csv':[12,226]
        ,'scp54.txt.csv':[13,242]
        ,'scp55.txt.csv':[14,211]
        ,'scp56.txt.csv':[15,213]
        ,'scp57.txt.csv':[16,293]
        ,'scp58.txt.csv':[17,288]
        ,'scp59.txt.csv':[18,279]
        ,'scp510.txt.csv':[19,265]
        ,'scp61.txt.csv':[20,138]
        ,'scp62.txt.csv':[21,146]
        ,'scp63.txt.csv':[22,145]
        ,'scp64.txt.csv':[23,131]
        ,'scp65.txt.csv':[24,161]
        ,'scpa1.txt.csv':[25,253]
        ,'scpa2.txt.csv':[26,252]
        ,'scpa3.txt.csv':[27,232]
        ,'scpa4.txt.csv':[28,234]
        ,'scpa5.txt.csv':[29,236]
        ,'scpb1.txt.csv':[30,69]
        ,'scpb2.txt.csv':[31,76]
        ,'scpb3.txt.csv':[32,80]
        ,'scpb4.txt.csv':[33,79]
        ,'scpb5.txt.csv':[34,72]
        ,'scpc1.txt.csv':[35,227]
        ,'scpc2.txt.csv':[36,219]
        ,'scpc3.txt.csv':[37,243]
        ,'scpc4.txt.csv':[38,219]
        ,'scpc5.txt.csv':[39,215]
        ,'scpd1.txt.csv':[40,60]
        ,'scpd2.txt.csv':[41,66]
        ,'scpd3.txt.csv':[42,72]
        ,'scpd4.txt.csv':[43,62]
        ,'scpd5.txt.csv':[44,61]
        ,'scpnre1.txt.csv':[45,29]
        ,'scpnre2.txt.csv':[46,30]
        ,'scpnre3.txt.csv':[47,27]
        ,'scpnre4.txt.csv':[48,28]
        ,'scpnre5.txt.csv':[49,28]
        ,'scpnrf1.txt.csv':[50,14]
        ,'scpnrf2.txt.csv':[51,15]
        ,'scpnrf3.txt.csv':[52,14]
        ,'scpnrf4.txt.csv':[53,14]
        ,'scpnrf5.txt.csv':[54,13]
        ,'scpnrg1.txt.csv':[55,176]
        ,'scpnrg2.txt.csv':[56,154]
        ,'scpnrg3.txt.csv':[57,166]
        ,'scpnrg4.txt.csv':[58,168]
        ,'scpnrg5.txt.csv':[59,168]
        ,'scpnrh1.txt.csv':[60,63]
        ,'scpnrh2.txt.csv':[61,63]
        ,'scpnrh3.txt.csv':[62,59]
        ,'scpnrh4.txt.csv':[63,58]
        ,'scpnrh5.txt.csv':[64,55]

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
        diferencias.append((-mejor-orden[filename][1])*100/orden[filename][1])
        instancias.append(filename)
        mejores.append(-mejor)
        peores.append(-peor)
        promedios.append(-promedio)
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
frame.to_csv('SCPresumen.csv', sep=';', decimal=',', index=False)