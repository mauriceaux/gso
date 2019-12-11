#!/usr/bin/python
# encoding=utf8
"""
Created on Tue Apr 23 19:05:37 2019

@author: cavasquez
"""
#import math
#import random
from threading import Lock
lock = Lock()
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class ReparaStrategy:
    def __init__(self, numItems, capacidad, valores, pesos):
        self.numItems = numItems
        self.capacidad = capacidad
        self.valores = valores
        self.pesos = pesos
        ponderacion = valores/(pesos+1)
        
        scaler = MinMaxScaler()
        ponderacion = scaler.fit_transform(ponderacion.reshape(-1,1)).reshape(self.valores.shape[0])
        
        self.ponderacionItems = 1-ponderacion
#        for i in range(self.ponderacionItems.shape[0]): print(f'linea {i} peso {self.pesos[i]} valor {self.valores[i]} ponderacion {self.ponderacionItems[i]}')
#        print(ponderacion)
#        print(self.ponderacionItems)
#        exit()
        
        
    def cumple(self, solucion):
        pesoSolucion = np.sum(solucion*self.pesos)
        if pesoSolucion > self.capacidad: return False
        return True
    
    def repara(self, solucion):
#        print(np.argpartition((solucion*self.ponderacionItems), -1)[-1:])
        cont = 0
        while not self.cumple(solucion):
            idxEliminar = np.random.choice(np.argpartition((solucion*self.ponderacionItems), -5)[-5:])
            solucion[idxEliminar] = 0.0
            cont += 1
        return solucion, cont
#        print(f'idx eliminar {idxEliminar}')