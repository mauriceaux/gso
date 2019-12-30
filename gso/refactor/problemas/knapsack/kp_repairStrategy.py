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
        ponderacion = 1/(valores/(pesos+1))
#        ponderacion = pesos
        
#        scaler = MinMaxScaler()
#        ponderacion = scaler.fit_transform(ponderacion.reshape(-1,1)).reshape(self.valores.shape[0])
        
#        self.ponderacionItems = ponderacion
#        print(self.ponderacionItems)
#        print(self.ponderacionItems.argsort()[-5:][::-1])
#        exit()
        self.ponderacionItems = ponderacion
#        for i in range(self.ponderacionItems.shape[0]): print(f'linea {i} peso {self.pesos[i]} valor {self.valores[i]} ponderacion {self.ponderacionItems[i]}')
#        print(ponderacion)
#        print(self.ponderacionItems)
#        exit()
        
        
    def cumple(self, solucion):
        pesoSolucion = np.sum(np.array(solucion)*self.pesos)
        #print(f'pesoSolucion {pesoSolucion} capacidad {self.capacidad} cumple {pesoSolucion <= self.capacidad}')
#        print(f'capacidad {self.capacidad}')
        if pesoSolucion <= self.capacidad: return True
        return False
    
    def elegirMejorAgregar(self, solucion):
#        solucion = np.array(solucion)
        capSinUsar = self.capacidad-np.sum(solucion*self.pesos)
#        print(f'capSinUsar {capSinUsar}')
#        print(f'')
        idx0 = np.array(np.where(solucion==0.0)[0])
        np.random.shuffle(idx0)
#        print(f'len(idx0) {idx0}')
#        print(f'solucion {solucion}')
#        print(f'np.random.shuffle(idx0) {np.random.shuffle(idx0)}')
        if capSinUsar > 0 and len(idx0) > 0:
            for idx in idx0:
    #            print(f'capSinUsar {capSinUsar}')
                if self.pesos[idx] <= capSinUsar:
                    solucion[idx]=1.0
                capSinUsar = self.capacidad-np.sum(solucion*self.pesos)
#        exit()
#        print(capSinUsar)
#        exit()
        
#        for i in range(100):
#            idxProbar = np.random.choice(np.where(solucion==0)[0])
#            solucion[idxProbar] = 1.0
#            if not self.cumple(solucion): solucion[idxProbar] = 0.0
        return solucion
    
    def repara(self, solucion):
        solucion = np.array(solucion)
#        print(np.argpartition((solucion*self.ponderacionItems), -1)[-1:])
        cont = 0
#        if not self.cumple(solucion): return solucion, 1
#        else: return solucion, 0
        while not self.cumple(solucion):
#            print(np.argpartition((solucion*self.ponderacionItems), -5))
#            exit()
#            idxEliminar = np.random.choice(np.argpartition((solucion*self.ponderacionItems), -5)[-5:])
            
            idxEliminar = np.random.choice((solucion*self.ponderacionItems).argsort()[-5:][::-1])
#            idxEliminar = np.random.choice(np.where(solucion==1)[0])
#            idxEliminar = np.random.choice((solucion*(self.pesos/self.valores)).argsort()[-5:][::-1])
#            print(idxEliminar)
            solucion[idxEliminar] = 0.0
            
            
            cont += 1
#        solucion = self.elegirMejorAgregar(solucion)
        return solucion, cont
#        print(f'idx eliminar {idxEliminar}')

    def reparaBatch(self, soluciones):
        start = datetime.now()
        pesos = self.pesos
        ponderacion = self.ponderacionItems
        capacidad = self.capacidad
        NUM_FILAS = soluciones.shape[0]
        s = datetime.now()
        pesosAplicados = soluciones * pesos
        e = datetime.now()
        s = datetime.now()
        #print(f'aplicacion pesos {e-s}')
        sumaPesos = np.sum(pesosAplicados, axis=1)
        e = datetime.now()
        #print(f'suma pesos {e-s}')
        #print(sumaPesos)
        incumplidas = sumaPesos > capacidad
        while (incumplidas).any():
                #print(f'incumplidas {incumplidas}')
                #exit()
                s = datetime.now()
                solucionesPonderadas = soluciones[incumplidas] * ponderacion
                e = datetime.now()
                #print(f'ponderacion columnas {e-s}')
                #print(solucionesPonderadas)
                k=4
                s = datetime.now()
                peoresIndices = np.argpartition(-solucionesPonderadas,k,axis=1)[:,k-1::-1]
                e = datetime.now()
                #print(f'peoresIndices {e-s}')
                #print(peoresIndices)
                peoresIndicesEliminar = np.random.randint(peoresIndices.shape[1], size=peoresIndices.shape[0])

                #print(peoresIndicesEliminar)

                #print(np.arange(len(peoresIndices)))
                #exit()


                s = datetime.now()
                indicesEliminar = peoresIndices[np.arange(len(peoresIndices)), peoresIndicesEliminar]
                e = datetime.now()
                #print(f'indicesEliminar {e-s}')
                #print(indicesEliminar)

                soluciones[incumplidas,indicesEliminar] = 0
                s = datetime.now()
                pesosAplicados = soluciones * pesos
                e = datetime.now()
                #print(f'pesosAplicados {e-s}')
                #print(f'pesosAplicados {pesosAplicados}')
                s = datetime.now()
                sumaPesos = np.sum(pesosAplicados, axis=1)
                e = datetime.now()
                #print(f'sumaPesos {e-s}')
                #print(sumaPesos)
                incumplidas = sumaPesos > capacidad
        end = datetime.now()
        #print(f'reparaBatch demoro {end-start}')
        return soluciones