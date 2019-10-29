#!/usr/bin/python
# encoding=utf8
"""
Created on Tue Apr 23 19:05:37 2019

@author: cavasquez
"""
import math
import random
import numpy as np
from datetime import datetime
class ReparaStrategy:
    def repara_oneModificado(self,solucion, m_restriccion,m_costos,r,c):
        incumplidas = self.cumpleModificado(solucion, m_restriccion, r, c)

        

#        print(np.argmax(np.sum(np.array(incumplidas),axis=0)))
#        exit()
        solucion = np.array(solucion)
        ultIncumplidas = len(incumplidas)
#        np_m_restriccion = np.array(m_restriccion)
        np_mcostos = np.array(m_costos)
        i =0
        while len(incumplidas) > 0:
            sumaFila = np.sum(incumplidas, axis=1)
            sumaCols = np.sum(incumplidas, axis=0)
            #print(sumaFila)
            #print(np.array(incumplidas))
            #print((np.array(incumplidas)*sumaFila.reshape((sumaFila.shape[0],1))).shape)
            #print(sumaCols)
            #print((np.array(incumplidas).T*sumaCols.reshape((sumaCols.shape[0],1))).T.shape)
            #exit()
            iFilas = np.array(incumplidas)*sumaFila.reshape((sumaFila.shape[0],1))
            iCols  = (np.array(incumplidas).T*sumaCols.reshape((sumaCols.shape[0],1))).T
            importanciaIncumplidas = iFilas + iCols
            print(np.where(importanciaIncumplidas>0))
            exit()
            #print(sumaFila)
            #exit()
            idxFila = np.argmax(sumaFila)
            #print(idxFila)
            #exit()
            indices = np.where(incumplidas[idxFila] > 0)
            #print(indices)
            #print(sumaCols[indices])
            #print(np_mcostos[indices])
            #print(np.argmax(sumaCols[indices]/np_mcostos[indices]))
            #print(indices[0][np.argmin(np_mcostos[indices])])
            #exit()
            #idxReemplazar = indices[0][np.argmin(np_mcostos[indices])]
            #print(np_mcostos[indices]/sumaCols[indices])
            #exit()
            #idxReemplazar = indices[0][np.argmin(np_mcostos[indices]/sumaCols[indices])]
            idxReemplazar = indices[0][np.argmax(sumaCols[indices]/np_mcostos[indices])]


            #indices = np.argmax(np.sum(np.array(incumplidas),axis=0))
            #idxReemplazar = indices
            #print(indices)
            #exit()


#            start = datetime.now()
            #indices = np.where(np.sum(np.array(incumplidas),axis=0) > 0)
#            end = datetime.now()
#            print(f'linea 23 demoro {end-start}')
#            start = datetime.now()
            #costos = np_mcostos[indices]
#            end = datetime.now()
#            print(f'linea 27 demoro {end-start}')
#            start = datetime.now()
            #sumas = np.sum(np.array(incumplidas),axis=0)[indices]
#            end = datetime.now()
#            print(f'linea 31 demoro {end-start}')
#            start = datetime.now()
            #division = costos/sumas
#            end = datetime.now()
#            print(f'linea 35 demoro {end-start}')
#            start = datetime.now()
            #n_costos = np.zeros(np_mcostos.shape)
#            end = datetime.now()
#            print(f'linea 39 demoro {end-start}')
#            start = datetime.now()
            #n_costos[indices] = division
#            end = datetime.now()
#            print(f'linea 43 demoro {end-start}')
#            start = datetime.now()
            #idxReemplazar = np.min(np.where(n_costos>0))
#            end = datetime.now()
#            print(f'linea 47 demoro {end-start}')
#            start = datetime.now()
            if(solucion[idxReemplazar] >= 1):
                raise Exception(f'ya reemplazado solucion[{idxReemplazar}] = {solucion[idxReemplazar]}')
            solucion[idxReemplazar] = 1
#            end = datetime.now()
#            print(f'linea 51-53 demoro {end-start}')
#            start = datetime.now()
            incumplidas = self.cumpleModificado(solucion, m_restriccion, r, c)
#            end = datetime.now()
#            print(f'linea 57 demoro {end-start}')
#            start = datetime.now()
            if ultIncumplidas < len(incumplidas):
                print(f'no disminuyen las incumplidas! {ultIncumplidas} > {len(incumplidas)}')
                exit()
            ultIncumplidas = len(incumplidas)
#            end = datetime.now()
#            print(f'linea 61-64 demoro {end-start}')
            
            #exit()
            i+=1
        print(f'total reparaciones {i}')
        return solucion
        
        
    def repara_one(self,solucion, m_restriccion,m_costos,r,c):
        #print("***********Solicion no reparada************")		
        aListColRestr = {} #lista de columnas restricciones, con su respectivo índice
        aListFilRestr = {} #lista de filas restricciones, con su respectivo índice
        aListU = [] #lista que contemdrá todas las filas que violan resticcion
        aListW = []
        
        for i in range(r): #recorre, por cada fila, todas las columnas, y aquellas que son restrictivas las guarda
            aListU.append(i)
            aListTemp = []
            for j in range(c):
                if (m_restriccion[i][j] == 1):
                    aListTemp.append(j)					
            aListColRestr[i] = aListTemp #//para la fila i, todas las columnas con 1--> asigna lista de índices de columnas con valor 1 en la restricción i
        
        
#        listOfCoordinates= list(zip(result[0], result[1]))
        

        # iterate over the list of coordinates
#        for cord in listOfCoordinates:
#            print(cord)
#        print(f'aListColRestr {aListColRestr}')
#        print(f'result {result}')
#        exit()
        
        for j in range(c): #recorre, por cada columna, todas las filas, y aquellas que son restrictivas las guarda
            aListTemp = []
            for i in range(r):
                if (m_restriccion[i][j] == 1):
                    aListTemp.append(i)
            aListFilRestr[j] = aListTemp #para la columna j, todas las filas con 1 --> asigna lista de índices de filas con valor 1 en la restricción i
            
        for i in range(r):
            for j in range(c):
                if (solucion[j] * m_restriccion[i][j] == 1): #si en la posicion j no viola restricción, se elimina el id de la fila de la lista 
                    if (i in aListU):
                        aListU.remove(i)
                    break
                
        #result = np.where(np.array(m_restriccion) == 1)
#        print(aListU)
#        exit()
#        suma = np.sum(m_restriccion, axis = 0)
#        print(suma)
#        print(np.array(m_restriccion)[:,0])
#        exit()
        
        hashSet = set(aListU)
        aListU=[]
        aListU = hashSet
        
#        print(f'aListU {len(aListU)}')
#        print(f'aListFilRestr {aListColRestr}')
#        exit()
        
        while len(aListU) > 0: #MIENTRAS QUEDEN COLUMNAS POR CORREGIR
            nFila = 0
            for fila in aListU:
                nFila = fila
                break
				
            nColumnSel = self.columnaMenorCosto(aListColRestr[nFila], m_restriccion, m_costos) #busca la columna de mayor ajuste (la que tenga mas opciones de ser reemplazada)
            solucion[nColumnSel] = 1

            for nFilaDel in aListFilRestr[nColumnSel]:
                if (nFilaDel in aListU):
                    aListU.remove(nFilaDel) #DADO QUE CORREGÍ ARRIBA, QUITO LA FILA DE LA LISTA --> borra la fila completa, pues tiene otra columna que la resuelve
        
        #LUEGO DE CORREGIR, VALIDAMOS CUÁNTAS FILAS QUEDAN SIN RESTRICCION POR CADA COLUMNA
        contFila = 0;
        for i in range(r):
            contFila = 0;
            for j in range(c):
                if (solucion[j] * m_restriccion[i][j] == 1):
                    contFila+=1
            aListW.append(contFila) #se agregan tantos elementos como filas con 1 existan en el nido
			
        aListU = []
        aNumRow = []
        bComp = 0
        for j in range(len(solucion)-1,-1,-1):
            bComp = 0
            aNumRow = []
            if (solucion[j] == 1):
                for i in range(r):
                    if (m_restriccion[i][j] == 1):
                        if (aListW[i] >= 2): #si la fila tiene más de dos alternativas, se guarda su índice
                            aNumRow.append(i) #agrega el número de la fila al arreglo
                            bComp = 1
                        else:
                            bComp = 0
                            break

                if (bComp==1):
                    for i in  aNumRow: #para todas aquellas filas que tenían más de una solución, se les resta una solución
                        #aListW.set(i, aListW.get(i) - 1)
                        aListW[i] = aListW[i] - 1
                    solucion[j] = 0 #y el valor del nido se deja en cero (chanchamente a cero)
                    #print("cambiando el valor en la posicion:"+str(j))
        #print("***********Solicion modificada************")
        return solucion
		
    def repara_two(self,solucion,m_restriccion,r,c):
        for i in range(r):
            nRC=-1
            for j in range(c):
                if (m_restriccion[i][j] == 1):
                    if (solucion[j] == 0):
                        if (nRC == -1):
                            nRC = j
                        else:
                            nRC = -1
                            break
                if (nRC != -1):
                    solucion[nRC] = 1      
        return solucion
        
    def columnaMenorCosto(self,arrayList,restricciones,costos):
        nValor = 0
        nValorTemp = 0
        nFila = 0
        cont = 0

        for nColumna in arrayList:
            sum = 0
            for i in range(0,len(restricciones)):
                if (restricciones[i][nColumna] == 1):
                    sum+=1

            if (cont == 0):
                nValor = costos[nColumna] / sum
                nValorTemp = costos[nColumna] / sum
                nFila = nColumna
            else:
                nValorTemp = costos[nColumna] / sum
			
            if (nValorTemp < nValor):
                nValor = nValorTemp
                nFila = nColumna
            
            cont+=1
        return nFila

    def find(self, target, myList):
        for i in range(len(myList)):
            if myList[i] == target:
                yield i
        
    def cumpleModificado(self, solucion, m_restriccion, r, c):
        start = datetime.now()
        incumplidas = []
        m_restriccion = np.array(m_restriccion)
        solucion = np.array(solucion)
#        print(list(self.find(1,m_restriccion[0])))
#        exit()
        incumplidas = [item for item in m_restriccion if np.sum(solucion[list(self.find(1,item))]) < 1]
        #incumplidas = [item for item in m_restriccion if np.sum(solucion[np.where(item == 1)]) < 1]
        #for restr in m_restriccion:
            
        #    suma = np.sum(solucion[np.where(restr == 1)])
        #    if suma < 1:
        #        incumplidas.append(restr)
        #        break
        end = datetime.now()
        #print(f'cumplemod demoro {end-start}')
        return incumplidas    
#        if len(incumplidas) >  0: return 0, incumplidas
#        if len(incumplidas) == 0: return 1, incumplidas
    
    
    def cumple(self, solucion, m_restriccion, r, c):
#        print(f'inicio cumple')
        start = datetime.now()
        cumpleTodas = 0
        SumaRestriccion = 0
        for i in range(r): 
#            print(f'i = {i}')
            for j in range(c):
#                print(f'j = {j}')
#                if j == 0:
#                print(f'm_restriccion[{i}][{j}] = {m_restriccion[i][j]} and solucion[{j}] = {solucion[j]} ')
                if m_restriccion[i][j] == 1 and solucion[j] == 1:
                    SumaRestriccion+=1
#                    print(f'SumaRestriccion = {SumaRestriccion}')
                    break
#            print(f'SumaRestriccion-1 = {SumaRestriccion-1} i = {i}')
            if i!=SumaRestriccion-1:
                break
            #else:
                #print('cumple')
#        print(f'SumaRestriccion = {SumaRestriccion} r = {r}')
        
        if SumaRestriccion == r:
            cumpleTodas = 1 
#        print(f'cumpleTodas = {cumpleTodas}')
#        print(f'fin cumple')
#        exit()
        end = datetime.now()
        #print(f'cumple demoro {end-start}')
        return cumpleTodas
    
    def incumplidas(self, solucion, m_restriccion, r, c):
#        cumpleTodas = 0
        SumaRestriccion = 0
        for i in range(r): 
            for j in range(c):
                if m_restriccion[i][j] == 1 and solucion[j] == 1:
                    SumaRestriccion+=1
                    break
            if i!=SumaRestriccion-1:
                break
            #else:
                #print('cumple')
#        if SumaRestriccion == r:
#            cumpleTodas = 1 
        return r-SumaRestriccion
#        return cumpleTodas
            
