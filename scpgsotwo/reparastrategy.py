#!/usr/bin/python
# encoding=utf8
"""
Created on Tue Apr 23 19:05:37 2019

@author: cavasquez
"""
import math
import random
import numpy as np
class ReparaStrategy:
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
        hashSet = set(aListU)
        aListU=[]
        aListU = hashSet
        
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
#                    print(f'm_restriccion[{i}][{j}]')
#                    print(f'm_restriccion[{i}][{j}] {np.array(m_restriccion).shape}')
#                    try:
                    if (m_restriccion[i][j] == 1):
                        if (aListW[i] >= 2): #si la fila tiene más de dos alternativas, se guarda su índice
                            aNumRow.append(i) #agrega el número de la fila al arreglo
                            bComp = 1
                        else:
                            bComp = 0
                            break
#                    except:
#                        print(f'm_restriccion[{i}][{j}] {np.array(m_restriccion).shape}')
##                        print(f'm_restriccion[{i}][{j}]')
##                        print(f'{np.array(m_restriccion).shape}')
##                        print(f'{len(solucion)}')
##                        print(f'{r}')
#                        exit()

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
        
    def cumple(self, solucion, m_restriccion, r, c):
        cumpleTodas = 0
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
        if SumaRestriccion == r:
            cumpleTodas = 1 
        return cumpleTodas