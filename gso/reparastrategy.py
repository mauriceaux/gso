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
    def repara_oneModificado(self,solucion, m_restriccion,m_costos,r,c):
        _, incumplidas = self.cumpleModificado(solucion, m_restriccion, r, c)
#        print(np.argmax(np.sum(np.array(incumplidas),axis=0)))
#        exit()
        solucion = np.array(solucion)
        ultIncumplidas = len(incumplidas)
        np_m_restriccion = np.array(m_restriccion)
        np_mcostos = np.array(m_costos)
        while len(incumplidas) > 0:
#            print(len(incumplidas))
            idxReemplazar = np.argmax(np.sum(np.array(incumplidas),axis=0))
#            print(f'largo de la incumplidas {np.array(incumplidas).shape[0]}')
#            print(f'incumplidas1 {np.array(incumplidas)}')
#            print(f'incumplidas1 shape {np.array(incumplidas).shape}')
#            print(f'incumplidas2 {np.sum(np.array(m_restriccion)[incumplidas], axis=0).shape}')
#            print(f'incumplidas2 {np.sum(np.array(m_restriccion)[incumplidas], axis=0)}')
#            print(f'solucion {solucion}')
#            print(f'np.sum(np_m_restriccion[incumplidas], axis=0) {np.sum(np_m_restriccion[incumplidas], axis=0) > 0}')
#            no_cumplidas = np.zeros(solucion.shape)
#            no_cumplidas[np.where(np.sum(np_m_restriccion[incumplidas], axis=0) > 0)] = 1
            
#            print(f'solucion {solucion}')
#            print(f'no_cumplidas {no_cumplidas}')
#            no_cumplidas = np.where(solucion < no_cumplidas)
#            print(f'no_cumplidas {no_cumplidas}')
#            print(f'np_mcostos[no_cumplidas] {np_mcostos[no_cumplidas]}')
            
#            idxReemplazar = np.argmin(np_mcostos[no_cumplidas])
#            print(np.argmin(np_mcostos[no_cumplidas]))
#            exit()
#            print(solucion > 0 * (np.sum(np_m_restriccion[incumplidas], axis=0) > 0))
#            print(np.sum(np_m_restriccion[no_cumplidas], axis=0))
#            print(np.argmax(np.sum(np_m_restriccion[no_cumplidas], axis=0)))
#            print(solucion[np.argmax(np.sum(np_m_restriccion[no_cumplidas], axis=0))])
#            exit()
#            difContr = solucion < 
            if(solucion[idxReemplazar] >= 1):
                raise Exception(f'ya reemplazado solucion[{idxReemplazar}] = {solucion[idxReemplazar]}')
            solucion[idxReemplazar] = 1
#            difContr = np.where(solucion < np.sum(np_m_restriccion[incumplidas], axis=0))
#            print(f'difContr {difContr}')
#            print(f'np_m_resticcion[:,difContr] {np.sum(np_m_restriccion[difContr], axis=0).shape}')
#            print(f'np_m_resticcion[:,difContr] {np.argmax(np.sum(np_m_restriccion[difContr], axis=0))}')
#            maxRestrIdx = np.argmax(np.sum(np_m_restriccion[difContr], axis=0))
            
#            print(f'diferencia en contra {difContr}')
#            exit()
#            print(f'costos {np.array(m_costos)[np.where(solucion < np.sum(np.array(m_restriccion)[incumplidas], axis=0))]}')
#            print(f'costos shape {np.array(m_costos)[np.where(solucion < np.sum(np.array(m_restriccion)[incumplidas], axis=0))].shape}')
#            print(f'costos sum {np.sum(np.array(m_costos)[np.where(solucion < np.sum(np.array(m_restriccion)[incumplidas], axis=0))])}')
#            print(f'costos argmin {np.argmin(np.array(m_costos)[np.where(solucion < np.sum(np.array(m_restriccion)[incumplidas], axis=0))])}')
#            minCostIdx = np.argmin(np_mcostos[difContr])
#            print(f'pos solucion costos argmin {minCostIdx}')
#            print(f'columna seleccionada para reparar {difContr[0][minCostIdx]}')
#            print(f'valor actual solucion en col selec {solucion[difContr[0][minCostIdx]]}')
#            maxRestrIdx
#            if solucion[maxRestrIdx] == 1:
#                raise Exception("ya estaba corregido")
#            if solucion[difContr[0][minCostIdx]] == 1:
#                raise Exception("ya estaba corregido")
                
#            solucion[difContr[0][minCostIdx]] = 1
#            print(f'incumplidas3 {np.array(m_restriccion)[incumplidas].shape}')
#            print(f'incumplidas4 {np.sum(np.array(m_restriccion)[incumplidas], axis=1)}')
#            print(f'incumplidas5 {np.sum(np.array(m_restriccion)[incumplidas], axis=1).shape}')
            
#            maxid = np.argmax(np.sum(np.array(m_restriccion)[incumplidas], axis=1))
#            print(f'incumplidas6 {maxid}')
#            print(f'incumplidas7 incumplidas[{np.argmax(np.sum(np.array(m_restriccion)[incumplidas]))}, axis=1)]')
#            print(f'incumplidas7 {incumplidas[maxid]}')
#            print(f'incumplidas8 {np_m_restriccion[incumplidas[maxid]]}')
#            print(f'costos {np.array(m_costos)[np_m_restriccion[incumplidas[maxid]]]}')
            
#            print(f'incumplidas9 {np.sum(np.array(m_restriccion)[incumplidas[maxid]],axis=0)}')
            
#            print(f'incumplidas10 {np.where(np_m_restriccion[incumplidas[maxid]] == 1)}')
#            costosIdx = np.where(np_m_restriccion[incumplidas[maxid]] == 1)
            
#            costos = np.array(m_costos)[costosIdx]
#            print(f'{costos}')
#            mincost = costos[np.argmin(costos)]
#            print(f'mincost {mincost}')
#            print(f'esta seleccionado? {solucion[mincost]==1}')
            
            
#            print(f'incumplidas n {np.array(m_restriccion)[incumplidas].shape}')
#            exit()
#            print(f'solucion[{incumplidas[0]}] {solucion[incumplidas[0]]}')
#            print(f'm_restriccion[incumpleidas[0]] {np.array(m_restriccion)[incumplidas[0]]}')
#            print(f'np.sum(np.array(m_restriccion)[incumplidas],axis=1) {np.sum(np.array(m_restriccion)[incumplidas],axis=1)}')
#            exit()
#            rIdx = np.argmax(np.sum(np_m_restriccion[incumplidas], axis=1))
#            rIdx = np.random.choice(incumplidas)
#            print(f'rIdx {rIdx}')
#            print(f'm_restriccion[{rIdx},:] {np.array(np_m_restriccion)[rIdx,:]}')
#            print(f'np.where(m_restriccion[{rIdx},:] == 1) {np.where(np_m_restriccion[rIdx,:] == 1)}')
#            sIdx = np.random.choice(np.where(np_m_restriccion[rIdx,:] == 1)[0])
#            print(incumplidas)
            
#            print(f'solucion[{rIdx}] {solucion[rIdx]}')
#            solucion[rIdx] = 1
#            solucion[rIdx] = 1
            _, incumplidas = self.cumpleModificado(solucion, m_restriccion, r, c)
#            print(f'*incumplidas {incumplidas}')
#            print(f'*solucion[{incumplidas[0]}] {solucion[incumplidas[0]]}')
            if ultIncumplidas < len(incumplidas):
                print(f'{ultIncumplidas} > {len(incumplidas)}')
                exit()
            ultIncumplidas = len(incumplidas)
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
                
        result = np.where(np.array(m_restriccion) == 1)
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
        
    def cumpleModificado(self, solucion, m_restriccion, r, c):
        incumplidas = []
        m_restriccion = np.array(m_restriccion)
        solucion = np.array(solucion)
        
#        exit()
#        print(f'solucion {solucion.shape}')
#        print(f'restriccion {m_restriccion.shape}')
#        exit()
        for idx in range(m_restriccion.shape[0]):
#            print(f'restriccion {m_restriccion[idx,:].shape}')
#            exit()
#            print(f'idx {idx}')
#            print(f'm_restriccion[{idx},:] {m_restriccion[idx,:][np.where(m_restriccion[idx,:] > solucion)]}')
#            print(f'solucion {solucion[np.where(m_restriccion[idx,:] > solucion)]}')
#            print(f' m_restriccion[{idx},:]==1 {np.where(m_restriccion[idx,:]==1)}')
#            print(solucion[np.where(m_restriccion[idx,:]==1)])
#            print(np.sum(solucion[np.where(m_restriccion[idx,:]==1)]))
#            raise Exception('excepcion','')
            inc = np.zeros(solucion.shape)
            inc[np.where(m_restriccion[idx,:] > solucion)] = 1
#            print(f'np.where(m_restriccion[idx,:] > solucion) {incumplidas}')
#            print(f'solucion[{incumplidas}] {solucion[incumplidas]}')
#            print(f'inc {inc}')
           
#            exit()
            res = np.sum(inc)
            if res > 0:
#                print(res)
#                exit()
                incumplidas.append(inc)
#                print(f'len(incumplidas) {len(incumplidas)}')
        if len(incumplidas) >  0: return 0, incumplidas
        if len(incumplidas) == 0: return 1, incumplidas
    
    
    def cumple(self, solucion, m_restriccion, r, c):
#        print(f'inicio cumple')
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
            
