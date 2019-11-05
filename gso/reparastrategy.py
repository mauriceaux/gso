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
class ReparaStrategy:
    def __init__(self, mRestriccion, mCostos, filas, columnas):
        self.m_restriccion = mRestriccion
        self.m_costos = mCostos
        self.importanciaRestricciones = None
        self.r = filas
        self.c = columnas
        self.aListColRestr = {}
        self.aListFilRestr = {}
        
        for i in range(self.r): #recorre, por cada fila, todas las columnas, y aquellas que son restrictivas las guarda
#            aListU.append(i)
            aListTemp = []
            for j in range(self.c):
                if (self.m_restriccion[i][j] == 1):
                    aListTemp.append(j)					
            self.aListColRestr[i] = aListTemp #//para la fila i, todas las columnas con 1--> asigna lista de índices de columnas con valor 1 en la restricción i
            
        for j in range(self.c): #recorre, por cada columna, todas las filas, y aquellas que son restrictivas las guarda
            aListTemp = []
            for i in range(self.r):
                if (self.m_restriccion[i][j] == 1):
                    aListTemp.append(i)
            self.aListFilRestr[j] = aListTemp #para la columna j, todas las filas con 1 --> asigna lista de índices de filas con valor 1 en la restricción i

    def genImpRestr(self):
        if self.m_restriccion is None:
            raise Exception('matriz restriccion es None')
        if self.m_costos is None:
            raise Exception('matriz costos es None')
        np_mr = np.array(self.m_restriccion)
        np_mc = np.array(self.m_costos)
        sumaFilas = np.sum(np_mr, axis=1)
        sumaCols = np.sum(np_mr, axis=0)
        iFilas = np_mr*sumaFilas.reshape((sumaFilas.shape[0],1))
        iCols  = np_mr.T*sumaCols.reshape((sumaCols.shape[0],1)).T
        #return iFilas+ iCols
        importanciaIncumplidas = iCols / (iFilas+1)
        
        importanciaIncumplidas = (importanciaIncumplidas.T/np_mc.reshape((np_mc.shape[0],1))).T
        #print(importanciaIncumplidas)
        #exit()  
        self.importanciaRestricciones = importanciaIncumplidas
        return importanciaIncumplidas


    def repara_oneModificado(self,solucion):
#        print('hola')
        self.genImpRestr()
        incumplidas = self.cumpleModificado(solucion, self.m_restriccion, self.r, self.c)
        solucion = np.array(solucion)
        while len(incumplidas) > 0:
            maximo =None
            idx = None
            idxIncumplida=None
            for i in range(len(incumplidas)):
                #print(f'self.importanciaRestricciones[{incumplidas[i]}] {self.importanciaRestricciones[incumplidas[i]]}')
                #print(f'maximo {maximo}')
                #print(f'self.importanciaRestricciones[incumplidas[i]] {self.importanciaRestricciones[incumplidas[i]]}')
                cumple = np.sum(solucion[np.where(self.m_restriccion[incumplidas[i]] > 0)]) > 0
                if cumple: continue
                maximoTmp = np.amax(self.importanciaRestricciones[incumplidas[i]])
                #print(f'maximoTmp {maximoTmp}')
                if maximo is None or maximoTmp >= maximo:
                    maximo = maximoTmp
                    idx = np.argmax(self.importanciaRestricciones[incumplidas[i]])
                    idxIncumplida=i

            
            #print(f'idx incumplidas {idxIncumplida}')
            #print(f'incumplidas {incumplidas}')
            if idxIncumplida is None: break
            if idx is not None: 
                #print(incumplidas[idxIncumplida])
                #print(self.m_restriccion[incumplidas[idxIncumplida]])
                #print(np.where(self.m_restriccion[incumplidas[idxIncumplida]]>0))
                #exit()
                #solucion = np.array(solucion)
                #solucion[np.where(self.m_restriccion[incumplidas[idxIncumplida]]>0)] = 0
                solucion[idx] = 1
            del incumplidas[idxIncumplida]
            
            #print(f'idx incumplidas {idxIncumplida}')
            #print(f'idx corregir {idx}')
            #print(f'total incumplidas {len(incumplidas)}')
            #print(f'indice modificar {idx}')
            #print(f'solucion[{idx}] {solucion[idx]}')
            
            #print(f'solucion {solucion}')
            #incumplidas = self.cumpleModificado(solucion, m_restriccion, r, c)
            #print(f'total incumplidas despues de reparacion {len(incumplidas)}')
        
        return solucion
        
        
    def repara_one(self,solucion):
#        print("***********Solicion no reparada************")		
#        print(f'repara one')
#        print(f'solucion no reparada {solucion}')
#        aListColRestr = {} #lista de columnas restricciones, con su respectivo índice
#        aListFilRestr = {} #lista de filas restricciones, con su respectivo índice
        aListU = [i for i in range(self.r)] #lista que contemdrá todas las filas que violan resticcion
        aListW = []
        
#        for i in range(self.r): #recorre, por cada fila, todas las columnas, y aquellas que son restrictivas las guarda
#            aListU.append(i)
#            aListTemp = []
#            for j in range(self.c):
#                if (self.m_restriccion[i][j] == 1):
#                    aListTemp.append(j)					
#            self.aListColRestr[i] = aListTemp #//para la fila i, todas las columnas con 1--> asigna lista de índices de columnas con valor 1 en la restricción i
        
        #print(f'alistU {aListU}')
        #print(f'aListColRestr {aListColRestr}')
        
#        listOfCoordinates= list(zip(result[0], result[1]))
        

        # iterate over the list of coordinates
#        for cord in listOfCoordinates:
#            print(cord)
#        print(f'aListColRestr {aListColRestr}')
#        print(f'result {result}')
#        exit()
        
#        for j in range(c): #recorre, por cada columna, todas las filas, y aquellas que son restrictivas las guarda
#            aListTemp = []
#            for i in range(r):
#                if (m_restriccion[i][j] == 1):
#                    aListTemp.append(i)
#            aListFilRestr[j] = aListTemp #para la columna j, todas las filas con 1 --> asigna lista de índices de filas con valor 1 en la restricción i
            
        #print(f'aListFilRestr {aListColRestr}')
        start = datetime.now()
        
#        aListUNuevo = aListU.copy()
        
#        bar = [solucion * fila  for fila in self.m_restriccion]
        bar = []
        
        try:
            solucion = np.array(solucion)
            bar = [solucion * fila  for fila in self.m_restriccion]
#            for fila in self.m_restriccion:
#                print(f'fila {type(fila)}')
#                print(f'solucion {type(solucion)}')
#                bar.append(solucion*fila)
#            solucion = np.array(solucion)
#            print(f'self.m_restriccion {np.array(self.m_restriccion).dtype}')
#            print(f'solucion {np.array(solucion).dtype}')
#            lock.acquire()
            _suma = np.sum(bar, axis=1)
#            print(f'suma {_suma}')
            indices = list(np.where(_suma>0)[0])
#            lock.release()
            
        except Exception as e:
            print(f'self.m_restriccion {np.array(self.m_restriccion).dtype}')
            print(f'solucion {np.array(solucion).dtype}')
            print(e)
            exit()
#        print(indices)
        [aListU.remove(item) for item in indices if item in aListU]
#        print(indices)
#        exit()
#        [aListUNuevo.remove(item) for item in indices if item in aListUNuevo]
#        for i in range(self.r):
#            for j in range(self.c):
#                if (solucion[j] * self.m_restriccion[i][j] == 1): #si en la posicion j no viola restricción, se elimina el id de la fila de la lista 
#                    if (i in aListU):
#                        aListU.remove(i)
#                        #print(f'aListU {aListU}')
#                    break
#        print(f'iguales? {(aListU == aListUNuevo)}')
#        exit()
        end = datetime.now()
#        print(f'duracion ciclo qlo 1 {end-start}')
#        print(f'aListU {aListU}')
                
        
        #result = np.where(np.array(m_restriccion) == 1)
#        print(aListU)
#        exit()
#        suma = np.sum(m_restriccion, axis = 0)
#        print(suma)
#        print(np.array(m_restriccion)[:,0])
#        exit()
        
#        hashSet = set(aListU)
#        aListU=[]
#        aListU = hashSet
        
        #print(f'aListU {aListU}')
#        print(f'aListU {len(aListU)}')
#        print(f'aListFilRestr {aListColRestr}')
#        exit()
        start = datetime.now()
        
#        cols = [self.columnaMenorCosto(self.aListColRestr[nFila], self.m_restriccion, self.m_costos)
#                 for nFila in aListU
#                 if np.sum(self.m_restriccion[nFila]*solucion) == 0]
#        cols = []
        for nFila in aListU:
            if np.sum(self.m_restriccion[nFila]*solucion) == 0:
                idx = self.columnaMenorCosto(self.aListColRestr[nFila], self.m_restriccion, self.m_costos)
                solucion[idx] = 1
#                cols.append(idx)
                
#        colsOriginal = []
#        while len(aListU) > 0: #MIENTRAS QUEDEN COLUMNAS POR CORREGIR
#            nFila = aListU[0]
#            
##            for fila in aListU:
##                nFila = fila
##                break
#            #print(f'nFila {nFila}')
#            
#            nColumnSel = self.columnaMenorCosto(self.aListColRestr[nFila], self.m_restriccion, self.m_costos) #busca la columna de mayor ajuste (la que tenga mas opciones de ser reemplazada)
#            colsOriginal.append(nColumnSel)
##            print(np.array(self.m_restriccion[nFila]).shape)
##            print(nColumnSel)
#
##            print(costos.shape)
##            exit()
#
##            print(np.argmin(costos[idxRestr]))
##            exit()
##            costos = np.array(self.m_restriccion[nFila])*np.array(self.m_costos)
##            idxRestr = np.where(costos>0)
##            menor = np.argmin(costos[idxRestr])
##            nColumnSel = idxRestr[0][menor]
##            print(nColumnSelNueva)
#            
##            print(np.argmin(np.array(self.m_restriccion[nFila])*np.array(self.m_costos)))
##            exit()
##            nColSelNueva = [nCol for nCol in np.array(self.m_costos)[self.aListColRestr[nFila]] if ]
#            #print(f'self.columnaMenorCosto(aListColRestr[{nFila}], m_restriccion, m_costos) {nColumnSel}')
#
#            solucion[nColumnSel] = 1
#            #print(f'solucion {solucion}')
##            aListUNuevo = aListU.copy()
##            lock.acquire()
#            [aListU.remove(nFilaDel) for nFilaDel in self.aListFilRestr[nColumnSel] if nFilaDel in aListU]
##            lock.release()
##            for nFilaDel in self.aListFilRestr[nColumnSel]:
##                if (nFilaDel in aListU):
##                    aListU.remove(nFilaDel) #DADO QUE CORREGÍ ARRIBA, QUITO LA FILA DE LA LISTA --> borra la fila completa, pues tiene otra columna que la resuelve
##            print(f'iguales? {aListU == aListUNuevo}')
#        print(cols)
#        print(colsOriginal)
#        exit()
        end = datetime.now()
#        print(f'duracion ciclo qlo 2 {end-start}')
        #LUEGO DE CORREGIR, VALIDAMOS CUÁNTAS FILAS QUEDAN SIN RESTRICCION POR CADA COLUMNA
#        contFila = 0;
        start = datetime.now()
#        print(f'{self.m_restriccion}')
        aListW = np.sum([fila * solucion for fila in self.m_restriccion], axis=1)
#        var = np.sum(self.m_restriccion, axis=0)*solucion
        
#        for i in range(self.r):
#            contFila = 0;
#            for j in range(self.c):
#                if (solucion[j] * self.m_restriccion[i][j] == 1):
#                    contFila+=1
#            aListW.append(contFila) #se agregan tantos elementos como filas con 1 existan en el nido
#        print((list(var)))
#        print((aListW))
#        print((list(var)==aListW))
        end = datetime.now()
#        print(f'duracion ciclo qlo 3 {end-start}')	
#        exit()
        #print(f'aListW {aListW}')
        aListU = []
        aNumRow = []
        bComp = 0
        start = datetime.now()
        for j in range(len(solucion)-1,-1,-1):
            bComp = 0
            aNumRow = []
            if (solucion[j] == 1):
                for i in range(self.r):
                    #print(f'm_restriccion[i] {m_restriccion[i]}')
                    if (self.m_restriccion[i][j] == 1):
                        if (aListW[i] >= 2): #si la fila tiene más de dos alternativas, se guarda su índice
                            aNumRow.append(i) #agrega el número de la fila al arreglo
                            #print(f'aNumRow {aNumRow}')
                            bComp = 1
                        else:
                            bComp = 0
                            break
                        
                

                if (bComp==1):
                    for i in  aNumRow: #para todas aquellas filas que tenían más de una solución, se les resta una solución
                        #aListW.set(i, aListW.get(i) - 1)
                        aListW[i] = aListW[i] - 1
                    solucion[j] = 0 #y el valor del nido se deja en cero (chanchamente a cero)
                    #print(f'solucion {solucion}')
                    #print("cambiando el valor en la posicion:"+str(j))
        end = datetime.now()
#        print(f'duracion ciclo qlo 4 {end-start}')
#        print(f"***********Solicion modificada************\n{solucion}")
#        print("***********Solicion modificada************")
        return solucion
		
    def repara_two(self,solucion):
        start = datetime.now()
        for i in range(self.r):
            nRC=-1
            for j in range(self.c):
                if (self.m_restriccion[i][j] == 1):
                    if (solucion[j] == 0):
                        if (nRC == -1):
                            nRC = j
                        else:
                            nRC = -1
                            break
                if (nRC != -1):
                    solucion[nRC] = 1      
        end = datetime.now()
        print(f'duracion repara two {end-start}')
        return solucion
        
    def columnaMenorCosto(self,arrayList,restricciones,costos):
        start = datetime.now()
        nValor = 0
        nValorTemp = 0
        nFila = 0
        cont = 0
        
        for nColumna in arrayList:
#            sum = 0
#            sum2 = 0
            suma=sum([1 for i in range(0,len(restricciones)) if (restricciones[i][nColumna] == 1)])
#            for i in range(0,len(restricciones)):
#                if (restricciones[i][nColumna] == 1):
#                    sum+=1
#            print(f'iguales? {sum==sum2}')
#            print(f'sum {sum} sum2 {sum2}')

            if (cont == 0):
                nValor = costos[nColumna] / suma
                nValorTemp = costos[nColumna] / suma
                nFila = nColumna
            else:
                nValorTemp = costos[nColumna] / suma
			
            if (nValorTemp < nValor):
                nValor = nValorTemp
                nFila = nColumna
            
            cont+=1
        end = datetime.now()
#        print(f'duracion columnaMenorCosto {end-start}')
        return nFila

    def find(self, target, myList):
        for i in range(len(myList)):
            if myList[i] == target:
                yield i
        
    def cumpleModificado(self, solucion, m_restriccion, r, c):
#        start = datetime.now()
        incumplidas = []
        m_restriccion = np.array(m_restriccion)
        solucion = np.array(solucion)
#        print(list(self.find(1,m_restriccion[0])))
#        exit()
        incumplidas = [i for i in range(m_restriccion.shape[0]) if np.sum(solucion[list(self.find(1,m_restriccion[i]))]) < 1]
        #incumplidas = [item for item in m_restriccion if np.sum(solucion[np.where(item == 1)]) < 1]
        #for restr in m_restriccion:
            
        #    suma = np.sum(solucion[np.where(restr == 1)])
        #    if suma < 1:
        #        incumplidas.append(restr)
        #        break
#        end = datetime.now()
        #print(f'cumplemod demoro {end-start}')
        #print(f'solucion {solucion}')
        #print(f'incumplidas {incumplidas}')
        return incumplidas    
#        if len(incumplidas) >  0: return 0, incumplidas
#        if len(incumplidas) == 0: return 1, incumplidas
    
    
    def cumple(self, solucion):
#        print(f'inicio cumple')
#        start = datetime.now()
        cumpleTodas = 0
        SumaRestriccion = 0
        for i in range(self.r): 
#            print(f'i = {i}')
            for j in range(self.c):
#                print(f'j = {j}')
#                if j == 0:
#                print(f'm_restriccion[{i}][{j}] = {m_restriccion[i][j]} and solucion[{j}] = {solucion[j]} ')
                if self.m_restriccion[i][j] == 1 and solucion[j] == 1:
                    SumaRestriccion+=1
#                    print(f'SumaRestriccion = {SumaRestriccion}')
                    break
#            print(f'SumaRestriccion-1 = {SumaRestriccion-1} i = {i}')
            if i!=SumaRestriccion-1:
                break
            #else:
                #print('cumple')
#        print(f'SumaRestriccion = {SumaRestriccion} r = {self.r}')
        
        if SumaRestriccion == self.r:
            cumpleTodas = 1 
#        print(f'cumpleTodas = {cumpleTodas}')
#        print(f'fin cumple')
#        exit()
#        end = datetime.now()
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
            
