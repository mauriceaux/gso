#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:02:44 2019

@author: mauri
"""
import numpy as np
import read_instance as r_instance
import binarizationstrategy as _binarization
import reparastrategy as _repara
from datetime import datetime
#import line_profiler

class Problem():
    def __init__(self, instancePath = None):
        print(f'LEYENDO INSTANCIA')
        self.instance = r_instance.Read(instancePath)
        print(f'FIN LEYENDO INSTANCIA')
        if(self.instance.columns != np.array(self.instance.get_c()).shape[0]):
            raise Exception(f'self.instance.columns {self.instance.columns} != np.array(self.instance.get_c()).shape[1] {np.array(self.instance.get_c()).shape[1]})')
#        self.repara = _repara.ReparaStrategy(self.instance.get_r())
#        self.repara.m_restriccion = np.array(self.instance.get_r())
#        self.repara.m_costos = np.array(self.instance.get_c())
#        self.repara.genImpRestr()
#        print(f'self.instance.get_r() {np.array(self.instance.get_r())[:,0]}')
#        print(f'self.instance.get_r() {np.sum(np.array(self.instance.get_r()),axis=0)}')
#        
#        np.savetxt(f"resultados/restriccion.csv", np.array(self.instance.get_r()), fmt='%d', delimiter=",")
#        exit()
        self.tTransferencia = "sShape1"
        self.tBinary = "Standar"
        self.minimize = True
        
        self.repair = _repara.ReparaStrategy(self.instance.get_r()
                                            ,self.instance.get_c()
                                            ,self.instance.get_rows()
                                            ,self.instance.get_columns())
    
#    @profile
    def evalEncMod(self, encodedInstance):
#        print(f'encodedInstance.shape {np.array(encodedInstance).shape}')
        decoded = self.decodeInstanceMod(encodedInstance)
#        fitness = self.evalInstance(decoded) * (incumplidas+1)
        
        fitness = self.evalInstance(decoded)
##        if self.repair.cumple(x) != 1:
#            fitness *= -1
        return fitness, decoded

    def evalEnc(self, encodedInstance):
#        print(f'encodedInstance.shape {np.array(encodedInstance).shape}')
#        start = datetime.now()
        decoded = self.decodeInstance(encodedInstance)
#        end = datetime.now()
#        decTime = end-start
        
#        fitness = self.evalInstance(decoded) * (incumplidas+1)
#        start = datetime.now()
        fitness = self.evalInstance(decoded)
#        if self.repair.cumple(decoded) != 1:
#            fitness *= fitness
#        end = datetime.now()
#        fitTime = end-start
#        print(f'decoding time {decTime} fitness time {fitTime}')
        return fitness, decoded
    
    def encodeInstance(self, decodedInstance, minVal, maxVal):
        decodedInstance[decodedInstance==1] = maxVal
        decodedInstance[decodedInstance==0] = minVal
        return decodedInstance

#    @profile
    def decodeInstance(self, encodedInstance):
#        time.sleep(0.1)
#        print(f'encodedInstance {list(encodedInstance)}')
#        exit()
        start = datetime.now()
        b = self.binarize(list(encodedInstance))
        end = datetime.now()
        binTime = end-start
#        return b.get_binary()
#        repair = _repara.ReparaStrategy()
#        start = datetime.now()
        encodedInstance = self.frepara(b.get_binary())
#        end = datetime.now()
#        repairTime = end-start
##        print(f'binarization time {binTime} repair time {repairTime}')
#        #print(f'repara {end-start}')
        return encodedInstance
        
#        incumplidas = repair.incumplidas(b.get_binary(), self.instance.get_r(),self.instance.get_rows(),self.instance.get_columns())
#        return b.get_binary(), incumplidas
    def decodeInstanceMod(self, encodedInstance):
#        time.sleep(0.1)
#        print(f'encodedInstance {list(encodedInstance)}')
#        exit()
        b = self.binarize(list(encodedInstance))
#        repair = _repara.ReparaStrategy()
#        start = datetime.now()
        encodedInstance = self.reparaMod(b.get_binary())
#        end = datetime.now()
#        print(f'repara mod {end-start}')
        return encodedInstance
        
#        incumplidas = repair.incumplidas(b.get_binary(), self.instance.get_r(),self.instance.get_rows(),self.instance.get_columns())
#        return b.get_binary(), incumplidas
#    @profile
    def binarize(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
    
    def binarizeMod(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary).get_binary()
    
#    @profile
    def evalInstance(self, decoded):
#        time.sleep(0.1)
        if self.minimize: return -(self.fObj(decoded, self.instance.get_c()))
        return self.fObj(decoded, self.instance.get_c())
    
#    @profile
    def fObj(self, pos,costo):
#        print(f'pos {pos} \ncosto {costo}')
#        exit()
        return np.sum(np.array(pos) * np.array(costo))
  
#    @profile
    def frepara(self,x):
        start = datetime.now()
        cumpleTodas=0
#        repair = _repara.ReparaStrategy()
#        matrizRestriccion = self.instance.get_r()
#        matrizCosto = self.instance.get_c()
#        r = self.instance.get_rows()
#        c = self.instance.get_columns()
        cumpleTodas=self.repair.cumple(x)
        if cumpleTodas == 1: return x
        x = self.repair.repara_one(x)    
        end = datetime.now()
#        print(f'repara one {end-start}')
        cumpleTodas = self.repair.cumple(x)
        if cumpleTodas == 1: return x
        x = self.repair.repara_two(x)    
        end = datetime.now()
#        print(f'repara two {end-start}')
        return x
    
    def reparaMod(self,x):
        #start = datetime.now()
#        cumpleTodas=0
        repair = self.repara
#        repairNum = 0
        r = self.instance.get_rows()
        c = self.instance.get_columns()
        
#        cumpleTodas, _=repair.cumpleModificado(x,self.instance.get_r(),r,c)
        x = repair.repara_oneModificado(x,self.instance.get_r(),self.instance.get_c(),r,c)    
#        print(f'sol original\n {x}\n cumple todas {cumpleTodas}')
#        if cumpleTodas==0:
#            x = repair.repara_oneModificado(x,self.instance.get_r(),self.instance.get_c(),r,c)    
#            repairNum += 1
#            print(f'repara_one {x}')
        
#        cumpleTodas, inc = repair.cumpleModificado(x,self.instance.get_r(),r,c)
#        if repair.cumple(x,self.instance.get_r(),r,c) < cumpleTodas:
#            print(f'solucion {x}')
#            print(f'cumple todas {cumpleTodas} cumple todas original {repair.cumple(x,self.instance.get_r(),r,c)}')
#            print(f'inc {inc}')
#            exit()
#        print(f'primera reparacion\n {x}\n cumple todas {cumpleTodas}')
#        if cumpleTodas==0:
#            x = repair.repara_two(x,self.instance.get_r(),r,c)    
#            repairNum += 1
#            print(f'repara_two {x}')
            
#        cumpleTodas, _ = repair.cumpleModificado(x,self.instance.get_r(),r,c)
#        print(f'cumple todas {cumpleTodas} cumple todas original {repair.cumple(x,self.instance.get_r(),r,c)}')
#        print(f'segunda reparacion\n {x}\n cumple todas {cumpleTodas}')
#        exit()
        end = datetime.now()
        #print(f'reparamod demoro {end-start}')
        return x
