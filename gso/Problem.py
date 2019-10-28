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
#import line_profiler

class Problem():
    def __init__(self, instancePath = None):
        self.instance = r_instance.Read(instancePath)
#        print(f'self.instance.get_r() {np.array(self.instance.get_r())[:,0]}')
#        print(f'self.instance.get_r() {np.sum(np.array(self.instance.get_r()),axis=0)}')
#        
#        np.savetxt(f"resultados/restriccion.csv", np.array(self.instance.get_r()), fmt='%d', delimiter=",")
#        exit()
        self.tTransferencia = "sShape1"
        self.tBinary = "Standar"
        self.minimize = True
    
#    @profile
    def evalEnc(self, encodedInstance):
#        print(f'encodedInstance.shape {np.array(encodedInstance).shape}')
        decoded, incumplidas = self.decodeInstance(encodedInstance)
#        fitness = self.evalInstance(decoded) * (incumplidas+1)
        
        fitness = self.evalInstance(decoded)
        return fitness, decoded
    
#    @profile
    def decodeInstance(self, encodedInstance):
#        time.sleep(0.1)
#        print(f'encodedInstance {list(encodedInstance)}')
#        exit()
        b = self.binarize(list(encodedInstance))
#        repair = _repara.ReparaStrategy()
        
        encodedInstance, repairNum = self.repara(b.get_binary())
        return encodedInstance, repairNum
        
#        incumplidas = repair.incumplidas(b.get_binary(), self.instance.get_r(),self.instance.get_rows(),self.instance.get_columns())
#        return b.get_binary(), incumplidas
    
#    @profile
    def binarize(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
    
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
    def repara(self,x):
        cumpleTodas=0
        repair = _repara.ReparaStrategy()
        repairNum = 0
        r = self.instance.get_rows()
        c = self.instance.get_columns()
        
#        cumpleTodas, _=repair.cumpleModificado(x,self.instance.get_r(),r,c)
        x = repair.repara_oneModificado(x,self.instance.get_r(),self.instance.get_c(),r,c)    
#        print(f'sol original\n {x}\n cumple todas {cumpleTodas}')
#        if cumpleTodas==0:
#            x = repair.repara_oneModificado(x,self.instance.get_r(),self.instance.get_c(),r,c)    
#            repairNum += 1
#            print(f'repara_one {x}')
        
        cumpleTodas, inc = repair.cumpleModificado(x,self.instance.get_r(),r,c)
        if repair.cumple(x,self.instance.get_r(),r,c) < cumpleTodas:
            print(f'solucion {x}')
            print(f'cumple todas {cumpleTodas} cumple todas original {repair.cumple(x,self.instance.get_r(),r,c)}')
            print(f'inc {inc}')
            exit()
#        print(f'primera reparacion\n {x}\n cumple todas {cumpleTodas}')
        if cumpleTodas==0:
            x = repair.repara_two(x,self.instance.get_r(),r,c)    
            repairNum += 1
#            print(f'repara_two {x}')
            
        cumpleTodas, _ = repair.cumpleModificado(x,self.instance.get_r(),r,c)
#        print(f'cumple todas {cumpleTodas} cumple todas original {repair.cumple(x,self.instance.get_r(),r,c)}')
#        print(f'segunda reparacion\n {x}\n cumple todas {cumpleTodas}')
#        exit()
        return x, repairNum
