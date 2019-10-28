#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:02:44 2019

@author: mauri
"""
import numpy as np
#import read_instance as r_instance
#import binarizationstrategy as _binarization
#import reparastrategy as _repara

class ProblemTest():
    def __init__(self, instancePath = None):
#        self.instance = r_instance.Read(instancePath)
##        print(f'self.instance.get_r() {self.instance.get_r()}')
##        exit()
#        self.tTransferencia = "sShape1"
#        self.tBinary = "Standar"
        self.minimize = True
        self.radius = 501
        
    def get_columns(self):
        return 2
    
    def evalEnc(self, encodedInstance):
#        decoded, incumplidas = self.decodeInstance(encodedInstance)
#        fitness = self.evalInstance(decoded) * (incumplidas+1)
        reparada,_  = self.repara(encodedInstance)
        fitness = self.evalInstance(reparada)
        
        return fitness, reparada
#    
#    def decodeInstance(self, encodedInstance):
##        time.sleep(0.1)
##        print(f'encodedInstance203. {list(encodedInstance)}')
##        exit()
#        b = self.binarize(list(encodedInstance))
##        repair = _repara.ReparaStrategy()
#        
#        encodedInstance, repairNum = self.repara(b.get_binary())
#        return encodedInstance, repairNum
        
#        incumplidas = repair.incumplidas(b.get_binary(), self.instance.get_r(),self.instance.get_rows(),self.instance.get_columns())
#        return b.get_binary(), incumplidas
    
#    def binarize(self, x):
#        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
    
    def evalInstance(self, decoded):
        
#        time.sleep(0.1)
#        suma = np.sqrt(np.power(decoded[0],2)) + np.sqrt(np.power(decoded[1],2)) + np.sqrt(np.power(decoded[2],2))
        suma = np.power((decoded[0]),2) +np.power((decoded[1]),2) 
#        +np.power((decoded[2]),2)
        
        
#        suma = np.sum(decoded)
        if self.minimize: return -suma
        return suma
#        return self.fObj(decoded, self.instance.get_c())
    
#    def fObj(self, pos,costo):
##        print(f'pos {pos} \ncosto {costo}')
##        exit()
#        return np.sum(np.array(pos) * np.array(costo))
  
    def repara(self,x):
        suma = np.power((x[0]),2) +np.power((x[1]),2) 
#        +np.power((x[2]),2)
        radius2 = np.power(self.radius, 2)
        
#        print(f'encoded {x}')
##        print(f'reparada {reparada}')
#        print(f'suma {suma}')
#        print(f'radius2 {radius2}')
#        exit()
#        print(f'x {x}')
        repairNum = 0
        if suma >= radius2:
#            print(x)
#            print(suma)
#            print(np.power(self.radius,2))
#            if np.power(x[0],2) > radius2: x[0] = self.radius/2
#            if np.power(x[1],2) > radius2: x[1] = self.radius/2
#            if np.power(x[2],2) > radius2: x[2] = self.radius/2
            x[0]=500
            x[1]=0
#            x[2]=0
            
#            suma = np.power((x[0]),2) +np.power((x[1]),2) +np.power((x[2]),2)
            repairNum += 1
#            print(x)
#            print(suma)
#            exit()
        return x, repairNum