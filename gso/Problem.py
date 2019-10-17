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

class Problem():
    def __init__(self, instancePath = None):
        self.instance = r_instance.Read(instancePath)
#        print(f'self.instance.get_r() {self.instance.get_r()}')
#        exit()
        self.tTransferencia = "sShape1"
        self.tBinary = "Standar"
        self.minimize = True
    
    def evalEnc(self, encodedInstance):
        decoded = self.decodeInstance(encodedInstance)
        fitness = self.evalInstance(decoded)
        return fitness, decoded
    
    def decodeInstance(self, encodedInstance):
#        time.sleep(0.1)
#        print(f'encodedInstance {list(encodedInstance)}')
#        exit()
        b = self.binarize(list(encodedInstance))
        encodedInstance = self.repara(b.get_binary())
        return encodedInstance
    
    def binarize(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
    
    def evalInstance(self, decoded):
#        time.sleep(0.1)
        if self.minimize: return -(self.fObj(decoded, self.instance.get_c()))
        return self.fObj(decoded, self.instance.get_c())
    
    def fObj(self, pos,costo):
#        print(f'pos {pos} \ncosto {costo}')
#        exit()
        return np.sum(np.array(pos) * np.array(costo))
  
    def repara(self,x):
        cumpleTodas=0
        repair = _repara.ReparaStrategy()
        r = self.instance.get_rows()
        c = self.instance.get_columns()
        
        cumpleTodas=repair.cumple(x,self.instance.get_r(),r,c)
#        print(f'sol {x} cumple todas {cumpleTodas}')
        if cumpleTodas==0:
            x = repair.repara_one(x,self.instance.get_r(),self.instance.get_c(),r,c)    
#            print(f'repara_one {x}')
        
        cumpleTodas = repair.cumple(x,self.instance.get_r(),r,c)
#        print(f'sol {x} cumple todas {cumpleTodas}')
        if cumpleTodas==0:
            x = repair.repara_two(x,self.instance.get_r(),r,c)    
#            print(f'repara_two {x}')
        return x