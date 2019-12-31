#!/usr/bin/python
# encoding=utf8
"""
Created on Tue Apr 23 19:05:37 2019

@author: cavasquez
"""
import math
import random
import multiprocessing as mp
import numpy as np

class BinarizationStrategy:
    def __init__(self,tecnica,binary):
        if(tecnica=="sShape1"):
            self.funcShape = self.sShape1
        if(tecnica=="sShape2"):
            self.funcShape = self.sShape2
        if(tecnica=="sShape3"):
            self.funcShape = self.sShape3
        if(tecnica=="sShape4"):
            self.funcShape = self.sShape4
        if(tecnica=="vShape1"):
            self.funcShape = self.vShape1
        if(tecnica=="vShape2"):
            self.funcShape = self.vShape2
        if(tecnica=="vShape3"):
            self.funcShape = self.vShape3
        if(tecnica=="vShape4"):
            self.funcShape = self.vShape4
        
        if(binary=="Standar"):
            self.funcBin = self.standard
        
        if(binary=="Complement"):
            self.funcBin = self.complement
        
        if (binary=="StaticProbability"):
            self.funcBin = self.staticProbability
            
        if(binary=="Elitist"):
            self.funcBin = self.elitist
        
    def binarize(self, x):
        tb = [self.funcShape(item) for item in x]        
        matrizbinaria = [self.funcBin(item) for item in tb]
        return matrizbinaria

    def binarizeBatch(self, matriz, mejorSol):
        #tb = [self.funcShape(item) for item in x]        
        #print(matriz.shape)
        #exit()
        tb = np.vectorize(self.funcShape)(matriz)
        #matrizbinaria = [self.funcBin(item) for item in tb]
        matrizbinaria = np.vectorize(self.funcBin)(tb)
        #matrizbinaria = np.ones(np.array(matriz).shape)
        #for idx in range(tb.shape[0]):
        #    for idy in range(tb.shape[1]):
        #        matrizbinaria[idx,idy] = self.funcBin(matrizbinaria[idx,idy],mejorSol[idy] )

        return matrizbinaria

        
    
    def set_binary(self,binary):
        self._b = binary
    
    def get_binary(self):
        return self._b
    
    def basic(self, x):
        if 0.5 <= x: 
           return 1 
        else:
           return 0
    
    def standard(self, x):
        if x is None: return 1
        if random.random() <= x:
            return 1
        else: 
            return 0
        
    def complement(self,x):
        if random.random() <= x:
            return  self.standard(1 -x)
        else: 
            return 0
    
    def staticProbability(self, x, alpha):
        if alpha >= x: 
            return 0 
        elif (alpha < x and x <= ((1 + alpha) / 2)):
            return self.standard(x)
        else: 
            return 1
        
    def elitist(self, x, mejorSol):
        if random.random() < x:
            return mejorSol
        else: 
            return 0
    
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape1(self,x):
        #print(x)
        #exit()
        try:
            return (1 / (1 + math.pow(math.e, -2 * x)))
        except OverflowError:
            print(f'x {x}\ne {math.e}')
            
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape2(self,x):
        return (1 / (1 + math.pow(math.e, -x)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape3(self,x):
        return (1 / (1 + math.pow(math.e, -x / 2)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape4(self,x):
        return (1 / (1 + math.pow(math.e, -x / 3)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape1(self,x):
        return abs(self.erf((math.sqrt(math.pi) / 2) * x))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape2(self,x):
        return abs(math.tanh(x))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape3(self,x):
        return abs(x / math.sqrt(1 + math.pow(x, 2)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape4(self,x):
        return abs((2 / math.pi) * math.atan((math.pi / 2) * x))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def erf(self,z):
        q = 1.0 / (1.0 + 0.5 * abs(z))
        ans = 1 - q * math.exp(-z * z - 1.26551223 + q * (1.00002368
                    + q * (0.37409196
                    + q * (0.09678418
                    + q * (-0.18628806
                    + q * (0.27886807
                    + q * (-1.13520398
                    + q * (1.48851587
                    + q * (-0.82215223
                    + q * (0.17087277))))))))))
        if z >= 0:
            return ans
        else:
            return -ans
    