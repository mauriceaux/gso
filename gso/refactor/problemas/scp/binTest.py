#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:50:02 2020

@author: mauri
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def sShape1(x):
        
        try:
            return (1 / (1 + math.pow(math.e, -2 * x)))
        except OverflowError:
            print(f'x {x}\ne {math.e}')
def sShape2(x):
        return (1 / (1 + math.pow(math.e, -x)))
#-----------------------------------------------------------------------------------------------------------------------------   
def sShape3(x):
    return (1 / (1 + math.pow(math.e, -x / 2)))
#-----------------------------------------------------------------------------------------------------------------------------   
def sShape4(x):
    return (1 / (1 + math.pow(math.e, -x / 3)))
#-----------------------------------------------------------------------------------------------------------------------------   
def vShape1(x):
    return abs(erf((math.sqrt(math.pi) / 2) * x))
#-----------------------------------------------------------------------------------------------------------------------------   
def vShape2(x):
    return abs(math.tanh(x))
#-----------------------------------------------------------------------------------------------------------------------------   
def vShape3(x):
    return abs(x / math.sqrt(1 + math.pow(x, 2)))
#-----------------------------------------------------------------------------------------------------------------------------   
def vShape4(x):
    return abs((2 / math.pi) * math.atan((math.pi / 2) * x))

def erf(z):
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
            
def standard(x):
        
        if random.random() <= x:
            return 1.
        else: 
            return 0.
        
def invStandard(idx):
        if random.random() >= idx:
            return 1.
        else: 
            return 0.
        
def complement(x):
        if random.random() <= x:
            return  standard(1 -x)
        else: 
            return 0
        
def staticProbability(x, alpha):
        if alpha >= x: 
            return 0 
        elif (alpha < x and x <= ((1 + alpha) / 2)):
            return standard(x)
        else: 
            return 1
        
y = [sShape1(x) for x in np.arange(-10,1.5,0.1)]
#y = [staticProbability(x,0.5) for x in np.arange(-10,1.5,0.1)]
#z = [standard(x) for x in y]
z = [staticProbability(x,0.4)  for x in y]

plt.plot(y)
plt.plot(z)
plt.ylabel('binarizacion')
plt.show()