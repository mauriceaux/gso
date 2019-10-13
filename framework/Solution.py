#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:22:13 2019

@author: mauri
"""
import numpy as np

class Solution():
    def __init__(self):
        self.solVec = np.zeros((1,))
        self.f = []
        self.i = []
        
    def setFitness(self, f):
        self.f = f
    def setInfact(self, i):
        self.i = i
    def getSolVec(self):
        return self.solVec