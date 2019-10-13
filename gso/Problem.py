#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:02:44 2019

@author: mauri
"""
import numpy as np

class Problem():
    def evalEnc(self, encodedInstance):
        decoded = self.decodeInstance(encodedInstance)
        fitness = self.evalInstance(decoded)
        return fitness
    
    def decodeInstance(self, encodedInstance):
        return encodedInstance
    
    def evalInstance(self, decoded):
        return np.sum(decoded)