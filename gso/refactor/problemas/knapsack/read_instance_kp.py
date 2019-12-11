#!/usr/bin/python
# encoding=utf8
from datetime import datetime
import numpy as np
class Read():
# -*- coding: utf-8 -*-

    def __init__(self,file):
        self.LeerInstancia(file)
        
    
    def LeerInstancia(self,file):
        itemWeights = []
        itemValues = []
        with open(file, "r") as instanceFile:
            cont = 0
            for data in iter(lambda: instanceFile.readline().split(), ''):
                cont += 1
#                print(f'linea {cont} data {data}')
                if cont == 1:
                    self.numItems = int(data[0])
                    self.capacidad = int(data[1])
                    continue
                if cont > self.numItems+1: break 
                itemWeights.append(int(data[0]))
                itemValues.append(int(data[1]))
        self.itemWeights = np.array(itemWeights)
        self.itemValues = np.array(itemValues)