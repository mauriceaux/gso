#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:52:47 2019

@author: mauri
"""

class ContenedorParametrosAlgoritmoIterator():
    
    def __init__(self,contenedor):
        self._contenedor = contenedor
        self.index = 0
        
    def __next__(self):
        
        