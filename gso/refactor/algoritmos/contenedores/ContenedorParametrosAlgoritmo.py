#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:37:27 2019

@author: mauri
"""

class ContenedorParametrosAlgoritmo():
    def __init__(self):
        self.parametros = {}
    
    def agregarParametro(self, nombre, valor):
        self.parametros[nombre] = valor