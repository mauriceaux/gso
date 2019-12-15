#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:37:01 2019

@author: mauri
"""

class OptimizadorParametros:
    
    def setParamDim(self, paramDim):
        self.paramDim = paramDim
        print(f'paramDim {paramDim}')
        
    def observarResultados(self, parametros, resultados):
        print(f'parametros {parametros}')
        print(f'resultados {resultados}')
        pass
    
    def mejorarParametros(self):
        pass

        

