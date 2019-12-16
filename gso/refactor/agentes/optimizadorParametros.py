#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:37:01 2019

@author: mauri
"""
import numpy as np

class OptimizadorParametros:

    def __init__(self):
        self.parametros = None
        self.iteraciones =60
    
    def setParamDim(self, paramDim):
        self.paramDim = paramDim
        print(f'paramDim {paramDim}')
        
    def observarResultados(self, parametros, resultados):
        self.parametros = parametros
        if not 'mediaResultadosReales' in resultados:
            self.mejoraResultados = True
        else:    
            self.mejoraResultados = np.mean(resultados['mediaResultadosReales'][-self.iteraciones]) < np.mean(resultados['mediaResultadosReales'][-2*self.iteraciones:-self.iteraciones])
        print(f'parametros {parametros.keys()}')
        print(f'resultados {resultados.keys()}')
        pass
    
    def mejorarParametros(self):
        self.parametros['numIteraciones'] = self.iteraciones
        if not self.mejoraResultados:
            self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
        return self.parametros

        

