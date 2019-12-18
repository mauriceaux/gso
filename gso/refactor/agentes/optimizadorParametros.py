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
        self.iteraciones =10
    
    def setParamDim(self, paramDim):
        self.paramDim = paramDim
        print(f'paramDim {paramDim}')
        
    def observarResultados(self, parametros, resultados):
        self.parametros = parametros
        #if not 'mediaResultadosReales' in resultados:
        #print(resultados)
        if not 'mejoresResultados' in resultados:
            self.mejoraResultados = None
        else:    
            #self.mejoraResultados = np.mean(resultados['mediaResultadosReales'][-self.iteraciones]) > np.mean(resultados['mediaResultadosReales'][-2*self.iteraciones:-self.iteraciones])
            self.mejoraResultados = np.mean(resultados['mejoresResultados'][-self.iteraciones:-1]) > np.mean(resultados['mejoresResultados'][-2*self.iteraciones:-self.iteraciones])
            
        #print(f'parametros {parametros.keys()}')
        #print(f'resultados {resultados.keys()}')
        pass
    
    def mejorarParametros(self):
        self.parametros['numIteraciones'] = self.iteraciones
        if self.mejoraResultados is None: return self.parametros
        if not self.mejoraResultados:
            print('NO MEJORA')
            self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
            self.parametros['numParticulas'] += int(self.parametros['numParticulas']*0.2)
            self.parametros['inercia'] += 0.5
        else:
            print('MEJORA')
            self.parametros['numParticulas'] -= int(self.parametros['numParticulas']*0.1)
            self.parametros['inercia'] -= 0.5
        if self.parametros['numParticulas'] > 100: self.parametros['numParticulas'] = 100
        if self.parametros['numParticulas'] < 10: self.parametros['numParticulas'] = 10
        return self.parametros

        

