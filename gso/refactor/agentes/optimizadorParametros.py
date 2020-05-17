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
        self.iteraciones = 20
        self.delta = 0
        self.contRechazo = 0
        self.maxRechazo = 4
        self.contExploracion = 0
        self.maxExploracion = 3
        self.numObs = 0
    
    def setParamDim(self, paramDim):
        self.paramDim = paramDim
#        print(f'paramDim {paramDim}')
        
    def observarResultados(self, parametros, resultados):
        self.parametros = parametros
        if not 'mejoresResultados' in resultados:
            self.mejoraResultados = None
            self.estadoReal=1
            self.difMediaMejorEval = 0.1
        else:        
#            print(resultados['mediaResultadosReales'])
#            ultResProm = np.mean(resultados['mediaResultadosReales'][-int(self.iteraciones/2):])
#            print(f'ultResProm {ultResProm}')
#            prmResProm = np.mean(resultados['mediaResultadosReales'][-self.iteraciones:-int(self.iteraciones/2)])
#            self.estadoReal = (prmResProm-ultResProm) / prmResProm
#            self.mediaUltRes = resultados['mediaResultadosReales'][-1] 
#            self.mejorEval = resultados['mejorObjetivo']
#            self.difMediaMejorEval = abs((self.mediaUltRes-self.mejorEval)/self.mejorEval)
            self.mejoraResultados = {}
#            print(f"en setParamDim resultados['mejoresResultadosReales'] {resultados['mejoresResultadosReales']}")
            for idGrupo in resultados['mejoresResultadosReales'][self.parametros['nivel']]:
#                print(idGrupo)
                prmPromMejRes = np.mean(resultados['mejoresResultadosReales'][self.parametros['nivel']][idGrupo][-self.iteraciones:-int(self.iteraciones/2)])
                ultPromMejRes = np.mean(resultados['mejoresResultadosReales'][self.parametros['nivel']][idGrupo][-int(self.iteraciones/2):])
                self.mejoraResultados[idGrupo] = (prmPromMejRes-ultPromMejRes)/prmPromMejRes
    
    def mejorarParametros(self):
#        print(f"optimizador parametros inicio {self.parametros['solPorGrupo']}")
        self.parametros['numIteraciones'] = self.iteraciones
#        self.parametros['nivel'] = 2
#        return self.parametros
        
        if self.mejoraResultados is None or not self.parametros['nivel'] in  self.parametros['inercia']:  
            return self.parametros
        print(f'**********************\nESTADO REAL\n{self.mejoraResultados}\n**********************\n')
        for idGrupo in self.mejoraResultados:
    #        return self.parametros
            
            if self.mejoraResultados[idGrupo] > 0.0:
                print(f"LAS SOLUCIONES MEJORAN GRUPO {idGrupo} NIVEL {self.parametros['nivel']}")
                self.parametros['inercia'][self.parametros['nivel']][idGrupo] *= 0.9
                self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=0.8,high=0.9)
                self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=0.7,high=0.9)
#                self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
                if self.parametros['nivel'] == 1:
#                    print(self.parametros['solPorGrupo'])            
                    self.parametros['solPorGrupo'][idGrupo] -= 1
            else:
                print(f"LAS SOLUCIONES NO MEJORAN GRUPO {idGrupo} NIVEL {self.parametros['nivel']}")
                self.parametros['inercia'][self.parametros['nivel']][idGrupo] *= 1.1
                self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.01,high=1.3)
                self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.01,high=1.3)
#                self.parametros['nivel'] = 1
                if self.parametros['nivel'] == 1:
#                    print(self.parametros['solPorGrupo'])
                    if self.parametros['solPorGrupo'][idGrupo] >= 15:
                        self.parametros['solPorGrupo'][idGrupo] -= 10
                    else:
                        self.parametros['solPorGrupo'][idGrupo] += 1
            if self.parametros['solPorGrupo'][idGrupo] > 15: self.parametros['solPorGrupo'][idGrupo] = 15
            if self.parametros['solPorGrupo'][idGrupo] < 3: self.parametros['solPorGrupo'][idGrupo] = 3
#            print(f"sols en grupo {idGrupo} {self.parametros['solPorGrupo'][idGrupo]}")
#        print("FIN MEJORA PARAMETROS")
#        print(f"optimizador parametros fin {self.parametros['solPorGrupo']}")
        self.parametros['nivel'] = 1 if self.parametros['nivel'] == 2 else 2
        if self.parametros['nivel'] == 1: self.parametros['numIteraciones'] = 20
        if self.parametros['nivel'] == 2: self.parametros['numIteraciones'] = 40
        return self.parametros

        

