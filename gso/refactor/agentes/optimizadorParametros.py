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
        self.iteraciones = 10
        self.delta = 0
        self.contRechazo = 0
        self.maxRechazo = 4
        self.contExploracion = 0
        self.maxExploracion = 3
    
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
#            print(f"resultados['mejoresResultados'] {resultados['mejoresResultados']}")
#            print(f"resultados['mejoresResultados'][{-int(self.iteraciones/2)}:-1] {resultados['mejoresResultados'][-int(self.iteraciones/2):-1]}")
#            print(f"resultados['mejoresResultados'][{self.iteraciones}:{-int(self.iteraciones/2)}] {resultados['mejoresResultados'][-self.iteraciones:-int(self.iteraciones/2)]}")
#            exit()
            ultResProm = np.mean(resultados['mediaResultadosReales'][-int(self.iteraciones/2):-1])
            prmResProm = np.mean(resultados['mediaResultadosReales'][-self.iteraciones:-int(self.iteraciones/2)])
            self.estadoReal = (prmResProm - ultResProm) * 100 / prmResProm
#            self.mejoraResultados = np.mean(resultados['mejoresResultados'][-int(self.iteraciones/2):-1]) - np.mean(resultados['mejoresResultados'][-self.iteraciones:-int(self.iteraciones/2)])
            porcentaje = (np.mean(resultados['mejoresResultados'][-int(self.iteraciones/2):-1]) - np.mean(resultados['mejoresResultados'][-self.iteraciones:-int(self.iteraciones/2)])) * 100
            porcentaje /= np.mean(resultados['mejoresResultados'][-int(self.iteraciones/2):-1])
#            print(-porcentaje)
#            exit()
#            
            self.mejoraResultados = -porcentaje
#            self.mejoraResultados = np.mean(resultados['mejoresResultadosReales'][-self.iteraciones:-1]) - np.mean(resultados['mejoresResultadosReales'][-3*self.iteraciones:-self.iteraciones])
            
        #print(f'parametros {parametros.keys()}')
        #print(f'resultados {resultados.keys()}')

        pass
    
    def mejorarParametros(self):
        self.parametros['numIteraciones'] = self.iteraciones
#        self.parametros['accelPer'] = 0.0005 #min 
#        self.parametros['accelPer'] *= 0.9 #max
#        self.parametros['accelBest'] += 0.0001
#        self.parametros['inercia'] = 3
#        self.parametros['nivel'] = 1
        if self.mejoraResultados is None: return self.parametros
#        print(self.estadoReal)
        if self.estadoReal > 0.5:
            print(f"LAS SOLUCIONES MEJORAN")
            self.parametros['accelPer'] *= 1.1 #max
            self.parametros['accelBest'] *= 1.1
            self.parametros['inercia'] *= 0.9
        else:
            print(f"LAS SOLUCIONES NO MEJORAN")
            self.parametros['accelPer'] *= .001 #max
            self.parametros['accelBest'] *= .001
            self.parametros['inercia'] *= 1.1
        print(self.parametros['accelPer'])
#        self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
        
        if self.mejoraResultados <= self.delta:
            print('ESTANCADO')
            #if self.parametros['inercia'] > 0:
            #    self.parametros['inercia'] = -0.5
            #else:
            #    self.parametros['inercia'] = 0.5
#            self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
#            self.parametros['nivel'] = 1
#            if self.contExploracion >= self.maxExploracion:
#                self.contExploracion+=1
##                pass
#            else:
            self.parametros['nivel'] = 1
#                self.parametros['numParticulas'] += int(self.parametros['numParticulas']*0.1)   
#                if self.contRechazo <= self.maxRechazo:
    #                self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
                    
#                    if self.parametros['accelBest'] > 0:
#                        self.parametros['accelBest'] = -2.05*np.random.uniform()
#                    else:
#                        self.parametros['accelBest'] = 2.05*np.random.uniform()
#                    if self.parametros['accelBest'] > 0:
#                        self.parametros['accelPer'] = -2.05*np.random.uniform()
#                    else:
#                        self.parametros['accelPer'] = 2.05*np.random.uniform()
#                    self.parametros['accelPer'] = -1 * abs(self.parametros['accelPer'])
#                self.parametros['inercia'] *= -1.1
#                self.parametros['inercia'] *= -2
#                print(self.parametros['inercia'])
#                if self.contRechazo % 3 == 0:
#                    self.parametros['accelPer'] = abs(self.parametros['accelPer'])
#                    self.parametros['accelBest'] *= -1
#                else:
#                    self.parametros['accelPer'] *= -1
#                    self.parametros['accelBest'] = abs(self.parametros['accelBest'])
#                    self.parametros['inercia'] = np.random.uniform(low=-1, high=-1)
                    
#                    self.contRechazo = 0
#                self.contRechazo += 1
#            self.parametros['accelBest'] = -1 * abs(self.parametros['accelBest'])
#            self.parametros['accelPer'] = -1 * abs(self.parametros['accelPer'])
            
            
#            print(f"self.parametros['numParticulas'] {self.parametros['numParticulas']}")
#            self.parametros['numParticulas'] += int(self.parametros['numParticulas']*0.1)
#            if self.parametros['inercia'] > 0: self.parametros['inercia'] = -1
#            self.parametros['inercia'] = 2
#            self.parametros['accelPer'] = 0
#            self.parametros['accelBest'] *= 1.1
#            if self.parametros['accelPer'] > 0:
#                self.parametros['accelPer'] *= -1
#            if self.parametros['accelBest'] > 0:
#                self.parametros['accelBest'] *= -1
#        if self.mejoraResultados < 0:
#            print('NO MEJORA')
#            self.parametros['numParticulas'] += int(self.parametros['numParticulas']*0.1)
#            #self.parametros['inercia'] *= -1
#            self.parametros['inercia'] -= 0.8
#            #self.parametros['accelPer'] -= 0.2
#            #self.parametros['accelBest'] += 0.2
        if self.mejoraResultados > self.delta:
#            self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
            print('MEJORA')
#            self.parametros['nivel'] = 2
            self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
#            self.parametros['accelBest'] = abs(self.parametros['accelBest'])
#            self.parametros['accelPer'] = abs(self.parametros['accelPer'])
#            self.parametros['numParticulas'] -= int(self.parametros['numParticulas']*0.1)
            self.contExploracion = 0
#            self.parametros['inercia'] = 1
#            if self.contExploracion >= self.maxExploracion: self.contExploracion = 0
            
            
#            if self.parametros['inercia'] < 0: self.parametros['inercia'] *= -1
#            self.parametros['inercia'] *= 1-0.1
#            if self.parametros['accelPer'] < 0:
#                self.parametros['accelPer'] *= -1
#            if self.parametros['accelBest'] < 0:
#                self.parametros['accelBest'] *= -1
#            self.parametros['inercia'] *= -1
#            self.parametros['accelPer'] *= 0.9
#            self.parametros['accelBest'] *= 0.9 
        if self.parametros['numParticulas'] > 100: self.parametros['numParticulas'] = 100
        if self.parametros['numParticulas'] < 10: self.parametros['numParticulas'] = 10

        return self.parametros

        

