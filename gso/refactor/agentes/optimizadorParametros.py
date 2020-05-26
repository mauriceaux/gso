#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:37:01 2019

@author: mauri
"""
import numpy as np
from hmmlearn import hmm
import os
import pickle
import math

class OptimizadorParametros:

    def __init__(self):
        self.parametros = None
        self.iteraciones = 1
        self.delta = 0
        self.contRechazo = 0
        self.maxRechazo = 4
        self.contExploracion = 0
        self.maxExploracion = 3
        self.numObs = 0
        self.dataEvol = {}
        self.wmax = 0.9
        self.wmin = 0.4

        self.states = ['Exploration', 'Exploitation', 'Convergence', 'Jump out']
        if os.path.exists('hmm-model.pkl'):
            with open("hmm-model.pkl", "rb") as file: 
                self.dhmm = pickle.load(file)
        else:
            Pi = np.array([1,0,0,0])
            A = np.array([
                [0.5,0.5,0  ,0 ],
                [0  ,0.5,0.5,0 ],
                [0  ,0  ,0.5,0.5],
                [0.5,0  ,0  ,0.5]
            ])
            B = np.array([
                [0   ,0   ,0   ,0.5,0.25,0.25,0  ],
                [0   ,0.25,0.25,0.5,0   ,0   ,0  ],
                [2/3 ,1/3 ,0   ,0  ,0   ,0   ,0  ],
                [0   ,0   ,0   ,0  ,0   ,1/3 ,2/3]
            ])
            n_states = len(self.states)
            self.dhmm = hmm.MultinomialHMM(n_components=n_states)
            self.dhmm.n_features = 7
            self.dhmm.startprob_=Pi
            self.dhmm.transmat_=A
            self.dhmm.emissionprob_=B
            with open("hmm-model.pkl", "wb") as file: pickle.dump(self.dhmm, file)
    
    def setParamDim(self, paramDim):
        self.paramDim = paramDim
#        print(f'paramDim {paramDim}')

    def disEstEvol(self, estado):
        if estado < 0.2: return 0
        if 0.2 <= estado < 0.3: return 1
        if 0.3 <= estado < 0.4: return 2
        if 0.4 <= estado < 0.6: return 3
        if 0.6 <= estado < 0.7: return 4
        if 0.7 <= estado < 0.8: return 5
        if 0.8 <= estado <= 1: return 6
        
    def observarResultados(self, parametros, resultados):
        self.parametros = parametros
        self.grupos = []
        if not self.parametros['nivel']in self.dataEvol:
            self.dataEvol[self.parametros['nivel']] = {}
        if not 'mejoresResultados' in resultados:
            self.mejoraResultados = None
            self.estadoReal=1
            self.difMediaMejorEval = 0.1
        else:      
            #print(self.parametros['estEvol'])  
            
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
                self.grupos.append(idGrupo)
                if not idGrupo in self.dataEvol[self.parametros['nivel']]:
                    self.dataEvol[self.parametros['nivel']][idGrupo] = []
                self.dataEvol[self.parametros['nivel']][idGrupo].extend([self.disEstEvol(estado) for estado in self.parametros['estEvol'][idGrupo]])
                #data = np.array([[self.disEstEvol(estado) for estado in self.parametros['estEvol'][idGrupo]]]).T
                #print(data)
                #print(self.dhmm.generate( 1 ))
                
                
                #print(sum([params[name] for name in params]))

                #self.dhmm.fit( data )
                #with open("hmm-model.pkl", "wb") as file: pickle.dump(self.dhmm, file)
                #x, z = self.dhmm.sample(10)
                #print(x)
                #print(z)
                #exit()
                #logprob, predicted = self.dhmm.decode(data, algorithm="viterbi")
                #print([self.states[idx] for idx in predicted])
                #exit()
                #print(self.dhmm.b)
                #exit()
                #( log_prob, s_seq ) =  self.dhmm.viterbi( self.dhmm.b )
                #print(s_seq)
                #exit()
#                print(idGrupo)
                #prmPromMejRes = np.mean(resultados['mejoresResultadosReales'][self.parametros['nivel']][idGrupo][-self.iteraciones:-int(self.iteraciones/2)])
                #ultPromMejRes = np.mean(resultados['mejoresResultadosReales'][self.parametros['nivel']][idGrupo][-int(self.iteraciones/2):])
                #self.mejoraResultados[idGrupo] = (prmPromMejRes-ultPromMejRes)/prmPromMejRes
            params =self.dhmm._get_n_fit_scalars_per_param()
            numParams = sum([params[nombre] for nombre in params])
            totalObs = 0
            obs = []
            lenghts = []
            for idGrupo in self.dataEvol[self.parametros['nivel']]:
                obs.extend(self.dataEvol[self.parametros['nivel']][idGrupo])
                lenghts.append(len(self.dataEvol[self.parametros['nivel']][idGrupo]))
            obs = np.array([obs]).T
            lenghts = np.array(lenghts)
            if np.prod(obs.shape) >= numParams:
                print('entrenando hmm')
                self.dhmm.fit( obs, lenghts )
                with open("hmm-model.pkl", "wb") as file: pickle.dump(self.dhmm, file)
            else: print('muy pocos datos para entrenar')
            #print(obs)
            #exit()

            

            
            
    
    def mejorarParametros(self):
#        print(f"optimizador parametros inicio {self.parametros['solPorGrupo']}")
        self.parametros['numIteraciones'] = self.iteraciones
#        self.parametros['nivel'] = 2
#        return self.parametros
        
        #if self.mejoraResultados is None or not self.parametros['nivel'] in  self.parametros['inercia']:  
        #    return self.parametros
        #print(f'**********************\nESTADO REAL\n{self.mejoraResultados}\n**********************\n')
        for idGrupo in self.grupos:
    #        return self.parametros
            #print(np.array([self.dataEvol[self.parametros['nivel']][idGrupo]]).T)
            last = self.dataEvol[self.parametros['nivel']][idGrupo][-10:]
            #print(self.dataEvol[self.parametros['nivel']][idGrupo])
            #print(np.array(last).reshape((-1,1)))
            #exit()
            data = np.array(last).reshape((-1,1))
            #print(data)
            ( log_prob, estadoGrupo ) = self.dhmm.decode(data, algorithm="viterbi")
            #print(estadoGrupo)
            estado = estadoGrupo[-1]
            print(f"el estado del grupo {idGrupo} es {self.states[estado]}")
            if self.states[estado] == 'Exploration':
                self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.2,high=1.3)
                self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=.7,high=.8)
                
                self.parametros['inercia'][self.parametros['nivel']][idGrupo] = self.wmin + (self.wmax - self.wmin) * np.random.uniform()
                if self.parametros['nivel'] == 1:
                    self.parametros['solPorGrupo'][idGrupo] -= 1
            if self.states[estado] == 'Exploitation':
                self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.05,high=1.1)
                self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=.9,high=.95)
                self.parametros['inercia'][self.parametros['nivel']][idGrupo] = 1/1+(1.5*math.exp(-2.6*self.parametros['estEvol'][idGrupo][-1]))
                if self.parametros['nivel'] == 1:
                    self.parametros['solPorGrupo'][idGrupo] += 1
            if self.states[estado] == 'Convergence':
                self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.05,high=1.1)
                self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.05,high=1.1)
                self.parametros['inercia'][self.parametros['nivel']][idGrupo] = self.wmin
                if self.parametros['nivel'] == 1:
                    self.parametros['solPorGrupo'][idGrupo] += 1
            if self.states[estado] == 'Jump out':
                if self.parametros['nivel'] == 1:
                    self.parametros['solPorGrupo'][idGrupo] -= 1
                self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=.7,high=.8)
                self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.2,high=1.3)
                self.parametros['inercia'][self.parametros['nivel']][idGrupo] = self.wmax
            #exit()


            #if self.mejoraResultados[idGrupo] > 0.0:
            #    print(f"LAS SOLUCIONES MEJORAN GRUPO {idGrupo} NIVEL {self.parametros['nivel']}")
                #self.parametros['inercia'][self.parametros['nivel']][idGrupo] *= 0.9
                #self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=0.8,high=0.9)
                #self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=0.7,high=0.9)
#                self.parametros['nivel'] = 2 if self.parametros['nivel'] == 1 else 1
                #if self.parametros['nivel'] == 1:
#                    print(self.parametros['solPorGrupo'])            
                #    self.parametros['solPorGrupo'][idGrupo] -= 1
            #else:
            #    print(f"LAS SOLUCIONES NO MEJORAN GRUPO {idGrupo} NIVEL {self.parametros['nivel']}")
                #self.parametros['inercia'][self.parametros['nivel']][idGrupo] *= 1.1
                #self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.01,high=1.3)
                #self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=1.01,high=1.3)
#                self.parametros['nivel'] = 1
                #if self.parametros['nivel'] == 1:
#                    print(self.parametros['solPorGrupo'])
                #    if self.parametros['solPorGrupo'][idGrupo] >= 15:
                #        self.parametros['solPorGrupo'][idGrupo] -= 10
                #    else:
                #        self.parametros['solPorGrupo'][idGrupo] += 1
            if self.parametros['solPorGrupo'][idGrupo] > 15: self.parametros['solPorGrupo'][idGrupo] = 15
            if self.parametros['solPorGrupo'][idGrupo] < 3: self.parametros['solPorGrupo'][idGrupo] = 3
#            print(f"sols en grupo {idGrupo} {self.parametros['solPorGrupo'][idGrupo]}")
#        print("FIN MEJORA PARAMETROS")
#        print(f"optimizador parametros fin {self.parametros['solPorGrupo']}")
        self.parametros['nivel'] = 1 if self.parametros['nivel'] == 2 else 2
        #if self.parametros['nivel'] == 1: self.parametros['numIteraciones'] = 20
        #if self.parametros['nivel'] == 2: self.parametros['numIteraciones'] = 40
        return self.parametros

        

