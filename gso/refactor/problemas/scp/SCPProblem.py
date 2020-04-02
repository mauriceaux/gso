#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:02:44 2019

@author: mauri
"""
import numpy as np
from . import read_instance as r_instance
from . import binarizationstrategy as _binarization
#import reparastrategy as _repara
#from .repair import ReparaStrategy2 as _repara
from .repair import ReparaStrategy as _repara
from graficos.Graficador import Graficador
from datetime import datetime
import multiprocessing as mp
#from numpy.random import default_rng
from sympy.combinatorics import Permutation
#import matplotlib.pyplot as plt
#import line_profiler

class SCPProblem():
    def __init__(self, instancePath = None):
#        print(f'LEYENDO INSTANCIA')
        self.instancia = instancePath
        self.instance = r_instance.Read(instancePath)
#        print(f'FIN LEYENDO INSTANCIA')
        if(self.instance.columns != np.array(self.instance.get_c()).shape[0]):
            raise Exception(f'self.instance.columns {self.instance.columns} != np.array(self.instance.get_c()).shape[1] {np.array(self.instance.get_c()).shape[1]})')
#        self.repara = _repara.ReparaStrategy(self.instance.get_r())
#        self.repara.m_restriccion = np.array(self.instance.get_r())
#        self.repara.m_costos = np.array(self.instance.get_c())
#        self.repara.genImpRestr()
#        print(f'self.instance.get_r() {np.array(self.instance.get_r())[:,0]}')
#        print(f'self.instance.get_r() {np.sum(np.array(self.instance.get_r()),axis=0)}')
#        
#        np.savetxt(f"resultados/restriccion.csv", np.array(self.instance.get_r()), fmt='%d', delimiter=",")
#        exit()
        self.tTransferencia = "sShape1"
#        self.tTransferencia = "vShape2"
        self.tBinary = "Standar"
#        self.tBinary = "invStandar"
        self.binarizationStrategy = _binarization.BinarizationStrategy(self.tTransferencia, self.tBinary)        
        self.repair = _repara.ReparaStrategy(self.instance.get_r()
                                            ,self.instance.get_c()
                                            ,self.instance.get_rows()
                                            ,self.instance.get_columns())
        self.paralelo = False
        self.penalizar = False
        self.mejorSolHist = np.ones((self.instance.get_columns())) * 0.5
#        self.graficador = None
#        self.gParam = Graficador()
#        self.gVels = Graficador()
#        self.line = None
#        self.graficador.inicio()
#        self.ax = plt.gca()
#        plt.ion()
#        plt.show()
   
    def initGrafico(self):
        self.graficador = Graficador()
        id1 = 'soluciones'
        id2 = 'velocidades'
        id3 = 'paramVel'
        self.graficador.add_plot(id1)
        self.graficador.add_plot(id2)
        self.graficador.add_plot(id3)
        self.graficador.createPlot()

    def getNombre(self):
        return 'SCP'
    
    def getMaximize(self):
        return False
#    def getNombre(self):
#        return self.instancia
    
    def getNumDim(self):
        return self.instance.columns

    def getRangoSolucion(self):
#        return {'max': 5, 'min':-5}
        return {'max': 1.5, 'min':-10}

    def eval(self, encodedInstance):
        decoded, numReparaciones = self.frepara(encodedInstance)
        fitness = self.evalInstance(encodedInstance)
        return fitness, decoded, numReparaciones

    def evalEnc(self, encodedInstance):
        if not self.penalizar:
            decoded, numReparaciones = self.decodeInstance(encodedInstance)
        else:
            decoded = self.binarizationStrategy.binarize(encodedInstance)
            numReparaciones = 0
        fitness = self.evalInstance(decoded)
        return fitness, decoded, numReparaciones

    def evalEncBatch(self, encodedInstances, mejorSol):
        decoded, numReparaciones = self.decodeInstancesBatch(encodedInstances, mejorSol)
        fitness = self.evalInstanceBatch(decoded)
        return fitness, decoded, numReparaciones
    
    def evalDecBatch(self, encodedInstances, mejorSol):
#        print(f'encodedInstance.shape {np.array(encodedInstance)}')
#        exit()
#        start = datetime.now()
#        fitness = []
#        decoded = []
#        numReparaciones = []
#        for encodedInstance in encodedInstances:
#                a,b,c = self.evalEnc(encodedInstance)
#                fitness.append(a)
#                decoded.append(b)
#                numReparaciones.append(c)
        
        
#        decoded, numReparaciones = self.decodeInstancesBatch(encodedInstances, mejorSol)
        fitness = self.evalInstanceBatch(encodedInstances)
        
        
        return fitness, encodedInstances, None
    
    def encodeInstance(self, decodedInstance):
        assert decodedInstance.shape[0] == self.getNumDim()
        idx = np.ones((self.getNumDim())) * -1
        cont = 0
        for i in range(decodedInstance.shape[0]):
            if decodedInstance[i] == 1:
                idx[i] = cont
                cont += 1
        totalUnos = cont
        for i in range(decodedInstance.shape[0]):
            if decodedInstance[i] == 0:
                idx[i] = cont
                cont += 1
        
        p = Permutation(idx)
        rank = p.rank()
        return np.array([totalUnos, rank])
#        decodedInstance[decodedInstance==1] = self.getRangoSolucion()['max']
#        decodedInstance[decodedInstance==0] = self.getRangoSolucion()['min']
#        return decodedInstance

#    @profile
        
    def decodeInstancesBatch(self, encodedInstances, mejorSol):
        start = datetime.now()
        b = self.binarizationStrategy.binarizeBatch(encodedInstances, mejorSol)
        end = datetime.now()
        binTime = end-start
#        encodedInstance, numReparaciones = self.freparaBatch(b)
#        print(b.shape)
#        exit()
        numReparaciones = 0
        repaired = self.repair.reparaBatch(b)
        return repaired, numReparaciones
    
#    def decodeInstance(self, encodedInstance):
#        encodedInstance, numReparaciones = self.frepara(decoded)
#        return encodedInstance, numReparaciones
    
    def decodeInstance(self, encodedInstance):
        start = datetime.now()
        arr = np.array(encodedInstance)
        assert arr.shape[0] != 2
        unrank = Permutation.unrank_lex(self.getNumDim(),arr[1])
        idx = np.array([i^unrank for i in range(unrank.size)])
        aux = np.zeros((self.getNumDim()))
        print(arr[0])
        aux[:arr[0]] = 1
        decoded = np.ones((self.getNumDim())) * -1
        
        for i in range(self.getNumDim()):
            decoded[i] = 1 if idx < arr[0] else 0
        return decoded
#        encodedInstance, numReparaciones = self.frepara(decoded)
#        return encodedInstance, numReparaciones
        
        
#        encodedInstance, numReparaciones = self.frepara(decoded)
#        return encodedInstance, numReparaciones
        
        
        
#        b = self.binarizationStrategy.binarize(encodedInstance)
#        end = datetime.now()
#        binTime = end-start
#        encodedInstance, numReparaciones = self.frepara(b)
#        return encodedInstance, numReparaciones
        
    def binarize(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
   
#    @profile
    def evalInstance(self, decoded):
#        time.sleep(0.1)
        return -(self.fObj(decoded, self.instance.get_c())) if self.repair.cumple(decoded) == 1 else -10000000000
    
    def evalInstanceBatch(self, decoded):
        start = datetime.now()
        #ret = np.apply_along_axis(self.fObj, -1, decoded)
#        print(np.array(self.instance.get_c()).shape)
#        print(decoded.shape)
#        print(np.sum(decoded*np.array(self.instance.get_c()),axis=1))
#        exit()
        ret = np.sum(np.array(self.instance.get_c())*decoded, axis=1)
#        print(-ret)
#        exit()
        #print(ret)
        #print(ret.shape)
        #exit()
        end = datetime.now()
        #print(f'evalInstanceBatch demoro {end-start}')
        return -ret
    
#    @profile
    def fObj(self, pos,costo):
#        print(f'pos {pos} \ncosto {costo}')
#        exit()
        return np.sum(np.array(pos) * np.array(costo))
  
#    @profile
    def freparaBatch(self,x):
        start = datetime.now()
        print(x.shape)
        exit()
        end = datetime.now()
    
    
    def frepara(self,x):
#        print(f'frepara {x}')
        start = datetime.now()
        cumpleTodas=0
#        repair = _repara.ReparaStrategy()
#        matrizRestriccion = self.instance.get_r()
#        matrizCosto = self.instance.get_c()
#        r = self.instance.get_rows()
#        c = self.instance.get_columns()
        cumpleTodas=self.repair.cumple(x)
        if cumpleTodas == 1: return x, 0
        
        x, numReparaciones = self.repair.repara_one(x)    
        end = datetime.now()
#        print(f'repara one {end-start}')
#        cumpleTodas = self.repair.cumple(x)
#        if cumpleTodas == 1: return x
#        x = self.repair.repara_two(x)    
#        end = datetime.now()
#        print(f'repara two {end-start}')
        return x, numReparaciones
    
    def generarSolsAlAzar(self, numSols, mejorSol=None):
#        args = []
        if mejorSol is None:
#            args = np.ones((numSols, self.getNumDim()), dtype=np.float) * self.getRangoSolucion()['max']
            args = np.ones((numSols, self.getNumDim()), dtype=np.float) * self.getRangoSolucion()['min']
#            args = np.ones((numSols, self.getNumDim()), dtype=np.float) * 0.1
#            args = np.zeros((numSols, self.getNumDim()), dtype=np.float)
        else:
            self.mejorSolHist = (mejorSol+self.mejorSolHist)/2
#            print(f'self.mejorSolHist {self.mejorSolHist}')
#            mejorSol = self.mejorSolHist
            args = np.repeat(np.array(self.mejorSolHist)[None, :], numSols, axis=0)
#            for i in range(numSols):
#                for j in range(args.shape[1]):
#                    args[i,j] = (self.getRangoSolucion()['min'] 
#                                    if np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max']) > args[i,j] 
#                                    else  self.getRangoSolucion()['max'])
#                    args[i,j] = (1
#                                    if np.random.uniform() < args[i,j] 
#                                    else  0)
#                print(args[i])
#                rng = default_rng()
#                unos = np.where(mejorSol>0)[0]
#                if len(unos) > 0:
#                    idxVariar = rng.choice(unos, size=(int((unos.shape[0]*.03)+1)), replace=False)
#                    args[i,idxVariar] = self.getRangoSolucion()['min']
#                args[i,idxVariar] *= np.random.uniform(-1,1) 
#            args = np.repeat(np.array(mejorSol)[None, :], numSols, axis=0)
#            args[args == 1] = self.getRangoSolucion()['max']
#            args[args == 0] = self.getRangoSolucion()['min']
#            print(f'idxVariar {idxVariar.shape}')
#            print(f'args {args.shape}')
            
#            args[np.arange(mejorSol.shape[0]),idxVariar] *= np.random.uniform(-1,1) * np.random.uniform()
#            exit()
#        fitness,sol,_ = self.evalEncBatch(args, args[0])
#        sol[sol==0] = self.getRangoSolucion()['min']
#        sol[sol==1] = self.getRangoSolucion()['max']
#        print(fitness)
#        exit()
#        args = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(numSols, self.getNumDim()))
#        print(args)
#        exit()
#        print(f'args {args}')
        fitness = []
        ant = self.penalizar
        self.penalizar = False
        if self.paralelo:
            pool = mp.Pool(4)
            ret = pool.map(self.evalEnc, args.tolist())
            pool.close()
            fitness =  np.array([item[0] for item in ret])
            sol = np.array([item[1] for item in ret])
#            sol[sol==0] = self.getRangoSolucion()['min']
#            sol[sol==1] = self.getRangoSolucion()['max']
#        print(sol)
#        exit()
        else:
            sol = []
            for arg in args:
###            print(len(arg))
                sol_ = np.array(self.evalEnc(arg)[1])
                fitness_ = np.array(self.evalEnc(arg)[0])
#                sol_[sol_==0] = self.getRangoSolucion()['min']
#                sol_[sol_==1] = self.getRangoSolucion()['max']
##            print(sol_)
##            exit()
                sol.append(sol_)
                fitness.append(fitness_)
            sol = np.array(sol)
            fitness = np.array(fitness)
        self.penalizar = ant
#        print(sol)
#        exit()
#        print(f'fin doluciones al azar')
#        exit()
#        divisor = 9
#        sol[sol==0] = self.getRangoSolucion()['min']/divisor
#        sol[sol==1] = self.getRangoSolucion()['max']/divisor
#        print(sol)
#        exit()
        return sol, fitness
    
    def graficarSol(self, datosNivel, parametros, nivel, id = 0):
        if not hasattr(self, 'graficador'):
#        if self.graficador is None:
            self.initGrafico()
#        x = np.arange(datosNivel['soluciones'].shape[1])
#        mejorGrupo = datosNivel['mejorSolGrupo'][datosNivel['grupos'][0]]
#        self.graficador.live_plotter(np.arange(mejorGrupo.shape[0]),mejorGrupo)
#        y = np.mean(datosNivel['soluciones'], axis=0)
        y = datosNivel['soluciones'][0]
#        vels = np.mean(datosNivel['velocidades'], axis=0)
        vels = datosNivel['velocidades'][0]
#        y = np.mean(datosNivel['soluciones'],axis=0)
        self.graficador.live_plotter(np.arange(y.shape[0]),y, 'soluciones', dotSize=0.1, marker='.')
        self.graficador.live_plotter(np.arange(vels.shape[0]), vels, 'velocidades', dotSize=0.1, marker='.')
        self.graficador.live_plotter(np.arange(parametros.shape[0]), parametros, 'paramVel', dotSize=1.5, marker='-')
        
#        self.line = self.graficador.setData(self.line)
        