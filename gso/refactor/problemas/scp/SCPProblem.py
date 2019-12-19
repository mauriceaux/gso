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
from .repair import ReparaStrategy as _repara
from datetime import datetime
import multiprocessing as mp
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
        self.tBinary = "Standar"
        
        self.repair = _repara.ReparaStrategy(self.instance.get_r()
                                            ,self.instance.get_c()
                                            ,self.instance.get_rows()
                                            ,self.instance.get_columns())
        self.paralelo = False
   

    def getNombre(self):
        return 'SCP'
    
#    def getNombre(self):
#        return self.instancia
    
    def getNumDim(self):
        return self.instance.columns

    def getRangoSolucion(self):
        return {'max': 5, 'min':-5}

    def evalEnc(self, encodedInstance):
#        print(f'encodedInstance.shape {np.array(encodedInstance)}')
#        exit()
#        start = datetime.now()
        decoded, numReparaciones = self.decodeInstance(encodedInstance)
        
#        print(f'decodedInstance.shape {np.array(decoded)}')
#        exit()
#        end = datetime.now()
#        decTime = end-start
        
#        fitness = self.evalInstance(decoded) * (incumplidas+1)
#        start = datetime.now()
        fitness = self.evalInstance(decoded)
#        if self.repair.cumple(decoded) != 1:
#            fitness *= fitness
#        end = datetime.now()
#        fitTime = end-start
#        print(f'decoding time {decTime} fitness time {fitTime}')
        #encoded = self.encodeInstance(decoded)
        return fitness, decoded, numReparaciones
    
    def encodeInstance(self, decodedInstance):
        decodedInstance[decodedInstance==1] = self.getRangoSolucion()['max']
        decodedInstance[decodedInstance==0] = self.getRangoSolucion()['min']
        return decodedInstance

#    @profile
    def decodeInstance(self, encodedInstance):
#        time.sleep(0.1)
        #print(f'encodedInstance {encodedInstance}')
#        exit()
        start = datetime.now()
#        print(f'binarizacion')
        b = self.binarize(list(encodedInstance))
        
#        print(f'b {list(b.get_binary())}')
        end = datetime.now()
        binTime = end-start
#        return b.get_binary()
#        repair = _repara.ReparaStrategy()
#        start = datetime.now()
#        print(f'binary {b.get_binary()}')
        encodedInstance, numReparaciones = self.frepara(b.get_binary())
#        print(f'reparacion')
        
#        exit()
#        end = datetime.now()
#        repairTime = end-start
##        print(f'binarization time {binTime} repair time {repairTime}')
#        #print(f'repara {end-start}')
        return encodedInstance, numReparaciones
        
    def binarize(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
   
#    @profile
    def evalInstance(self, decoded):
#        time.sleep(0.1)
        return -(self.fObj(decoded, self.instance.get_c()))
    
#    @profile
    def fObj(self, pos,costo):
#        print(f'pos {pos} \ncosto {costo}')
#        exit()
        return np.sum(np.array(pos) * np.array(costo))
  
#    @profile
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
    
    def generarSolsAlAzar(self, numSols):
        args = np.ones((numSols, self.getNumDim()), dtype=np.float) * self.getRangoSolucion()['min']
        
        #args = np.random.uniform(low=-2, high=2, size=(numSols, self.getNumDim()))
        #print(args)
#        exit()
        if self.paralelo:
            pool = mp.Pool(4)
            ret = pool.map(self.evalEnc, args.tolist())
            pool.close()
            sol = np.array([item[1] for item in ret])
            sol[sol==0] = self.getRangoSolucion()['min']
            sol[sol==1] = self.getRangoSolucion()['max']
#        print(sol)
#        exit()
        else:
            sol = []
            for arg in args:
###            print(len(arg))
                sol_ = np.array(self.evalEnc(arg)[1])
                sol_[sol_==0] = self.getRangoSolucion()['min']
                sol_[sol_==1] = self.getRangoSolucion()['max']
##            print(sol_)
##            exit()
                sol.append(sol_)
            sol = np.array(sol)
        
#        print(f'fin doluciones al azar')
#        exit()
        return sol