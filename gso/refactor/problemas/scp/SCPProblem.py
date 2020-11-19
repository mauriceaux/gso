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
#from graficos.Graficador import Graficador
from datetime import datetime
import multiprocessing as mp
from numpy.random import default_rng

from .reparacionGpu import cumpleRestricciones as reparaGpu

#import matplotlib.pyplot as plt
#import line_profiler

class SCPProblem():
    def __init__(self, instancePath = None):
#        print(f'LEYENDO INSTANCIA')
        self.instancia = instancePath
        self.instance = r_instance.Read(instancePath)
        self.optimo = self.instance.optimo
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
#        self.tTransferencia = "sShape1"
        self.tTransferencia = "sShape2"
        self.tBinary = "Standar"
#        self.tBinary = "Elitist"
#        self.tBinary = "invStandar"
        self.binarizationStrategy = _binarization.BinarizationStrategy(self.tTransferencia, self.tBinary)        
        self.repair = _repara.ReparaStrategy(self.instance.get_r()
                                            ,self.instance.get_c()
                                            ,self.instance.get_rows()
                                            ,self.instance.get_columns())
        self.paralelo = False
        self.penalizar = False
        self.mejorSolHist = np.ones((self.instance.get_columns())) * 0.5
        self.mejorFitness = None
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
    
#    def getNombre(self):
#        return self.instancia
    
    def getNumDim(self):
        return self.instance.columns

    def getRangoSolucion(self):
        return {'max': 1.5, 'min':-10}

    def eval(self, encodedInstance):
        decoded, numReparaciones = self.frepara(encodedInstance)
        fitness = self.evalInstance(encodedInstance)
        return fitness, decoded, numReparaciones

    def evalEnc(self, encodedInstance):
#        print(f'encodedInstance.shape {np.array(encodedInstance)}')
#        exit()
#        start = datetime.now()
#        if mejorSol is not None:
#            self.binarizationStrategy.mejorSol = mejorSol
        if not self.penalizar:
#            print("repara")
            decoded, numReparaciones = self.decodeInstance(encodedInstance)
#            print('fin repara')
        else:
            decoded = self.binarizationStrategy.binarize(encodedInstance)
#            print("penaliza")
            numReparaciones = 0
#        print(f'decodedInstance.shape {np.array(decoded)}')
#        exit()
#        end = datetime.now()
#        decTime = end-start
        
#        fitness = self.evalInstance(decoded) * (incumplidas+1)
#        start = datetime.now()
#        print('evaluando')
        fitness = self.evalInstance(decoded)
        if self.mejorFitness is None or fitness > self.mejorFitness:
            self.mejorFitness = fitness
            self.binarizationStrategy.mejorSol = decoded
#        print('fin evaluando')
#        if self.repair.cumple(decoded) != 1:
#            fitness *= fitness
#        end = datetime.now()
#        fitTime = end-start
#        print(f'decoding time {decTime} fitness time {fitTime}')
        #encoded = self.encodeInstance(decoded)
        return fitness, decoded, numReparaciones

    def binariceBatch(self, encodedInstances):
            decoded = [self.binarizationStrategy.binarize(encodedInstances[i]) for i in range(encodedInstances.shape[0])]
            decoded = np.array(decoded)
            #print(decoded.shape)
            #print(encodedInstances.shape)
            #exit()
            return decoded
    
    def evalEncBatch(self, encodedInstances):
        decoded = self.binariceBatch(encodedInstances)
        #print(decoded.shape)
        reparadas = reparaGpu.reparaSoluciones(decoded, self.instance.get_r(), self.instance.get_c(), self.instance.pondReparaciones)

        # reparadas = np.array([self.mejoraSolucion(sol) for sol in reparadas])
        if np.random.uniform() < 0.15:
            reparadas = self.mejoraSoluciones(reparadas)
        # reparadas = reparaGpu.reparaSoluciones(decoded, self.instance.get_r(), self.instance.get_c(), self.instance.pondReparaciones)
        # nCol = 10
        # costos = reparadas * self.instance.get_c()
        # costosMasAltos = np.argsort(-costos, axis=1)[:,:nCol]
        # indices = np.random.randint(low=0,high=costosMasAltos.shape[1],size=(costosMasAltos.shape[0],3))
        # indiceColRandom = costosMasAltos[np.arange(costosMasAltos.shape[0]).reshape((-1,1)), indices]
        # reparadas[np.arange(costos.shape[0]).reshape(-1,1), indiceColRandom] = 0
        # reparadas = reparaGpu.reparaSoluciones(reparadas, self.instance.get_r(), self.instance.get_c(), self.instance.pondReparaciones)
        #print(reparadas.shape)
        #exit()
        fitness = self.evalInstanceBatch(reparadas)
        
        
        return fitness, reparadas, 0
    
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
        decodedInstance[decodedInstance==1] = self.getRangoSolucion()['max']
        decodedInstance[decodedInstance==0] = self.getRangoSolucion()['min']
        return decodedInstance

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
    
    
    def decodeInstance(self, encodedInstance):
#        time.sleep(0.1)
        #print(f'encodedInstance {encodedInstance}')
#        exit()
        start = datetime.now()
#        print(f'binarizacion')
#        b = self.binarize(list(encodedInstance))
        b = self.binarizationStrategy.binarize(encodedInstance)
        
#        print(b)
#        exit()
        end = datetime.now()
        binTime = end-start
#        return b.get_binary()
#        repair = _repara.ReparaStrategy()
#        start = datetime.now()
#        print(f'binary {b.get_binary()}')
        encodedInstance, numReparaciones = self.frepara(b)
#        if (np.array(encodedInstance) > 1).any():
#            print(f'encodedInstance {encodedInstance}')
#        
#            exit()
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
        return -(self.fObj(decoded, self.instance.get_c())) if self.repair.cumple(decoded) == 1 else -10000000000
    
    def evalInstanceBatch(self, decoded):
        start = datetime.now()
        #ret = np.apply_along_axis(self.fObj, -1, decoded)
#        print(np.array(self.instance.get_c()).shape)
#        print(decoded.shape)
#        print(np.sum(decoded*np.array(self.instance.get_c()),axis=1))
#        exit()
        ret = np.sum(np.array(self.instance.get_c())*decoded, axis=1)
        #ret = [self.evalInstance(decoded[i] for i in range(decoded.shape[0]))]
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
        x = self.mejoraSolucion(x)
        end = datetime.now()
#        print(f'repara one {end-start}')
#        cumpleTodas = self.repair.cumple(x)
#        if cumpleTodas == 1: return x
#        x = self.repair.repara_two(x)    
#        end = datetime.now()
#        print(f'repara two {end-start}')
        return x, numReparaciones
    
    def mejoraSoluciones(self, soluciones):
        # print(f"inicio mejor soluciones")
        soluciones = np.array(soluciones)
        
        for iter in range(200):
            inicio = datetime.now()
            solucionesOriginal = soluciones.copy()
            costos = soluciones * self.instance.get_c()
            # print(f"soluciones cambia? {(soluciones!=solucionesOriginal).any()}")
            nCols = int(costos.shape[1])
            # nCols = 40
            cosOrd = np.argsort(-costos, axis=1)[:,:nCols]
            # modificado = soluciones.copy()
            # modificados = []
            modificados = np.repeat(soluciones[None,...],cosOrd.shape[1],axis=0)
            for pos in range(cosOrd.shape[1]):
                # if np.count_nonzero(costos[np.arange(costos.shape[0]).reshape((-1,1)),cosOrd[:,pos].reshape(-1,1)] == 0) > (costos.shape[1]*0.3): break
                if (costos[np.arange(costos.shape[0]).reshape((-1,1)),cosOrd[:,pos].reshape(-1,1)] == 0).all(): break
                # modificado = soluciones.copy()
                # modificado[np.arange(modificado.shape[0]).reshape(-1,1), cosOrd[:,pos].reshape(-1,1)] = 0
                # modificados.append(modificado)
                modificados[pos,np.arange(soluciones.shape[0]).reshape(-1,1), cosOrd[:,pos].reshape(-1,1)] = 0
                # fact = [self.repair.cumple(modificado[id]) for id in range(modificado.shape[0])]
                # fact = np.array(fact)
            fin = datetime.now()
            # print(f"tiempo creacion de modificados {fin-inicio}")
            pendientes = np.ones(soluciones.shape[0]) == 1
            inicio = datetime.now()
            # modificados = np.array(modificados)
            originalShape = modificados.shape
            modificados = modificados.reshape((modificados.shape[0]*modificados.shape[1], modificados.shape[2]))
            # print(modificados.shape)
            fact = reparaGpu._procesarFactibilidadGPU(modificados, np.array(self.instance.get_r()))
            fact = (fact!=0).all(axis=1)
            modificados = modificados.reshape(originalShape)
            fact = fact.reshape((originalShape[0],originalShape[1]))
            
            # print(fact.shape)
            # print((fact==True).any(axis=1).shape)
            # print(np.argwhere((fact==True).any(axis=1)))
            # exit()
            factEncontradas = np.argwhere((fact==True).any(axis=1))
            # idxFact = 0
            fin = datetime.now()
            # print(f"tiempo verificacion infactibles {fin-inicio}")
            pendientes = np.ones(soluciones.shape[0]) == 1
            inicio = datetime.now()
            for idxFact in factEncontradas:
                if np.count_nonzero(pendientes == True) == 0: break
                
                # solucionesPendientes = soluciones[pendientes]
                factPendientes = fact[idxFact[0]]

                if ((pendientes & factPendientes) == True).any():
                    modificado = modificados[idxFact[0]]
                    # print(modificado)
                    # modificadoPendiente = modificado[pendientes]
                    # print(f"Existe un factible en posición {np.argwhere(factPendientes)}")
                    # print(f"pendientes {pendientes.shape}")
                    # print(f"factPendientes {factPendientes}")
                    # print(f"factPendientes {factPendientes.shape}")
                    # print(f"soluciones[pendientes][factPendientes] {soluciones[pendientes][factPendientes].shape}")
                    # print(f"modificadoPendiente[factPendientes] {modificadoPendiente[factPendientes].shape}")
                    # A = np.array([False,False,True,False])
                    # B = np.array([True,False,True,True])
                    # print(np.argwhere(A & B))
                    # exit()



                    # print(f"antes del cambio, iguales? {(soluciones[pendientes][factPendientes]==modificadoPendiente[factPendientes]).all()}")
                    # exit()                
                
                    

                    # print(f"factibles {np.count_nonzero(factPendientes == True)} sols modificadas {modificadoPendiente[factPendientes].shape[0]}")
                    # print(soluciones[pendientes][factPendientes])
                    # print(factPendientes)
                    soluciones[np.argwhere(pendientes & factPendientes),:] = modificado[np.argwhere(pendientes & factPendientes),:]
                    # print(soluciones[pendientes][factPendientes])
                    # soluciones[pendientes][factPendientes] = 8
                    # print(f"despues del cambio, iguales? {(soluciones[pendientes][factPendientes]==modificadoPendiente[factPendientes]).all()}")
                    # exit()

                # soluciones[0,0] = 8
                
                    pendientes[pendientes] = factPendientes[pendientes]==False
                # print(f"iter {idxFact} pendientes {np.count_nonzero(pendientes == True)} soluciones cambia? {(soluciones!=solucionesOriginal).any()}")
                
                # exit()
                # idxFact += 1
            fin = datetime.now()
            # print(f"tiempo asignacion {fin-inicio}")
            # print(f"fin iteracion {iter}")
            pendientes = np.ones(soluciones.shape[0]) == 1
            if (soluciones==solucionesOriginal).all(): break
                
        # exit()
        # print(f"fin mejor soluciones")
        return soluciones

    def mejoraSolucion(self, solucion):
        solucion = np.array(solucion)
        costos = solucion * self.instance.get_c()
        cosOrd = np.argsort(costos)[::-1]
        for pos in cosOrd:
            if costos[pos] == 0: break
            modificado = solucion.copy()
            modificado[pos] = 0
            if self.repair.cumple(modificado) == 1:
                solucion = modificado
        return solucion
    
    def generarSolsAlAzar(self, numSols, mejorSol=None):
#        args = []
        if mejorSol is None:
#            args = np.ones((numSols, self.getNumDim()), dtype=np.float) * self.getRangoSolucion()['max']
#            args = np.ones((numSols, self.getNumDim()), dtype=np.float) * self.getRangoSolucion()['min']
#            args = np.random.uniform(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max']+1, size=(numSols, self.getNumDim()))
#            args = np.ones((numSols, self.getNumDim()), dtype=np.float) * 0.1
            args = np.zeros((numSols, self.getNumDim()), dtype=np.float)
        else:
            self.mejorSolHist = (mejorSol+self.mejorSolHist)/2
#            print(f'self.mejorSolHist {self.mejorSolHist}')
#            mejorSol = self.mejorSolHist
            args = []
            for i in range(numSols):
                sol = mejorSol.copy()
                idx = np.random.randint(low=0, high=sol.shape[0])
                sol[idx] = 1 if sol[idx] == 0 else 0
                args.append(sol)
            args = np.array(args)
#            args = np.repeat(np.array(self.mejorSolHist)[None, :], numSols, axis=0)
            
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


        fitness, sol,_ = self.evalEncBatch(args)


        """

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

        """



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
        