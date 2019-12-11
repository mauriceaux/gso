import numpy as np
from scipy.cluster.vq import kmeans,vq
import multiprocessing as mp
from datetime import datetime

class GSO():
    def __init__(self, niveles=2, numParticulas=50, iterPorNivel={1:50,2:250}, gruposPorNivel={1:12,2:12}):
        self.contenedorParametros = {}
        self.contenedorParametros['niveles'] = niveles
        self.contenedorParametros['numParticulas'] = numParticulas
        self.contenedorParametros['iterPorNivel'] = iterPorNivel
        self.contenedorParametros['gruposPorNivel'] = gruposPorNivel
        self.contenedorParametros['rangoSolucion'] = {'min' : -3}
        self.contenedorParametros['rangoSolucion']['max'] = 3
        self.contenedorParametros['datosNivel'] = {}
        self.contenedorParametros['mejorEvalGlobal'] = None
        self.contenedorParametros['mejorSolucion'] = None
        self.contenedorParametros['mejorSolucionBin'] = None
        self.contenedorParametros['mejorSolGlobal'] = None
        self.contenedorParametros['accelPer'] = 2.05*np.random.uniform()
        self.contenedorParametros['accelBest'] = 2.05*np.random.uniform()
        self.contenedorParametros['maxVel'] = 2
        self.contenedorParametros['minVel'] = -2
        self.procesoParalelo = False
        self.indicadores = {}
        self.mejorEval = None
        
        pass
    
    def getIndicadores(self):
        return self.indicadores
    
    def getParametros(self):
        return self.contenedorParametros
    
    def setParametros(self, parametros):
        for key in parametros:
            self.contenedorParametros[key] = parametros[key]
    
    def setProblema(self, problema):
#        print(problema)
#        exit()
        self.problema = problema
        self.indicadores['problema'] = problema.getNombre()
#        self.indicadores['instancia'] = problema.instancia
#        pass
    
    def moveSwarm(self, swarm, velocity, personalBest, bestFound, inertia):        
        print(f'swarm {swarm}')
        print(f'personalBest {personalBest}')
        print(f'bestFound {bestFound}')
#        exit()
        randPer = np.random.uniform(low=-1, high=1)
        randBest = np.random.uniform(low=-1, high=1)
        accelPer = self.contenedorParametros['accelPer']
        accelBest = self.contenedorParametros['accelBest']
#        self.randPer = 1
#        self.randBest = 1
        personalDif = personalBest - swarm
        personalAccel = accelPer * randPer * personalDif
        bestDif = bestFound - swarm
        bestAccel = accelBest * randBest * bestDif
        acceleration =  personalAccel + bestAccel
        
        nextVel = (inertia*velocity) + acceleration
#        print(f'velocidad anterior {velocity}')
        
        nextVel[nextVel > self.contenedorParametros['maxVel']]  = self.contenedorParametros['maxVel']
        nextVel[nextVel < self.contenedorParametros['maxVel']]  = self.contenedorParametros['maxVel']
#        print(f'velocidad actual {nextVel}')
        ret = swarm+nextVel
        ret[ret > self.contenedorParametros['rangoSolucion']['max']]  = self.contenedorParametros['rangoSolucion']['max']
        ret[ret < self.contenedorParametros['rangoSolucion']['min']] = self.contenedorParametros['rangoSolucion']['min']
        return ret, nextVel
    
    def aplicarMovimiento(self, datosNivel, iteracion, totIteraciones):        
        args = [
                [datosNivel['soluciones'][idx]
                ,datosNivel['velocidades'][idx]
                ,datosNivel['mejoresSoluciones'][idx]
                ,self.contenedorParametros['mejorSolGlobal']
                ,1 - (iteracion/(totIteraciones + 1))
                ] 
                for idx in range(datosNivel['soluciones'].shape[0])]
        resultadoMovimiento = {}
        if self.procesoParalelo:
            pool = mp.Pool(4)
            ret = pool.starmap(self.moveSwarm, args)
            pool.close()
            solucionesBin, evaluaciones = self.evaluarSoluciones([item[0] for item in ret])
            
            resultadoMovimiento['soluciones'] = np.vstack(np.array(ret)[:,0])
            resultadoMovimiento['solucionesBin'] = solucionesBin
            resultadoMovimiento['evalSoluciones'] = evaluaciones
            resultadoMovimiento['velocidades'] = np.vstack(np.array(ret)[:,1])
        else:
            soluciones = []
            solucionesBin = []
            evaluaciones = []
            velocidades = []
#            print(f'velocidad {args[0][1][1]}')
            for arg in args:
                
                sol, vel = self.moveSwarm(arg[0], arg[1], arg[2], arg[3], arg[4])
#                print(f'solucion inicial {arg[0]}')
#                print(f'velocidad inicial {arg[1]}')
#                print(f'solucion calculada {sol}')
#                exit()
                evals, solBin, _ = self.problema.evalEnc(sol)
                soluciones.append(sol)
                solucionesBin.append(solBin)
                evaluaciones.append(evals)
                velocidades.append(vel)
            resultadoMovimiento['soluciones'] = np.vstack(np.array(soluciones))
            resultadoMovimiento['solucionesBin'] = np.array(solucionesBin)
            resultadoMovimiento['evalSoluciones'] = np.array(evaluaciones)
            resultadoMovimiento['velocidades'] = np.vstack(np.array(velocidades))
                
        return resultadoMovimiento
    
    def evaluarSoluciones(self, soluciones):
        pool = mp.Pool(4)
        ret = pool.map(self.problema.evalEnc, soluciones)
        pool.close()
        solucionesBin = np.vstack(np.array(ret)[:,1])
        evaluaciones = np.vstack(np.array(ret)[:,0])
#        mejoresEvaluaciones.reshape((mejoresEvaluaciones.shape[0]))
        return solucionesBin, evaluaciones.reshape((evaluaciones.shape[0]))
    
    def generarSolucion(self):
        start = datetime.now()
        niveles = self.contenedorParametros['niveles']
        for nivel in range(niveles):
            nivel += 1
            print(f'ACTUALIZANDO NIVEL {nivel}')
            if not nivel in self.contenedorParametros['datosNivel']: 
                self.contenedorParametros['datosNivel'][nivel] = self.generarNivel(nivel)
            datosNivel = self.contenedorParametros['datosNivel'][nivel]
            
            for iteracion in range(self.contenedorParametros['iterPorNivel'][nivel]):
                print(f'mejor valor encontrado {self.contenedorParametros["mejorEvalGlobal"]}')
                resultadoMovimiento = self.aplicarMovimiento(datosNivel, iteracion, self.contenedorParametros['iterPorNivel'][nivel])
                datosNivel['soluciones']     = resultadoMovimiento['soluciones']
#                datosNivel['soluciones'][datosNivel['soluciones'] == 1] = self.contenedorParametros['rangoSolucion']['max']
#                datosNivel['soluciones'][datosNivel['soluciones'] == 0] = self.contenedorParametros['rangoSolucion']['min']
#                print(datosNivel['soluciones'])
#                exit()
                datosNivel['solucionesBin']  = resultadoMovimiento['solucionesBin']
                datosNivel['evalSoluciones'] = resultadoMovimiento['evalSoluciones']
                datosNivel['velocidades']    = resultadoMovimiento['velocidades']
                self.contenedorParametros['datosNivel'][nivel] = self.evaluarGrupos(datosNivel)
        end = datetime.now()
        self.indicadores['tiempoEjecucion'] = end-start
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']
                
    def generarSolucionAlAzar(self, numSols):
        return self.problema.generarSolsAlAzar(numSols)
        
                        
    def generarNivel(self, nivel):
        velocidades = np.random.uniform(size=(self.contenedorParametros['iterPorNivel'][nivel], self.problema.getNumDim()))
        if nivel == 1:
#            print(self.problema.getNumDim())
            soluciones = np.array(self.generarSolucionAlAzar(self.contenedorParametros['iterPorNivel'][nivel]))
            mejoresSoluciones = np.array(self.generarSolucionAlAzar(self.contenedorParametros['iterPorNivel'][nivel]))
#            velocidades = np.random.uniform(size=(self.contenedorParametros['iterPorNivel'][nivel], self.problema.getNumDim()))
            solucionesBin, evaluaciones               = self.evaluarSoluciones(soluciones)
            mejoresSolucionesBin, mejoresEvaluaciones = self.evaluarSoluciones(mejoresSoluciones)
            mejoresEvaluaciones[evaluaciones>mejoresEvaluaciones] = evaluaciones[evaluaciones>mejoresEvaluaciones]
            mejoresSoluciones[evaluaciones>mejoresEvaluaciones] = soluciones[evaluaciones>mejoresEvaluaciones]
            mejoresSolucionesBin[evaluaciones>mejoresEvaluaciones] = solucionesBin[evaluaciones>mejoresEvaluaciones]
            datosNivel = {}
            datosNivel['mejoresEvaluaciones'] = mejoresEvaluaciones
            datosNivel['mejoresSoluciones'] = mejoresSoluciones
            datosNivel['mejoresSolucionesBin'] = mejoresSolucionesBin
            datosNivel['evalSoluciones'] = evaluaciones
            datosNivel['velocidades']    = velocidades
            datosNivel['soluciones']     = soluciones
            datosNivel['solucionesBin']  = solucionesBin
            
            datosNivel = self.agruparNivel(datosNivel, nivel)
        else:
            nivelAnterior = self.contenedorParametros['datosNivel'][nivel-1]
#            print(len(nivelAnterior['mejorSolGrupo']))
#            exit()
            soluciones = np.array([nivelAnterior['mejorSolGrupo'][key] for key in nivelAnterior['mejorSolGrupo']])
            evals = np.array([nivelAnterior['mejorEvalGrupo'][key] for key in nivelAnterior['mejorEvalGrupo']])
            solsBin = np.array([nivelAnterior['mejorSolGrupoBin'][key] for key in nivelAnterior['mejorSolGrupoBin']])
            mejoresSol = soluciones.copy()
            
            mejoresEvals = evals.copy()
#            print(soluciones.shape)
#            print(mejoresSol.shape)
#            exit()
            datosNivel = {}
            datosNivel['soluciones']     = soluciones
            datosNivel['mejoresEvaluaciones'] = mejoresEvals
            datosNivel['mejoresSoluciones'] = mejoresSol
            datosNivel['mejoresSolucionesBin'] = solsBin.copy()
            datosNivel['evalSoluciones'] = evals
            datosNivel['velocidades']    = velocidades
            datosNivel['solucionesBin']  = solsBin
            datosNivel = self.agruparNivel(datosNivel, nivel)
        return datosNivel
    
    def agruparNivel(self, datosNivel, nivel):
        
#        print(datosNivel['soluciones'].shape)
#        print(f'num grupos {self.contenedorParametros["gruposPorNivel"][nivel]}')
        centroids,_ = kmeans(datosNivel['soluciones'],self.contenedorParametros['gruposPorNivel'][nivel])
        grupos,_ = vq(datosNivel['soluciones'],centroids)
#        print(grupos)
#        exit()
        datosNivel['grupos'] = grupos
#        for idxSolucion in range(len(grupos)):
#            if not grupos[idxSolucion] in datosNivel['grupos']:
#                datosNivel['grupos'][grupos[idxSolucion]] = []
        datosNivel = self.evaluarGrupos(datosNivel)
        return datosNivel    
        
            
    def evaluarGrupos(self, datosNivel):
        mejorSolucionGrupo = {}
        mejorSolucionGrupoBin = {}
        mejorEvaluacionGrupo = {}
        mejorGlobal = self.contenedorParametros['mejorSolGlobal']
        mejorGlobalBin = self.contenedorParametros['mejorSolucionBin']
        mejorEvalGlobal = self.contenedorParametros['mejorEvalGlobal']
        
        for idxSolucion in range(len(datosNivel['soluciones'])):
#            print(datosNivel['evalSoluciones'][idxSolucion])
            if datosNivel['evalSoluciones'][idxSolucion] > datosNivel['mejoresEvaluaciones'][idxSolucion]:
                datosNivel['mejoresEvaluaciones'][idxSolucion] = datosNivel['evalSoluciones'][idxSolucion]
                datosNivel['mejoresSoluciones'][idxSolucion] = datosNivel['soluciones'][idxSolucion]
            
            
            if (not datosNivel['grupos'][idxSolucion] in mejorEvaluacionGrupo 
                    or datosNivel['mejoresEvaluaciones'][idxSolucion] > mejorEvaluacionGrupo[datosNivel['grupos'][idxSolucion]]): 
                mejorEvaluacionGrupo[datosNivel['grupos'][idxSolucion]] = datosNivel['evalSoluciones'][idxSolucion]
                mejorSolucionGrupo[datosNivel['grupos'][idxSolucion]] = datosNivel['soluciones'] [idxSolucion]
                mejorSolucionGrupoBin[datosNivel['grupos'][idxSolucion]] = datosNivel['solucionesBin'] [idxSolucion]
                if mejorEvalGlobal is None or mejorEvalGlobal < datosNivel['evalSoluciones'][idxSolucion]:
                    mejorEvalGlobal = datosNivel['evalSoluciones'][idxSolucion]
                    mejorGlobal = datosNivel['soluciones'][idxSolucion]
                    mejorGlobalBin = datosNivel['solucionesBin'][idxSolucion]
        self.contenedorParametros['mejorEvalGlobal'] = mejorEvalGlobal
        self.contenedorParametros['mejorSolGlobal'] = mejorGlobal
        self.contenedorParametros['mejorSolucionBin'] = mejorGlobalBin
                    
        datosNivel['mejorEvalGrupo'] = mejorEvaluacionGrupo
        datosNivel['mejorSolGrupo']  = mejorSolucionGrupo
        datosNivel['mejorSolGrupoBin']  = mejorSolucionGrupoBin
        return datosNivel
    