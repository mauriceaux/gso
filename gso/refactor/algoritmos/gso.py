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
        #self.contenedorParametros['rangoSolucion'] = {'min' : -3}
        #self.contenedorParametros['rangoSolucion']['max'] = 3
        self.contenedorParametros['datosNivel'] = {}
        self.contenedorParametros['mejorEvalGlobal'] = None
        self.contenedorParametros['mejorSolucion'] = None
        self.contenedorParametros['mejorSolucionBin'] = None
        self.contenedorParametros['mejorSolGlobal'] = None
        self.contenedorParametros['accelPer'] = 2.05*np.random.uniform()
        self.contenedorParametros['accelBest'] = 2.05*np.random.uniform()
        self.contenedorParametros['maxVel'] = 0.2
        self.contenedorParametros['minVel'] = -0.2
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
        #print(f'sol          {swarm}')
        #print(f'personalBest {personalBest}')
        #print(f'bestFound    {bestFound}')
        #print(f'velocity     {velocity}')
#        exit()
        randPer = np.random.uniform(low=-1, high=1)
        randBest = np.random.uniform(low=-1, high=1)
        accelPer = self.contenedorParametros['accelPer']
        accelBest = self.contenedorParametros['accelBest']
#        self.randPer = 1
#        self.randBest = 1
        personalDif = personalBest - swarm
        personalAccel = accelPer * randPer * personalDif
        #print(f'personalAccel {personalAccel}')
        bestDif = bestFound - swarm
        bestAccel = accelBest * randBest * bestDif
        #print(f'bestAccel {bestAccel}')
        acceleration =  personalAccel + bestAccel
        
        nextVel = (inertia*velocity) + acceleration
        #print(f'velocidad anterior {nextVel}')
        
        nextVel[nextVel > self.contenedorParametros['maxVel']]  = self.contenedorParametros['maxVel']
        nextVel[nextVel < self.contenedorParametros['minVel']]  = self.contenedorParametros['minVel']
        #print(f'velocidad nueva {nextVel}')
        #exit()
        ret = swarm+nextVel
        ret[ret > self.problema.getRangoSolucion()['max']]  = self.problema.getRangoSolucion()['max']
        ret[ret < self.problema.getRangoSolucion()['min']] = self.problema.getRangoSolucion()['min']
        return ret, nextVel
    
    def aplicarMovimiento(self, datosNivel, iteracion, totIteraciones):        
        #args = [
        #        [datosNivel['soluciones'][idx]
        #        ,datosNivel['velocidades'][idx]
        #        ,datosNivel['mejoresSoluciones'][idx]
        #        ,self.contenedorParametros['mejorSolGlobal']
        #        ,1 - (iteracion/(totIteraciones + 1))
        #        ] 
        #        for idx in range(datosNivel['soluciones'].shape[0])]
        args = []
        for idx in range(datosNivel['soluciones'].shape[0]):
            mejorGrupo = datosNivel['mejorSolGrupo'][datosNivel['grupos'][idx]]
            if datosNivel['solPorGrupo'][datosNivel['grupos'][idx]] == 1:
                mejorGrupo = self.contenedorParametros['mejorSolGlobal']
            args.append([datosNivel['soluciones'][idx]
                ,datosNivel['velocidades'][idx]
                ,datosNivel['mejoresSoluciones'][idx]
                ,mejorGrupo
                ,1 - (iteracion/(totIteraciones + 1))
                ] )


        
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
            cont = 0
            for arg in args:
                #if cont == 0: print(f'solucion {arg[0]}')
                #cont +=1
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
        self.inicio = datetime.now()
        niveles = self.contenedorParametros['niveles']
        for nivel in range(niveles):
            nivel += 1
            print(f'ACTUALIZANDO NIVEL {nivel}')
            if not nivel in self.contenedorParametros['datosNivel']: 
                self.contenedorParametros['datosNivel'][nivel] = self.generarNivel(nivel)
            datosNivel = self.contenedorParametros['datosNivel'][nivel]
            
            for iteracion in range(self.contenedorParametros['iterPorNivel'][nivel]):
                print(f'nivel {nivel} iteracion {iteracion} mejor valor encontrado {self.contenedorParametros["mejorEvalGlobal"]}')
                resultadoMovimiento = self.aplicarMovimiento(datosNivel, iteracion, self.contenedorParametros['iterPorNivel'][nivel])
                datosNivel['soluciones']     = resultadoMovimiento['soluciones']
                datosNivel['solucionesBin']  = resultadoMovimiento['solucionesBin']
                datosNivel['evalSoluciones'] = resultadoMovimiento['evalSoluciones']
                datosNivel['velocidades']    = resultadoMovimiento['velocidades']
                self.contenedorParametros['datosNivel'][nivel] = self.agruparNivel(datosNivel, nivel)
        self.fin = datetime.now()
        self.indicadores['tiempoEjecucion'] = self.fin-self.inicio
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']
                
    def generarSolucionAlAzar(self, numSols):
        sols = self.problema.generarSolsAlAzar(numSols)
#        print(sols[0])
#        exit()
        return sols
        
                        
    def generarNivel(self, nivel):
        if nivel == 1:
            soluciones = np.array(self.generarSolucionAlAzar(self.contenedorParametros['numParticulas']))
            mejoresSoluciones = np.array(self.generarSolucionAlAzar(self.contenedorParametros['numParticulas']))
            velocidades = np.random.uniform(low=self.contenedorParametros['minVel'], high=self.contenedorParametros['maxVel'], size=(self.contenedorParametros['numParticulas'], self.problema.getNumDim()))
            solucionesBin, evaluaciones               = self.evaluarSoluciones(soluciones)
            mejoresSolucionesBin, mejoresEvaluaciones = self.evaluarSoluciones(mejoresSoluciones)
            idxMejores = evaluaciones>mejoresEvaluaciones
            mejoresEvaluaciones[idxMejores] = evaluaciones[idxMejores]
            mejoresSoluciones[idxMejores] = soluciones[idxMejores]
            mejoresSolucionesBin[idxMejores] = solucionesBin[idxMejores]
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
            soluciones = np.array([nivelAnterior['mejorSolGrupo'][key] for key in nivelAnterior['mejorSolGrupo']])
            velocidades = np.random.uniform(size=(len(soluciones), self.problema.getNumDim()))
            evals = np.array([nivelAnterior['mejorEvalGrupo'][key] for key in nivelAnterior['mejorEvalGrupo']])
            solsBin = np.array([nivelAnterior['mejorSolGrupoBin'][key] for key in nivelAnterior['mejorSolGrupoBin']])
            mejoresSol = soluciones.copy()
            
            mejoresEvals = evals.copy()
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
        datosNivel['soluciones'] = datosNivel['soluciones'].astype('float64')
        centroids,_ = kmeans(datosNivel['soluciones'],len(datosNivel['soluciones']))
        grupos,_ = vq(datosNivel['soluciones'],centroids)
        datosNivel['grupos'] = grupos
        datosNivel['solPorGrupo'] = {}
        for idGrupo in grupos:
            if not idGrupo in datosNivel['solPorGrupo']:
                datosNivel['solPorGrupo'][idGrupo] = 1
            else:
                datosNivel['solPorGrupo'][idGrupo] += 1
        datosNivel = self.evaluarGrupos(datosNivel)
        return datosNivel    
        
            
    def evaluarGrupos(self, datosNivel):
        mejorSolucionGrupo = {}
        mejorSolucionGrupoBin = {}
        mejorEvaluacionGrupo = {}
        datosNivel['mejorSolPorGrupos'] = {}
        datosNivel['mejorEvalPorGrupos'] = {}
        mejorGlobal = self.contenedorParametros['mejorSolGlobal']
        mejorGlobalBin = self.contenedorParametros['mejorSolucionBin']
        mejorEvalGlobal = self.contenedorParametros['mejorEvalGlobal']
        
        for idxSolucion in range(len(datosNivel['soluciones'])):
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
    