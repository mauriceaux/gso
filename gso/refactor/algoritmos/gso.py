import numpy as np
#from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt

class GSO():
    def __init__(self, niveles=2, numParticulas=50, iterPorNivel={1:50,2:250}, gruposPorNivel={1:12,2:12}):
        self.contenedorParametros = {}
        self.contenedorParametros['niveles'] = niveles
        self.contenedorParametros['nivel'] = 1
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
        self.contenedorParametros['autonomo'] = True
        self.procesoParalelo = False
        self.indicadores = {}
        self.indicadores['tiempos'] = {}
        self.paramDim = {}
        self.dataEvals = []
        self.nivelEjecutar = None
        
        self.mejorEval = None
        self.calcularParamDim()
        
        pass
    
    def getIndicadores(self):
        return self.indicadores
    
    def getParametros(self):
        return {
                'nivel':self.contenedorParametros['nivel']
                ,'numParticulas':self.contenedorParametros['numParticulas']
                ,'iterPorNivel':self.contenedorParametros['iterPorNivel']
                }
    
    def setParametros(self, parametros):
        for key in parametros:
            self.contenedorParametros[key] = parametros[key]
    
    def setProblema(self, problema):
#        print(problema)
#        exit()
        self.problema = problema
        self.indicadores['problema'] = problema.getNombre()
        self.indicadores['numDimProblema'] = problema.getNumDim()
#        self.indicadores['instancia'] = problema.instancia
#        pass
        
    def moveSwarm(self, swarm, velocity, personalBest, bestFound, inertia):
        accelPer = self.contenedorParametros['accelPer']
        accelBest = self.contenedorParametros['accelBest']
        maxVel = self.contenedorParametros['maxVel']
        minVel = self.contenedorParametros['minVel']
        maxVal = self.problema.getRangoSolucion()['max']
        minVal = self.problema.getRangoSolucion()['min']
        randPer = np.random.uniform(low=-1, high=1)
        randBest = np.random.uniform(low=-1, high=1)
        personalDif = personalBest - swarm
        personalAccel = accelPer * randPer * personalDif
        #print(f'personalAccel {personalAccel}')
        bestDif = bestFound - swarm
        bestAccel = accelBest * randBest * bestDif
        #print(f'bestAccel {bestAccel}')
        acceleration =  personalAccel + bestAccel
        
        nextVel = (inertia*velocity) + acceleration
        #print(f'velocidad anterior {nextVel}')
        
        nextVel[nextVel > maxVel]  = maxVel
        nextVel[nextVel < minVel]  = minVel
        #print(f'velocidad nueva {nextVel}')
        #exit()
        ret = swarm+nextVel
        ret[ret > maxVal]  = maxVal
        ret[ret < minVal] = minVal
        return ret, nextVel
    
#    def moveSwarm(self, swarm, velocity, personalBest, bestFound, inertia):
##        iodata = [swarm
##                  , velocity
##                  , personalBest
##                  , bestFound
##                  , inertia
##                  , self.contenedorParametros['accelPer']
##                  , self.contenedorParametros['accelBest']
##                  , self.contenedorParametros['maxVel']
##                  , self.contenedorParametros['minVel']
##                  , self.problema.getRangoSolucion()['max']
##                  , self.problema.getRangoSolucion()['min']
##                  , np.random.uniform(low=-1, high=1)
##                  , np.random.uniform(low=-1, high=1)
##                  , []
##                  , []]
#        ret = None
#        nextVel = None
##         Set the number of threads in a block
#        threadsperblock = 32 
#        
##         Calculate the number of thread blocks in the grid
#        blockspergrid = (15 + (threadsperblock - 1)) // threadsperblock
#        moveSwarm_(swarm
##        moveSwarm_[threadsperblock, blockspergrid](swarm
#                  , velocity
#                  , personalBest
#                  , bestFound
#                  , inertia
#                  , self.contenedorParametros['accelPer']
#                  , self.contenedorParametros['accelBest']
#                  , self.contenedorParametros['maxVel']
#                  , self.contenedorParametros['minVel']
#                  , self.problema.getRangoSolucion()['max']
#                  , self.problema.getRangoSolucion()['min']
#                  , np.random.uniform(low=-1, high=1)
#                  , np.random.uniform(low=-1, high=1)
#                  , ret
#                  , nextVel)
#        exit()
##        self.moveSwarm_[blockspergrid, threadsperblock](iodata)
##        return iodata[-2], iodata[-1]
#        return ret, nextVel
    
#    @staticmethod
    
        
#        return ret, nextVel
    
    
    #def aplicarMovimiento(self, datosNivel, iteracion, totIteraciones):
    #    iodata = [datosNivel, iteracion, totIteraciones, {}]
    #    self.aplicarMovimiento_(iodata)
    #    return iodata[3]
    
    
    def aplicarMovimiento(self, datosNivel, iteracion, totIteraciones):
#    def aplicarMovimiento_(self, iodata):        
#        datosNivel, iteracion, totIteraciones, resultadoMovimiento = iodata  
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
                mejorGrupo = datosNivel['mejorGlobal']
            args.append([datosNivel['soluciones'][idx]
                ,datosNivel['velocidades'][idx]
                ,datosNivel['mejoresSoluciones'][idx]
                ,mejorGrupo
                ,1 - (iteracion/(totIteraciones + 1))
                ] )


        
        resultadoMovimiento = {}
        if self.procesoParalelo and False:
            start = datetime.now()
            pool = mp.Pool(4)
            ret = pool.starmap(self.moveSwarm, args)
            pool.close()
            end = datetime.now()
            self.guardarIndicadorTiempo('moveSwarm', len(args), end-start)
            
            start = datetime.now()
            solucionesBin, evaluaciones = self.evaluarSoluciones([item[0] for item in ret])
            
            end = datetime.now()
            self.guardarIndicadorTiempo('evaluarSoluciones', len(ret), end-start)
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
#                print(arg)
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
                #self.guardarIndicadorTiempo('generarSolucionAlAzar', len(soluciones), end-start)
            evaluaciones = np.array(evaluaciones)
            self.agregarDataEjec(evaluaciones)
            resultadoMovimiento['soluciones'] = np.vstack(np.array(soluciones))
            resultadoMovimiento['solucionesBin'] = np.array(solucionesBin)
            resultadoMovimiento['evalSoluciones'] = evaluaciones
            resultadoMovimiento['velocidades'] = np.vstack(np.array(velocidades))
                
        return resultadoMovimiento
    
    def evaluarSoluciones(self, soluciones):
        start = datetime.now()
        pool = mp.Pool(4)
        ret = pool.map(self.problema.evalEnc, soluciones)
        pool.close()
        solucionesBin = np.vstack(np.array(ret)[:,1])
        evaluaciones = np.vstack(np.array(ret)[:,0])
#        mejoresEvaluaciones.reshape((mejoresEvaluaciones.shape[0]))
        end = datetime.now()
#        if self.evaluaciones is None:
#            self.evaluaciones = {}
#        self.evaluaciones(evaluaciones)
        self.guardarIndicadorTiempo('generarSolucionAlAzar', len(soluciones), end-start)
        self.agregarDataEjec(evaluaciones)
        return solucionesBin, evaluaciones.reshape((evaluaciones.shape[0]))
    
    def agregarDataEjec(self, evaluaciones):
        self.agregarMejorResultado()
#        if self.dataEvals is None:
#            self.dataEvals = []
        self.dataEvals.append(evaluaciones.tolist())
        self.totalEvals = len(self.dataEvals)
        #if not 'mejoresResultados' in self.indicadores:
        #    self.indicadores['mejoresResultados'] = []
        if not 'mejoresResultadosReales' in self.indicadores:
            self.indicadores['mejoresResultadosReales'] = []
        self.indicadores['mejoresResultadosReales'].append(np.max(evaluaciones))
        if not 'mediaResultadosReales' in self.indicadores:
            self.indicadores['mediaResultadosReales'] = []
        self.indicadores['mediaResultadosReales'].append(np.mean(evaluaciones))
        
        
    
    def generarSolucion(self):
        self.inicio = datetime.now()
        niveles = self.contenedorParametros['niveles']
        for nivel in range(niveles):
            nivel += 1
            print(f'ACTUALIZANDO NIVEL '+ str(nivel))
            if not nivel in self.contenedorParametros['datosNivel']: 
                self.contenedorParametros['datosNivel'][nivel] = self.generarNivel(nivel)
            datosNivel = self.contenedorParametros['datosNivel'][nivel]
            
            for iteracion in range(self.contenedorParametros['iterPorNivel'][nivel]):
                string = 'nivel '+str(nivel)+' iteracion '+str(iteracion)+' mejor valor encontrado '+str(self.contenedorParametros["mejorEvalGlobal"])
                print(string)
                resultadoMovimiento = self.aplicarMovimiento(datosNivel, iteracion, self.contenedorParametros['iterPorNivel'][nivel])
                datosNivel['soluciones']     = resultadoMovimiento['soluciones']
                datosNivel['solucionesBin']  = resultadoMovimiento['solucionesBin']
                datosNivel['evalSoluciones'] = resultadoMovimiento['evalSoluciones']
                datosNivel['velocidades']    = resultadoMovimiento['velocidades']
                self.contenedorParametros['datosNivel'][nivel] = self.evaluarGrupos(datosNivel)
                
#                self.contenedorParametros['datosNivel'][nivel] = self.agruparNivel(datosNivel, nivel)
        self.fin = datetime.now()
        self.indicadores['tiempoEjecucion'] = self.fin-self.inicio
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']
        
    def generarSolucionReducida(self):
        self.inicio = datetime.now()
        nivel = self.contenedorParametros['nivel']
        print(f'ACTUALIZANDO NIVEL '+ str(nivel))
        #if not nivel in self.contenedorParametros['datosNivel'] or nivel > 1: 
        if not nivel in self.contenedorParametros['datosNivel']: 
            self.contenedorParametros['datosNivel'][nivel] = self.generarNivel(nivel)
        
        datosNivel = self.contenedorParametros['datosNivel'][nivel]
        
        for iteracion in range(self.contenedorParametros['numIteraciones']):
            string = 'nivel '+str(nivel)+' iteracion '+str(iteracion)+' mejor valor encontrado '+str(self.contenedorParametros["mejorEvalGlobal"])
            print(string)
            resultadoMovimiento = self.aplicarMovimiento(datosNivel, iteracion, self.contenedorParametros['numIteraciones'])
            datosNivel['soluciones']     = resultadoMovimiento['soluciones']
            datosNivel['solucionesBin']  = resultadoMovimiento['solucionesBin']
            datosNivel['evalSoluciones'] = resultadoMovimiento['evalSoluciones']
            datosNivel['velocidades']    = resultadoMovimiento['velocidades']
            self.contenedorParametros['datosNivel'][nivel] = self.evaluarGrupos(datosNivel)
                
#                self.contenedorParametros['datosNivel'][nivel] = self.agruparNivel(datosNivel, nivel)
        self.fin = datetime.now()
        self.indicadores['tiempoEjecucion'] = self.fin-self.inicio
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']
                
    def generarSolucionAlAzar(self, numSols):
        start = datetime.now()
        sols = self.problema.generarSolsAlAzar(numSols)
        end = datetime.now()
        self.guardarIndicadorTiempo('generarSolucionAlAzar', numSols, end-start)
#        print(sols[0])
#        exit()
        return sols
        
                        
    def generarNivel(self, nivel):
        start = datetime.now()
        totalNivel = 0
        if nivel == 1:
            totalNivel = self.contenedorParametros['numParticulas']
            soluciones = np.array(self.generarSolucionAlAzar(totalNivel))
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
            if not nivel-1 in self.contenedorParametros['datosNivel']:
                self.contenedorParametros['datosNivel'][nivel-1] = self.generarNivel(nivel-1)
            nivelAnterior = self.contenedorParametros['datosNivel'][nivel-1]
            soluciones = np.array([nivelAnterior['mejorSolGrupo'][key] for key in nivelAnterior['mejorSolGrupo']])
            totalNivel = len(soluciones)
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
        end = datetime.now()
        self.guardarIndicadorTiempo('generarNivel', totalNivel, end-start)
        
        return datosNivel
    
    def agruparNivel(self, datosNivel, nivel):
        datosNivel['soluciones'] = datosNivel['soluciones'].astype('float64')
        totalNivel = len(datosNivel['soluciones'])
        start = datetime.now()
#        if not self.contenedorParametros['autonomo']: numGrupos = self.contenedorParametros['gruposPorNivel'][nivel]
#        else:    
        numGrupos = self.contenedorParametros['gruposPorNivel'][nivel]
#            numGrupos = self.calcularNumGrupos(datosNivel['soluciones'])
        kmeans = KMeans(n_clusters=numGrupos, init='k-means++')
        grupos = kmeans.fit_predict(datosNivel['soluciones'])
#        print(grupos)
#        exit()
#        centroids,_ = kmeans(datosNivel['soluciones'],len(datosNivel['soluciones']))
#        grupos,_ = vq(datosNivel['soluciones'],centroids)
        datosNivel['grupos'] = grupos
        datosNivel['solPorGrupo'] = {}
        for idGrupo in grupos:
            if not idGrupo in datosNivel['solPorGrupo']:
                datosNivel['solPorGrupo'][idGrupo] = 1
            else:
                datosNivel['solPorGrupo'][idGrupo] += 1
        datosNivel = self.evaluarGrupos(datosNivel)
        end = datetime.now()
        self.guardarIndicadorTiempo('agruparNivel', totalNivel, end-start)
        return datosNivel    
        
    def calcularNumGrupos(self, soluciones):
        wcss = []
#        for i in range(20):
#        for i in range(len(soluciones)):
        for i in range(int(len(soluciones)/3)):
            kmeans = KMeans(n_clusters=i+1, init='k-means++', max_iter=300, n_init=10)
            res = kmeans.fit(soluciones)
            wcss.append(res.inertia_)
        
        return self.optimal_number_of_clusters(wcss)
        
    
    def optimal_number_of_clusters(self,wcss):
        x1, y1 = 2, wcss[0]
        x2, y2 = 20, wcss[len(wcss)-1]
    
        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
#        print(distances)
#        print(distances.index(max(distances)) + 2)
#        plt.plot(wcss, 'bx-')
#        plt.show()
#        exit()
        return distances.index(max(distances)) + 2
            
    def evaluarGrupos(self, datosNivel):
        start = datetime.now()
        mejorSolucionGrupo = {}
        mejorSolucionGrupoBin = {}
        mejorEvaluacionGrupo = {}
        datosNivel['mejorSolPorGrupos'] = {}
        datosNivel['mejorEvalPorGrupos'] = {}
        mejorGlobal = self.contenedorParametros['mejorSolGlobal']
        mejorGlobalBin = self.contenedorParametros['mejorSolucionBin']
        mejorEvalGlobal = self.contenedorParametros['mejorEvalGlobal']
        total = len(datosNivel['soluciones'])
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
        datosNivel['mejorGlobal']  = mejorGlobal
        end = datetime.now()
        self.guardarIndicadorTiempo('evaluarGrupos', total, end-start)
        return datosNivel
    
    def guardarIndicadorTiempo(self, nombre, numEjec, timedelta):
        if not nombre in self.indicadores['tiempos']:
            self.indicadores['tiempos'][nombre] = {}
        if not numEjec in self.indicadores['tiempos'][nombre]:
            valorAnterior = timedelta.total_seconds()
        else:
            valorAnterior = self.indicadores['tiempos'][nombre][numEjec]
        
        self.indicadores['tiempos'][nombre][numEjec] = np.mean([timedelta.total_seconds(), valorAnterior])
        
    def agregarMejorResultado(self):
        if not 'mejoresResultados' in self.indicadores:
            self.indicadores['mejoresResultados'] = []
        self.indicadores['mejoresResultados'].append(self.contenedorParametros['mejorEvalGlobal'])
        
    def getParamDim(self):
#        if self.paramDim is None
#        paramDim = {}
        
        return self.paramDim
    
    def calcularParamDim(self):
        self.paramDim['nivel'] : {'min' : 1
                    , 'max' : self.contenedorParametros['niveles']
                    , 'tipo' : int
                }
        self.paramDim['totalParticulas'] = {
                    'min' : 1
                    , 'max' : self.contenedorParametros['numParticulas']
                    , 'tipo' : int
                }
        self.paramDim['numIteraciones'] = {
                    'min' : 1
                    , 'max' : 30 #30 porque si
                    , 'tipo' : int
                }
        
        self.paramDim['gruposPorNivel'] = {
                    'min' : {1:1,2:1}
                    , 'max' : {1:int(self.contenedorParametros['numParticulas']/2) #la mitad de las particlas porque si
                                ,2:1}
                    , 'tipo' : int
                }
    
    
    
