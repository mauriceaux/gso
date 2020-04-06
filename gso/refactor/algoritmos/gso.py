import numpy as np
#from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
import multiprocessing.dummy as mp
from datetime import datetime
from graficos import Graficador
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

class GSO():
    def __init__(self, niveles=2, numParticulas=50, iterPorNivel={1:50,2:250}, gruposPorNivel={1:12,2:12}):
        self.contenedorParametros = {}
        self.contenedorParametros['niveles'] = niveles
        self.contenedorParametros['nivel'] = 1
        self.contenedorParametros['numParticulas'] = numParticulas
        self.contenedorParametros['iterPorNivel'] = iterPorNivel
        self.contenedorParametros['gruposPorNivel'] = gruposPorNivel
        self.contenedorParametros['porcentajePartPorGrupoNivel'] = {1:0.1, 2:1}
        #self.contenedorParametros['rangoSolucion'] = {'min' : -3}
        #self.contenedorParametros['rangoSolucion']['max'] = 3
        self.contenedorParametros['datosNivel'] = {}
        self.contenedorParametros['mejorEvalGlobal'] = None
        self.contenedorParametros['mejorSolucion'] = None
        self.contenedorParametros['mejorSolucionBin'] = None
        self.contenedorParametros['mejorSolGlobal'] = None
#        self.contenedorParametros['accelPer'] = 2.05*np.random.uniform()
#        self.contenedorParametros['accelBest'] = 2.05*np.random.uniform()
        self.contenedorParametros['accelPer'] = 2
        self.contenedorParametros['accelBest'] = 3
        self.contenedorParametros['maxVel'] = 5
        self.contenedorParametros['minVel'] = -5
        self.contenedorParametros['autonomo'] = True

        self.contenedorParametros['inercia'] = 3
        self.procesoParalelo = False
        self.indicadores = {}
        self.indicadores['tiempos'] = {}
        self.paramDim = {}
        self.dataEvals = []
        self.nivelEjecutar = None
        self.ultimoNivelEvaluado = None        
        self.mejorEval = None
        self.calcularParamDim()
        self.nivelAnterior = 1
        
        self.mostrarGraficoParticulas = True
        self.nivel1=None
        self.nivel2=None
        self.mejores = None
        self.mejorGlobal = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax = plt.gca()
        
        self.niveles = {}
        self.scaler = None
        self.plotShowing = False
        self.solPromedio = None
        self.mostrarPromedio = True
        self.geometric = False
        self.guardarDatosEjec = True
        self.nomArchivoDatosEjec = f"ejecucion{datetime.now()}.csv"
        self.graficador = Graficador.Graficador()
        
        
        pass
    
    def getIndicadores(self):
        return self.indicadores
    
    def getParametros(self):
        return {
                'nivel':self.contenedorParametros['nivel']
                ,'numParticulas':self.contenedorParametros['numParticulas']
                ,'iterPorNivel':self.contenedorParametros['iterPorNivel']
                ,'inercia':self.contenedorParametros['inercia']
                ,'accelPer':self.contenedorParametros['accelPer']
                ,'accelBest':self.contenedorParametros['accelBest']
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
        self.indicadores['numLlamadasFnObj'] = 0
        self.problema.paralelo = self.procesoParalelo
#        self.ax.set_ylim(-.000000000000000009, .000000000000000009)
#        escala = 10
#        escalax = 899 - (1/10**121) 
#        self.ax.set_ylim(-escala, escala)
#        self.ax.set_xlim(-escala, escala)
#        self.indicadores['instancia'] = problema.instancia
#        pass
    
    def moveSwarmGeometric(self, swarm, velocity, personalBest, bestFound, inertia):
        if len(swarm.shape) == 1:
            swarm = np.array([swarm])
        indicesOcupados = np.zeros(swarm.shape)
        #print(f"swarm.shape {swarm.reshape(-1,1)}")
        #print(f"indicesOcupados {indicesOcupados}")
        idx = 0
        for indice in indicesOcupados:
            #print(f"indice {indice}")
            disponible = np.where(indice==0)[0]
            #print(f"disponible.shape[0] {disponible.shape[0]}")
            #print(f"int(swarm.shape[1]*self.contenedorParametros['accelBest']) {int(swarm.shape[1]*self.contenedorParametros['accelBest'])}")
            if disponible.shape[0] >= int(swarm.shape[1]*self.contenedorParametros['accelBest']):
                eleccionRandomMejor = np.random.choice(np.where(indice==0)[0], int(swarm.shape[1]*self.contenedorParametros['accelBest']), replace=False)
                indice[eleccionRandomMejor] = 1
                swarm[idx,eleccionRandomMejor] = bestFound[eleccionRandomMejor]
            disponible = np.where(indice==0)[0]
            if disponible.shape[0] >= int(swarm.shape[1]*self.contenedorParametros['accelPer']):
                eleccionRandomPersonal = np.random.choice(np.where(indice==0)[0], int(swarm.shape[1]*self.contenedorParametros['accelPer']), replace=False)
                indice[eleccionRandomPersonal] = 1
                if len(personalBest.shape) == 1: personalBest = np.array([personalBest])
                swarm[idx,eleccionRandomPersonal] = personalBest[idx,eleccionRandomPersonal]
            idx+=1
        
        return swarm, None
    
    def moveSwarm(self, swarm, velocity, personalBest, bestFound, inertia, accelPer, accelBest):
        maxVel = self.contenedorParametros['maxVel']
        minVel = self.contenedorParametros['minVel']
        randPer = np.random.uniform(low=-1,high=2)
        randBest = np.random.uniform(low=-1,high=2)
        personalDif = personalBest - swarm
        
#        exit()        
        personalAccel = int(accelPer * randPer) * personalDif
        
        #print(f'personalAccel {personalAccel}')
        bestDif = bestFound - swarm
        
        bestAccel = int(accelBest * randBest) * bestDif
        
        #print(f'bestAccel {bestAccel}')
        acceleration =  personalAccel + bestAccel
        vInercia = inertia*velocity
        nextVel = vInercia + acceleration
        ret = swarm+nextVel
        return ret, nextVel
    
    def moveSwarmDiscrete(self, swarm, velocity, personalBest, bestFound, inertia, accelPer, accelBest):
        accelPer = self.contenedorParametros['accelPer']
        accelBest = self.contenedorParametros['accelBest']
        maxVel = self.contenedorParametros['maxVel']
        minVel = self.contenedorParametros['minVel']
        maxVal = self.problema.getRangoSolucion()['max']
        minVal = self.problema.getRangoSolucion()['min']
        randPer = 1
        randBest = 1
        
        personalDif = abs(personalBest - swarm)
        idsDistinto0 = np.argwhere(personalDif != 0)
        idsDistinto0 = idsDistinto0.reshape(np.prod(idsDistinto0.shape))
        idsSel = np.random.choice(idsDistinto0,int(accelPer))
        personalAccel = np.zeros(swarm.shape)
        personalAccel[idsSel] = personalDif[idsSel]
        
        bestDif = abs(bestFound - swarm)
        idsDistinto0 = np.argwhere(bestDif != 0)
        idsDistinto0 = idsDistinto0.reshape(np.prod(idsDistinto0.shape))
        idsSel = np.random.choice(idsDistinto0,int(accelBest))
        bestAccel = np.zeros(swarm.shape)
        bestAccel[idsSel] = personalDif[idsSel]
        
        acceleration =  personalAccel + bestAccel
        
        idsDistinto0 = np.argwhere(velocity != 0)
        idsDistinto0 = idsDistinto0.reshape(np.prod(idsDistinto0.shape))
        idsSel = np.random.choice(idsDistinto0,int(inertia))
        vInercia = np.zeros(swarm.shape)
        vInercia[idsSel] = velocity[idsSel]
        
        nextVel = vInercia + acceleration
        ret = swarm+nextVel
        ret[ret > maxVal]  = maxVal
        ret[ret < minVal] = minVal
        return ret, nextVel
    
    def aplicarMovimiento(self, datosNivel, iteracion, totIteraciones):
        start = datetime.now()
        args = []
        start = datetime.now()
        if self.contenedorParametros['autonomo']:
            inercia = self.contenedorParametros['inercia']
        else:
            inercia = 1 - (iteracion/(totIteraciones + 1)) 
        for idx in range(datosNivel['soluciones'].shape[0]):
            mejorGrupo = datosNivel['mejorSolGrupo'][datosNivel['grupos'][idx]]
            if idx < len(datosNivel['grupos']) and datosNivel['solPorGrupo'][datosNivel['grupos'][idx]] == 1:
                mejorGrupo = datosNivel['mejorGlobal']
            args.append([datosNivel['soluciones'][idx]
                ,datosNivel['velocidades'][idx]
                ,datosNivel['mejoresSoluciones'][idx]
                ,mejorGrupo
                ,inercia
                ,self.contenedorParametros['accelPer']
                ,self.contenedorParametros['accelBest']
                ] )

        end = datetime.now()
        start = datetime.now()
        resultadoMovimiento = {}
        ret = []
        if self.procesoParalelo:
            pool = mp.Pool(4)
            if self.geometric:
                ret = pool.starmap(self.moveSwarmGeometric, args)
            else:
                ret = pool.starmap(self.moveSwarm, args)
            pool.close()
        else:
            solucionesBin = []
            evaluaciones = []
            ret = []
            for arg in args:
                if self.geometric:
                    sol, vel = self.moveSwarmGeometric(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6])
                else:
                    sol, vel = self.moveSwarm(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6])

                ret.append([sol,vel])
        end = datetime.now()
        start = datetime.now()
        solucionesBin, evaluaciones = self.evaluarSoluciones(np.array([item[0] for item in ret]))
        end=datetime.now()
#        resultadoMovimiento['soluciones'] = np.vstack(np.array(ret)[:,0])
#        resultadoMovimiento['soluciones'][resultadoMovimiento['soluciones'] == 1] = self.problema.getRangoSolucion()['max']
#        resultadoMovimiento['soluciones'][resultadoMovimiento['soluciones'] == 0] = self.problema.getRangoSolucion()['min']
        resultadoMovimiento['soluciones'] = np.array([item[0] for item in ret])
#        print(resultadoMovimiento['soluciones'][0])
        resultadoMovimiento['solucionesBin'] = solucionesBin
        resultadoMovimiento['evalSoluciones'] = evaluaciones
        if not self.geometric: resultadoMovimiento['velocidades'] = np.vstack(np.array(ret)[:,1])
        end = datetime.now()
        self.guardarIndicadorTiempo('aplicarMovimiento', len(args), end-start)
        return resultadoMovimiento
    
    def evaluarSoluciones(self, soluciones):
        start = datetime.now()
        solucionesBin = None
        evaluaciones = None
        binarizationStrategy = getattr(self.problema, "binarizationStrategy", None)
        if binarizationStrategy is not None:
            self.problema.binarizationStrategy.mejorSol = self.contenedorParametros['mejorSolGlobal']
        if self.procesoParalelo:
            
            pool = mp.Pool(4)
            if not self.geometric: ret = pool.map(self.problema.evalEnc, soluciones)
            else: ret = pool.map(self.problema.eval, soluciones)
            pool.close()
            solucionesBin = np.vstack(np.array(ret)[:,1])
            evaluaciones = np.vstack(np.array(ret)[:,0])
        else:
            solucionesBin = []
            evaluaciones = []
#            print(f"soluciones {soluciones}")
            for sol in soluciones:
                
                if not self.geometric: eval, bin, _ = self.problema.evalEnc(sol)
                else: eval, bin, _ = self.problema.eval(sol)
                solucionesBin.append(bin)
                evaluaciones.append(eval)
            solucionesBin = np.array(solucionesBin)
            evaluaciones = np.array(evaluaciones)
            #print(evaluaciones)
            #exit()
        
#        if self.mostrarPromedio:
#            tmp = np.mean(soluciones, axis=0)
#            if self.solPromedio is None:
#                self.solPromedio = tmp
#            else:
#                
#                self.solPromedio = np.append(self.solPromedio, tmp).reshape(2, tmp.shape[0])
#    #            print(self.solPromedio.shape)
#    #            exit()
#                self.solPromedio = np.mean(self.solPromedio, axis=0)
        
        end = datetime.now()
        self.guardarIndicadorTiempo('evaluarSoluciones', len(soluciones), end-start)
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
        #self.indicadores['mejoresResultadosReales'].append(np.max(evaluaciones))
        self.indicadores['numLlamadasFnObj'] += evaluaciones.shape[0]
        if not 'mejoresResultadosReales' in self.indicadores:
            self.indicadores['mejoresResultadosReales'] = []
        self.indicadores['mejoresResultadosReales'].append(np.max(evaluaciones))
        if not 'mediaResultadosReales' in self.indicadores:
            self.indicadores['mediaResultadosReales'] = []
        self.indicadores['mediaResultadosReales'].append(np.mean(evaluaciones))
        
        
    
    def generarSolucion(self):
        if self.mostrarGraficoParticulas:
#            plt.ion()
#            plt.show()
            input("Press Enter to continue...")
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
                self.contenedorParametros['datosNivel'][nivel] = self.agruparNivel(datosNivel, nivel)
                if self.mostrarGraficoParticulas:
#                    self.line1, = self.ax.plot(datosNivel['soluciones'][0,0], datosNivel['soluciones'][0,1], 'r-', marker='o') 
                    self.graficarParticulas(datosNivel, nivel)
#                    if self.nivel1 is None and nivel == 1:
#                        self.nivel1, = self.ax.plot(datosNivel['soluciones'][:,0], datosNivel['soluciones'][:,1], c='r', marker='o', linestyle='None') 
#                    if self.nivel2 is None and nivel == 2:
#                        self.nivel2, = self.ax.plot(datosNivel['soluciones'][:,0], datosNivel['soluciones'][:,1], c='b', marker='o', linestyle='None') 
#                    
#                    if nivel == 1:
#                        self.nivel1.set_xdata(datosNivel['soluciones'][:,0])
#                        self.nivel1.set_ydata(datosNivel['soluciones'][:,1])
#                    if nivel == 2:
#                        self.nivel2.set_xdata(datosNivel['soluciones'][:,0])
#                        self.nivel2.set_ydata(datosNivel['soluciones'][:,1])
#
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
#                    self.graficarParticulas(datosNivel)
                    
#                self.contenedorParametros['datosNivel'][nivel] = self.agruparNivel(datosNivel, nivel)
        self.fin = datetime.now()
        self.indicadores['tiempoEjecucion'] = self.fin-self.inicio
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']
        
    def generarSolucionReducida(self):
        if 1 in self.contenedorParametros['datosNivel']:
            dif = self.contenedorParametros['numParticulas'] - self.contenedorParametros['datosNivel'][1]['soluciones'].shape[0]
            print(f'DIFERENCIA {dif}')
            if dif != 0:
                self.agregarEliminarParticulas(dif)
        nivel = self.contenedorParametros['nivel']
        if not nivel in self.contenedorParametros['datosNivel'] or (self.nivelAnterior == 1 and nivel == 2): 
            self.contenedorParametros['datosNivel'][nivel] = self.generarNivel(nivel)
        datosNivel = self.contenedorParametros['datosNivel'][nivel]
        self.nivelAnterior = nivel
        self.inicio = datetime.now()
        
        if self.mostrarGraficoParticulas and not self.plotShowing:
#            escala = abs(self.contenedorParametros['mejorEvalGlobal'])*2
#            self.ax.set_ylim(-escala, escala)
#            self.ax.set_xlim(-escala, escala)
#            plt.ion()
#            plt.show()
#            plt.draw()
            self.plotShowing = True
#            self.ax = plt.gca()
#            evalProm = int(abs(np.mean(datosNivel['evalSoluciones'])))
#            self.ax.set_ylim(-evalProm, evalProm)
#            self.ax.set_xlim(-evalProm, evalProm)
#            input("press ENTER")
#        nivel = self.contenedorParametros['nivel']
        print(f'ACTUALIZANDO NIVEL '+ str(nivel))
        #if not nivel in self.contenedorParametros['datosNivel'] or nivel > 1: 
        
#        datosNivel = self.contenedorParametros['datosNivel'][nivel]
        
#        datosNivel = self.agruparNivel(datosNivel, nivel)
        
        for iteracion in range(self.contenedorParametros['numIteraciones']):
            
            resultadoMovimiento = self.aplicarMovimiento(datosNivel, iteracion, self.contenedorParametros['numIteraciones'])
            strPromedio = f" promedio {np.mean(resultadoMovimiento['evalSoluciones'])} " if self.mostrarPromedio else ''
            string = f'nivel {nivel} iteracion {iteracion} mejor valor encontrado {self.contenedorParametros["mejorEvalGlobal"]} {strPromedio} num particulas {datosNivel["soluciones"].shape[0]}'
            print(string)
            datosNivel['soluciones']     = resultadoMovimiento['soluciones']
#            datosNivel['soluciones'][resultadoMovimiento['solucionesBin'] == 0] = self.problema.getRangoSolucion()['min']
#            datosNivel['soluciones'][resultadoMovimiento['solucionesBin'] == 1] = self.problema.getRangoSolucion()['max']
            datosNivel['solucionesBin']  = resultadoMovimiento['solucionesBin']
            datosNivel['evalSoluciones'] = resultadoMovimiento['evalSoluciones']
            if not self.geometric: datosNivel['velocidades']    = resultadoMovimiento['velocidades']
            self.contenedorParametros['datosNivel'][nivel] = self.evaluarGrupos(datosNivel)
            if self.mostrarGraficoParticulas:
#                self.line1, = self.ax.plot(datosNivel['soluciones'][0,0], datosNivel['soluciones'][0,1], 'r-', marker='o') 
                self.graficarParticulas(datosNivel, nivel)
#                    
#                if self.nivel1 is None:
#                    self.nivel1, = self.ax.plot(datosNivel['soluciones'][:,0], datosNivel['soluciones'][:,1], c='r', marker='o', linestyle='None') 
#                else:
#                    self.nivel1.set_xdata(datosNivel['soluciones'][:,0])
#                    self.nivel1.set_ydata(datosNivel['soluciones'][:,1])
#
#                self.fig.canvas.draw()
#                self.fig.canvas.flush_events()
                
#                self.contenedorParametros['datosNivel'][nivel] = self.agruparNivel(datosNivel, nivel)
        self.fin = datetime.now()
        self.indicadores['tiempoEjecucion'] = self.fin-self.inicio
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']


    def agregarEliminarParticulas(self, dif):
        nivel1 = self.contenedorParametros['datosNivel'][1]
        #print(f' soluciones inicio {nivel1["soluciones"]}')
        if dif > 0:
            print(f'AGREGANDO {dif} SOLUCIONES')
            solucionesBin, evaluaciones = self.generarSolucionAlAzar(dif)
            
            mejoresSoluciones = np.array([self.contenedorParametros['mejorSolGlobal'] for _ in range(dif)])
            mejoresSolucionesBin, mejoresEvaluaciones = self.evaluarSoluciones(mejoresSoluciones)
#            velocidades = np.random.uniform(low=self.contenedorParametros['minVel'], high=self.contenedorParametros['maxVel'], size=(dif, self.problema.getNumDim()))
#            velocidades = np.ones((dif, self.problema.getNumDim()),dtype=np.int64)
            velocidades = np.random.randint(low=-50, high=50, size=(dif, self.problema.getNumDim()), dtype=np.int64)
            mejoresVelocidades = velocidades.copy()
#            solucionesBin, evaluaciones               = self.evaluarSoluciones(soluciones)
#            mejoresSolucionesBin, mejoresEvaluaciones = self.evaluarSoluciones(mejoresSoluciones)
            soluciones = solucionesBin.copy()
#            soluciones[soluciones == 0] = self.problema.getRangoSolucion()['min']
#            soluciones[soluciones == 1] = self.problema.getRangoSolucion()['max']
            
            mejoresSoluciones = mejoresSolucionesBin.copy()
#            mejoresSoluciones[mejoresSoluciones == 0] = self.problema.getRangoSolucion()['min']
#            mejoresSoluciones[mejoresSoluciones == 1] = self.problema.getRangoSolucion()['max']
            
            
            idxMejores = evaluaciones>mejoresEvaluaciones
            mejoresEvaluaciones[idxMejores] = evaluaciones[idxMejores]
            mejoresSoluciones[idxMejores] = soluciones[idxMejores]
            mejoresVelocidades[idxMejores] = velocidades[idxMejores]
            mejoresSolucionesBin[idxMejores] = solucionesBin[idxMejores]

            nmejoresEvaluaciones = list(nivel1['mejoresEvaluaciones'])
            nmejoresSoluciones = list(nivel1['mejoresSoluciones'])
            nmejoresSolucionesBin = list(nivel1['mejoresSolucionesBin'])
            nevalSoluciones = list(nivel1['evalSoluciones'])
            nvelocidades = list(nivel1['velocidades'])
            nMejoresVelocidades = list(nivel1['mejoresVel'])
            nsoluciones = list(nivel1['soluciones'])
            nsolucionesBin = list(nivel1['solucionesBin'])

            nmejoresEvaluaciones.extend(mejoresEvaluaciones)
            nmejoresSoluciones.extend(mejoresSoluciones)
            nmejoresSolucionesBin.extend(mejoresSolucionesBin)
            nevalSoluciones.extend(evaluaciones)
            nvelocidades.extend(velocidades)
            nMejoresVelocidades.extend(mejoresVelocidades)
            nsoluciones.extend(soluciones)
            nsolucionesBin.extend(solucionesBin)
            #if len(nmejoresEvaluaciones) != self.contenedorParametros['numParticulas']:
            #    print(f'NO SE AGREGARON BIEN LAS PARTICULAS actuales: {len(nmejoresEvaluaciones)} deberian ser {self.contenedorParametros["numParticulas"]}')
            #print(len(nmejoresEvaluaciones))
            #exit()
            nivel1['mejoresEvaluaciones'] = np.array(nmejoresEvaluaciones)
            nivel1['mejoresSoluciones'] = np.array(nmejoresSoluciones)
            nivel1['mejoresSolucionesBin'] = np.array(nmejoresSolucionesBin)
            nivel1['evalSoluciones'] = np.array(nevalSoluciones)
            nivel1['velocidades']    = np.array(nvelocidades)
            nivel1['mejoresVel']     = np.array(nMejoresVelocidades)
            nivel1['soluciones']     = np.array(nsoluciones)
            nivel1['solucionesBin']  = np.array(nsolucionesBin)
            
        else:
            print(f'ELIMINANDO {dif} SOLUCIONES')
            while nivel1['soluciones'].shape[0] > self.contenedorParametros['numParticulas']:
                #idxPeor = np.random.choice(np.arange(nivel1['evalSoluciones'].shape[0]), 1)[0]
                idxPeor = np.argmin(nivel1['evalSoluciones'])
                mejoresEvaluaciones = nivel1['mejoresEvaluaciones'].tolist()
                mejoresSoluciones = nivel1['mejoresSoluciones'].tolist()
                mejoresSolucionesBin = nivel1['mejoresSolucionesBin'].tolist()
                evalSoluciones = nivel1['evalSoluciones'].tolist()
                velocidades = nivel1['velocidades'].tolist()
#                mejoresVel = nivel1['mejoresVel'].tolist()
                soluciones = nivel1['soluciones'].tolist()
                solucionesBin = nivel1['solucionesBin'].tolist()
                grupos = nivel1['grupos'].tolist()

                del mejoresEvaluaciones[idxPeor]
                del mejoresSoluciones[idxPeor]
                del mejoresSolucionesBin[idxPeor]
                del evalSoluciones[idxPeor]
                del velocidades[idxPeor]
#                del mejoresVel[idxPeor]
                del soluciones[idxPeor]
                del solucionesBin[idxPeor]
                del grupos[idxPeor]
#                print(f'self.niveles[1] {self.niveles[1]} idxPeor {idxPeor} nivel1["grupos"][idxPeor] {nivel1["grupos"][idxPeor]}')
#                peor =  self.niveles[1][nivel1['grupos'][idxPeor]]
#                del peor
#                del self.niveles[1][nivel1['grupos'][idxPeor]]
                
                

                #if len(mejoresEvaluaciones) != self.contenedorParametros['numParticulas']:
                #    print(f'NO SE ELIMINARON BIEN LAS PARTICULAS actuales: {len(mejoresEvaluaciones)} deberian ser {self.contenedorParametros["numParticulas"]}')
                nivel1['mejoresEvaluaciones'] = np.array(mejoresEvaluaciones)
                nivel1['mejoresSoluciones'] = np.array(mejoresSoluciones)
#                nivel1['mejoresVel']     = np.array(mejoresVel)
                nivel1['mejoresSolucionesBin'] = np.array(mejoresSolucionesBin)
                nivel1['evalSoluciones'] = np.array(evalSoluciones)
                nivel1['velocidades']    = np.array(velocidades)
                
                nivel1['soluciones']     = np.array(soluciones)
                nivel1['solucionesBin']  = np.array(solucionesBin)
                nivel1['grupos']         = np.array(grupos)

        #print(f' soluciones fin {nivel1["soluciones"]}')
        
        self.contenedorParametros['datosNivel'][1] = self.agruparNivel(nivel1, 1)


    def generarSolucionAlAzar(self, numSols):
        start = datetime.now()
#        print(self.contenedorParametros[''])
        sols, evals = self.problema.generarSolsAlAzar(numSols, self.contenedorParametros['mejorSolGlobal'])
#        sols, evals = self.problema.generarSolsAlAzar(numSols, self.solPromedio)
        end = datetime.now()
        self.guardarIndicadorTiempo('generarSolucionAlAzar', numSols, end-start)
#        print(sols[0])
#        exit()
        return sols, evals
        
                        
    def generarNivel(self, nivel):
        start = datetime.now()
        totalNivel = 0
        if nivel == 1:
            totalNivel = self.contenedorParametros['numParticulas']
            solucionesBin, evaluaciones = self.generarSolucionAlAzar(totalNivel)
            
            mejoresSolucionesBin, mejoresEvaluaciones = self.generarSolucionAlAzar(self.contenedorParametros['numParticulas'])
#            velocidades = np.random.uniform(low=self.contenedorParametros['minVel'], high=self.contenedorParametros['maxVel'], size=(self.contenedorParametros['numParticulas'], self.problema.getNumDim()))
            velocidades = np.random.randint(low=-50, high=50, size=(self.contenedorParametros['numParticulas'], self.problema.getNumDim()), dtype=np.int64)
#            velocidades = np.ones((self.contenedorParametros['numParticulas'], self.problema.getNumDim()), dtype=np.int64)  
#            print(f"velocidades.shape {velocidades}")
#            exit()
#            solucionesBin, evaluaciones               = self.evaluarSoluciones(soluciones)
#            mejoresSolucionesBin, mejoresEvaluaciones = self.evaluarSoluciones(mejoresSoluciones)
            soluciones = solucionesBin.copy()
#            print(f"soluciones {soluciones}")
#            exit()
#            divisor=1
#            soluciones[soluciones == 0] = self.problema.getRangoSolucion()['min']/divisor
#            soluciones[soluciones == 1] = self.problema.getRangoSolucion()['max']/divisor
            
            mejoresSoluciones = mejoresSolucionesBin.copy()
#            mejoresSoluciones[mejoresSoluciones == 0] = self.problema.getRangoSolucion()['min']/divisor
#            mejoresSoluciones[mejoresSoluciones == 1] = self.problema.getRangoSolucion()['max']/divisor
            
            idxMejores = evaluaciones>mejoresEvaluaciones
            mejoresEvaluaciones[idxMejores] = evaluaciones[idxMejores]
#            print(f"soluciones {soluciones.shape}")
#            print(f"idxMejores {idxMejores.shape}")
#            exit()
            mejoresSoluciones[idxMejores] = soluciones[idxMejores]
            mejoresSolucionesBin[idxMejores] = solucionesBin[idxMejores]
            datosNivel = {}
            datosNivel['mejoresEvaluaciones'] = mejoresEvaluaciones
            datosNivel['mejoresSoluciones'] = mejoresSoluciones
            datosNivel['mejoresSolucionesBin'] = mejoresSolucionesBin
            datosNivel['evalSoluciones'] = evaluaciones
            datosNivel['velocidades']    = velocidades
            datosNivel['mejoresVel']    = velocidades
            datosNivel['soluciones']     = soluciones
            datosNivel['solucionesBin']  = solucionesBin
            datosNivel = self.agruparNivel(datosNivel, nivel)
            
#            print(datosNivel['velocidades'])
#            print(datosNivel['grupos'])
#            exit()
        else:
            if not nivel-1 in self.contenedorParametros['datosNivel']:
                self.contenedorParametros['datosNivel'][nivel-1] = self.generarNivel(nivel-1)
            nivelAnterior = self.contenedorParametros['datosNivel'][nivel-1]
            soluciones = np.array([nivelAnterior['mejorSolGrupo'][key] for key in nivelAnterior['mejorSolGrupo'] if len(nivelAnterior['mejorSolGrupo'][key]) > 0])
#            print(f"nivelAnterior['mejoresVel'] {nivelAnterior['mejoresVel']}")
#            exit()
#            velocidades = np.array([nivelAnterior['mejoresVel'][key] for key in nivelAnterior['mejoresVel'] if len(nivelAnterior['mejoresVel'][key]) > 0])
            velocidades = np.random.randint(low=-50, high=50, size=(self.contenedorParametros['numParticulas'], self.problema.getNumDim()), dtype=np.int64)
#            velocidades = np.ones((self.contenedorParametros['numParticulas'], self.problema.getNumDim()), dtype=np.int64)  
#            print(f"nivelAnterior['mejorSolGrupo'] {nivelAnterior['mejorSolGrupo']}")
#            exit()
#            print(f"mejorSolGrupo {nivelAnterior['mejorSolGrupo']}")
            
            grupos = np.array([key for key in nivelAnterior['mejorSolGrupo'] ])
#            print(soluciones)
            
#            exit()
            totalNivel = len(soluciones)
#            velocidades = np.random.uniform(size=(len(soluciones), self.problema.getNumDim()))
            evals = np.array([nivelAnterior['mejorEvalGrupo'][key] for key in nivelAnterior['mejorEvalGrupo']])
            solsBin = np.array([nivelAnterior['mejorSolGrupoBin'][key] for key in nivelAnterior['mejorSolGrupoBin']])
            mejoresSol = soluciones.copy()
            datosNivel = {}
            datosNivel['soluciones']     = soluciones
            datosNivel['mejoresEvaluaciones'] = evals
            datosNivel['mejoresSoluciones'] = mejoresSol
            datosNivel['mejoresSolucionesBin'] = solsBin.copy()
            datosNivel['evalSoluciones'] = evals
            datosNivel['velocidades']    = velocidades
#            datosNivel['mejoresVel']    = mejoresVel
            datosNivel['solucionesBin']  = solsBin
            datosNivel['grupos'] = grupos
#            print(f'GRUPOS {grupos}')
            solPorGrupo = {}
            for grupo in grupos:
                solPorGrupo[grupo] = 1
            datosNivel['solPorGrupo'] = solPorGrupo
#            datosNivel['solPorGrupo'] = np.ones(grupos.shape)
            datosNivel = self.evaluarGrupos(datosNivel)
        end = datetime.now()
        self.guardarIndicadorTiempo('generarNivel', totalNivel, end-start)
#        print(f"datosNivel['soluciones'].shape {datosNivel['soluciones'].shape}")
#        exit() 
        return datosNivel
    
    def agruparNivel(self, datosNivel, nivel):
#        print(datosNivel['soluciones'])
#        exit()
#        datosNivel['soluciones'] = datosNivel['soluciones'].astype('float64')
        totalNivel = len(datosNivel['soluciones'])
#        print(int(totalNivel * 0.2))
#        exit()
        #totalParticulas = self.contenedorParametros['numParticulas']
        start = datetime.now()
#        if not self.contenedorParametros['autonomo']: numGrupos = self.contenedorParametros['gruposPorNivel'][nivel]
#        else:    
        
        
        if  False and 'grupos' in datosNivel and len(datosNivel['grupos']) > 0:
            nGrupos = np.max(datosNivel['grupos'])
            datosNivel['grupos'] = datosNivel['grupos'].tolist()
            
            for idx in range(len(datosNivel['grupos']), len(datosNivel['soluciones']), 1):
                datosNivel['grupos'].append(np.random.randint(low=0, high=nGrupos))
            datosNivel['grupos'] = np.array(datosNivel['grupos'])
        else:
        
        
            if nivel == 1:
                numGrupos = int(totalNivel * 0.1)
                numGrupos = 1 if numGrupos < 1 else numGrupos
            else:
                numGrupos = totalNivel
            #print(f'NUMERO GRUPOS {numGrupos}')
            #numGrupos = int(totalNivel * self.contenedorParametros['porcentajePartPorGrupoNivel'][nivel])
    #            numGrupos = self.calcularNumGrupos(datosNivel['soluciones'])
            kmeans = KMeans(n_clusters=numGrupos, init='k-means++')
    #        grupos = kmeans.fit_predict(datosNivel['velocidades'])
            grupos = kmeans.fit_predict(datosNivel['evalSoluciones'].reshape(-1,1))
            print(f"WSS {kmeans.inertia_}")
#            grupos = kmeans.fit_predict(datosNivel['soluciones'])
    
    #        print(f"grupos {grupos} {type(grupos)}")
    #        exit()
    #        centroids,_ = kmeans(datosNivel['soluciones'],len(datosNivel['soluciones']))
    #        grupos,_ = vq(datosNivel['soluciones'],centroids)
            datosNivel['grupos'] = grupos
        datosNivel['solPorGrupo'] = {}
        for idGrupo in datosNivel['grupos']:
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
        mejorVelGrupo = {}
        mejorSolucionGrupoBin = {}
        mejorEvaluacionGrupo = {}
#        if not 'mejoresVel' in datosNivel: datosNivel['mejoresVel'] = {}
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
#                datosNivel['mejoresVel'][idxSolucion] = datosNivel['velocidades'][idxSolucion]
            
            
            if (not datosNivel['grupos'][idxSolucion] in mejorEvaluacionGrupo 
                    or datosNivel['mejoresEvaluaciones'][idxSolucion] > mejorEvaluacionGrupo[datosNivel['grupos'][idxSolucion]]): 
                mejorEvaluacionGrupo[datosNivel['grupos'][idxSolucion]] = datosNivel['evalSoluciones'][idxSolucion]
                mejorSolucionGrupo[datosNivel['grupos'][idxSolucion]] = datosNivel['soluciones'] [idxSolucion]
                mejorVelGrupo[datosNivel['grupos'][idxSolucion]] = datosNivel['velocidades'] [idxSolucion]
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
        datosNivel['mejoresVel']  = mejorVelGrupo
        datosNivel['mejorSolGrupoBin']  = mejorSolucionGrupoBin
        datosNivel['mejorGlobal']  = mejorGlobal
#        print(f"datosNivel['mejorSolGrupo'] {datosNivel['mejorSolGrupo']}")
#        exit()
        end = datetime.now()
        self.guardarIndicadorTiempo('evaluarGrupos', total, end-start)
        if self.guardarDatosEjec:
            with open(self.nomArchivoDatosEjec, "a") as myfile:
                
                linea = f"{self.contenedorParametros['nivel']}"
                linea += f",{len(datosNivel['soluciones'])}"                
                linea += f",{self.contenedorParametros['accelPer']}"
                linea += f",{self.contenedorParametros['accelBest']}"
                linea += f",{self.contenedorParametros['inercia']}"
                linea += f",{self.contenedorParametros['mejorEvalGlobal']}"
                linea += f",{np.mean(datosNivel['evalSoluciones'])}"
                linea += f",{np.std(datosNivel['evalSoluciones'])}\n"
                myfile.write(linea)
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
        if self.contenedorParametros['mejorEvalGlobal'] is not None:
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
    
    
    
    def graficarParticulas(self, datosNivel, nivel):
#        print("ids de grupos ordenados de menor a mayor")
#        print(np.unique(datosNivel['grupos']))
#        print("indice soluciones del grupo 1")
#        print(datosNivel['grupos'] == 1)
#        print("indice soluciones del grupo 1")
#        print(datosNivel['soluciones'][np.where(datosNivel['grupos'] == 1)])
#        print("graficando")
#        exit()
#        if self.problema.getNombre() == "SCP":
#            parametrosGso = np.array([self.contenedorParametros['accelPer'],self.contenedorParametros['accelBest'],self.contenedorParametros['inercia']])
#            self.problema.graficarSol(datosNivel, parametrosGso, nivel, id = 0)
#            return
        plt.show()
        if self.mejores is None:
            self.mejores = {}
        if nivel not in self.niveles:
            
            self.niveles[nivel] = {}
            self.mejores[nivel] = {}
#        colores = None
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0,int(0Xcccccc)))
        self.colores = self.scaler.fit_transform(np.array([key for key in datosNivel['grupos']]).reshape((-1,1)))
#        else:
#            colores = self.scaler.transform(np.array([key for key in datosNivel['grupos']]).reshape((-1,1)))
        cont = 0
#        if nivel == 2:
#            print(self.contenedorParametros['datosNivel'][nivel-1]['mejorSolGrupo'])
##            print(self.contenedorParametros['datosNivel'][nivel-1]['mejorSolGrupo'][list(self.contenedorParametros['datosNivel'][nivel-1]['mejorSolGrupo'].values())[0]])
#            print(datosNivel['soluciones'])
#            print(datosNivel['mejorSolGrupo'])
#            print(datosNivel['grupos'])
#            print(datosNivel['soluciones'][np.where(datosNivel['grupos'] == 0)])
#            exit()
#        print(datosNivel['mejorSolGrupo'])
#        if nivel == 2:
#            print(f'NIVEL 2')
#            print(datosNivel['grupos'])
#            exit()
        
        for idGrupo in datosNivel['grupos']:
#            print(f'self.colores {self.colores}')
            color = self.colores[idGrupo]
#            print(hex(int(color)))
#            exit()
            color = str(hex(int(color)))[2:].upper()
            while len(color) < 6:
                color = '0' + color
            color = '#' + color
            cont += 1
            solsGrupo = datosNivel['soluciones'][np.where(datosNivel['grupos'] == idGrupo)]
            mejor = datosNivel['mejorSolGrupo'][idGrupo]
#            print(mejor)
#            exit()
            
            
            if idGrupo not in self.niveles[nivel]:
#                marker = f'${nivel}.{idGrupo}$'# if nivel == 1 else 'o'
                marker = 'o'
                markerSizeLvl = 5 if nivel == 1 else 14
                
                self.niveles[nivel][idGrupo], = self.ax.plot(solsGrupo[:,0], solsGrupo[:,1], marker=marker, linestyle='None', markersize=markerSizeLvl, color=color)
                self.mejores[nivel][idGrupo], = self.ax.plot(mejor[0], mejor[1], marker=marker, linestyle='None', markersize=10, color=color)
#                if self.mejorGlobal is not None:
#                    del self.mejorGlobal
            if self.mejorGlobal is None:
                self.mejorGlobal, = self.ax.plot(self.contenedorParametros['mejorSolGlobal'][0], self.contenedorParametros['mejorSolGlobal'][1], marker = '*', linestyle='None',  markersize=24, color='r')
#                self.niveles[nivel][idGrupo], = self.ax.plot(solsGrupo[:,0], solsGrupo[:,1], marker=marker, linestyle='None')
            
            self.niveles[nivel][idGrupo].set_xdata(solsGrupo[:,0])
            self.niveles[nivel][idGrupo].set_ydata(solsGrupo[:,1])
            self.mejores[nivel][idGrupo].set_xdata(mejor[0])
            self.mejores[nivel][idGrupo].set_ydata(mejor[1])
            self.mejorGlobal.set_xdata(self.contenedorParametros['mejorSolGlobal'][0])
            self.mejorGlobal.set_ydata(self.contenedorParametros['mejorSolGlobal'][1])
            
#            self.nivel1, = 
#            
#            if nivel == 1:
#                self.niveles[idGrupo] = datosGrupo[]
#                self.nivel1, = self.ax.plot(datosNivel['soluciones'][:,0], datosNivel['soluciones'][:,1], marker='o', linestyle='None') 
            
        
#        if self.nivel1 is None and nivel == 1:
#            
#            
#            
#            
#            self.nivel1, = self.ax.plot(datosNivel['soluciones'][                  :,0], datosNivel['soluciones'][:,1], marker='o', linestyle='None') 
#        if self.nivel2 is None and nivel == 2:
#            self.nivel2, = self.ax.plot(datosNivel['soluciones'][:,0], datosNivel['soluciones'][:,1], marker='o', linestyle='None') 
#        if self.mejores is None:
#            self.mejores, = self.ax.plot(datosNivel['soluciones'][:,0], datosNivel['soluciones'][:,1], marker='o', linestyle='None') 
#        if nivel == 1:
#            self.nivel1.set_xdata(datosNivel['soluciones'][:,0])
#            self.nivel1.set_ydata(datosNivel['soluciones'][:,1])
#        if nivel == 2:
#            self.nivel2.set_xdata(datosNivel['soluciones'][:,0])
#            self.nivel2.set_ydata(datosNivel['soluciones'][:,1])

        #DIBUJA LOS GRUPOS CON COLORES DISTINTOS
#        print(datosNivel['mejorSolGrupo'])
#        exit()
#        print(datosNivel['grupos'])
#        
#        exit()
#        if nivel == 1:
#            
##            for idGrupo in datosNivel['grupos']:
#                
#            self.nivel1.set_xdata(datosNivel['soluciones'][:,0])
#            self.nivel1.set_ydata(datosNivel['soluciones'][:,1])
#        if nivel == 2:
#            self.nivel2.set_xdata(datosNivel['soluciones'][:,0])
#            self.nivel2.set_ydata(datosNivel['soluciones'][:,1])
#        
#
#        self.mejores.set_xdata(datosNivel['soluciones'][:,0])
#        self.mejores.set_ydata(datosNivel['soluciones'][:1])
#
#        self.fig.canvas.draw()
#        self.fig.canvas.flush_events()
##        if self.fig is None:
##            self.fig = plt.figure()
##            self.ax = self.fig.add_subplot(111)
###            plt.show()
##            
##        
##        for nivel in range(self.contenedorParametros['niveles']):
##            nivel += 1            
##            
##            self.fig.canvas.draw()
##            self.fig.canvas.flush_events()