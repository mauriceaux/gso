import numpy as np
#from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
import multiprocessing.dummy as mp
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sqlalchemy as db
import json
import pickle
import zlib


class GSO():
    def __init__(self, niveles=2, idInstancia=datetime.timestamp(datetime.now()) ,numParticulas=50, iterPorNivel={1:50,2:250}, gruposPorNivel={1:12,2:12}):
        self.idInstancia = idInstancia
        self.contenedorParametros = {}
        self.contenedorParametros['niveles'] = niveles
        self.contenedorParametros['nivel'] = 2
        self.contenedorParametros['numParticulas'] = numParticulas
        self.contenedorParametros['iterPorNivel'] = iterPorNivel
        self.contenedorParametros['gruposPorNivel'] = gruposPorNivel
        self.contenedorParametros['porcentajePartPorGrupoNivel'] = {1:0.1, 2:1}
        self.contenedorParametros['datosNivel'] = {}
        self.contenedorParametros['mejorEvalGlobal'] = None
        self.contenedorParametros['mejorSolucion'] = None
        self.contenedorParametros['mejorSolucionBin'] = None
        self.contenedorParametros['mejorSolGlobal'] = None
        self.contenedorParametros['accelPer'] = {}
        self.contenedorParametros['accelBest'] = {}
        self.contenedorParametros['maxVel'] = 5
        self.contenedorParametros['minVel'] = -5
        self.contenedorParametros['autonomo'] = True
        self.contenedorParametros['solPorGrupo'] = {}
        self.contenedorParametros['inercia'] = {}
        self.contenedorParametros['numIteraciones'] = 1
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
        self.fig = plt.figure()
        self.mostrarGraficoParticulas = False
        self.nivel1=None
        self.nivel2=None
        self.mejores = None
        self.mejorGlobal = None
        self.ax = self.fig.add_subplot(111)
        self.niveles = {}
        self.scaler = None
        self.plotShowing = False
        self.solPromedio = None
        self.mostrarPromedio = True
        self.geometric = False
        self.guardarDatosEjec = True
        self.nomArchivoDatosEjec = f"ejecucion{datetime.now()}.csv"
        
        pass
    
    def getIndicadores(self):
        return self.indicadores
    
    def getParametros(self):
        nivel = self.contenedorParametros['nivel']
        estEvol = None
        if nivel in self.contenedorParametros['datosNivel']:
            estEvol = self.contenedorParametros['datosNivel'][nivel]['estEvol']
        return {
                'nivel':nivel
                ,'solPorGrupo':self.contenedorParametros['solPorGrupo']
                ,'iterPorNivel':self.contenedorParametros['iterPorNivel']
                ,'inercia':self.contenedorParametros['inercia']
                ,'accelPer':self.contenedorParametros['accelPer']
                ,'accelBest':self.contenedorParametros['accelBest']
                ,'estEvol':estEvol
                }
    
    def setParametros(self, parametros):
        for key in parametros:
            self.contenedorParametros[key] = parametros[key]
    
    def setProblema(self, problema):
        self.problema = problema
        self.indicadores['problema'] = problema.getNombre()
        self.indicadores['numDimProblema'] = problema.getNumDim()
        self.indicadores['numLlamadasFnObj'] = 0
        self.problema.paralelo = self.procesoParalelo
    
    def moveSwarmGeometric(self, swarm, velocity, personalBest, bestFound, inertia):
        if len(swarm.shape) == 1:
            swarm = np.array([swarm])
        indicesOcupados = np.zeros(swarm.shape)
        idx = 0
        for indice in indicesOcupados:
            disponible = np.where(indice==0)[0]
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
        maxVal = self.problema.getRangoSolucion()['max']
        minVal = self.problema.getRangoSolucion()['min']
#        randPer = np.random.uniform(low=0, high=1)
#        randBest = np.random.uniform(low=0, high=1)
        randPer = 1
        randBest = 1
        personalDif = personalBest - swarm
        personalAccel = accelPer * randPer * personalDif
        bestDif = bestFound - swarm
        bestAccel = accelBest * randBest * bestDif
        acceleration =  personalAccel + bestAccel
        acceleration[acceleration > maxVel]  = maxVel
        acceleration[acceleration < minVel]  = minVel
        vInercia = inertia*velocity
        vInercia[vInercia > maxVel]  = maxVel
        vInercia[vInercia < minVel]  = minVel
        nextVel = vInercia + acceleration
        nextVel[nextVel > maxVel]  = maxVel
        nextVel[nextVel < minVel]  = minVel
        ret = swarm+nextVel
        ret[ret > maxVal]  = maxVal
        ret[ret < minVal] = minVal
        return ret, nextVel
    
    def aplicarMovimiento(self, datosNivel, iteracion, totIteraciones):
        start = datetime.now()
        args = []
        start = datetime.now()
        if self.contenedorParametros['autonomo']:
            inercia = self.contenedorParametros['inercia'][self.contenedorParametros['nivel']][datosNivel['grupos'][0]]
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
                ,self.contenedorParametros['accelPer'][self.contenedorParametros['nivel']][datosNivel['grupos'][idx]]
                ,self.contenedorParametros['accelBest'][self.contenedorParametros['nivel']][datosNivel['grupos'][idx]]
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
        totalGrupos = {}
        for idGrupo in np.unique(datosNivel['grupos']):
            totalGrupos[idGrupo] = np.count_nonzero(datosNivel['grupos'] == idGrupo)
        self.agregarDataEjec(evaluaciones, datosNivel['grupos'])
        end=datetime.now()
        resultadoMovimiento['soluciones'] = np.vstack(np.array(ret)[:,0])
        resultadoMovimiento['soluciones'][resultadoMovimiento['soluciones'] == 1] = self.problema.getRangoSolucion()['max']
        resultadoMovimiento['soluciones'][resultadoMovimiento['soluciones'] == 0] = self.problema.getRangoSolucion()['min']
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
            for sol in soluciones:
                if not self.geometric: eval, bin, _ = self.problema.evalEnc(sol)
                else: eval, bin, _ = self.problema.eval(sol)
                solucionesBin.append(bin)
                evaluaciones.append(eval)
            solucionesBin = np.array(solucionesBin)
            evaluaciones = np.array(evaluaciones)
        
        if self.mostrarPromedio:
            tmp = np.mean(soluciones, axis=0)
            if self.solPromedio is None:
                self.solPromedio = tmp
            else:
                
                self.solPromedio = np.append(self.solPromedio, tmp).reshape(2, tmp.shape[0])
                self.solPromedio = np.mean(self.solPromedio, axis=0)
        end = datetime.now()
        self.guardarIndicadorTiempo('evaluarSoluciones', len(soluciones), end-start)
        return solucionesBin, evaluaciones.reshape((evaluaciones.shape[0]))
    
    def agregarDataEjec(self, evaluaciones, idGrupos):
        self.agregarMejorResultado()
        self.dataEvals.append(evaluaciones.tolist())
        self.totalEvals = len(self.dataEvals)
        self.indicadores['numLlamadasFnObj'] += evaluaciones.shape[0]
        totalEvalsGrupo = {}
        for idGrupo in np.unique(idGrupos):
            evalsGrupo = evaluaciones[np.argwhere(idGrupos==idGrupo)]
            totalEvalsGrupo[idGrupo] = evalsGrupo.shape
            if not 'mejoresResultadosReales' in self.indicadores:
                self.indicadores['mejoresResultadosReales'] = {}
            if not self.contenedorParametros['nivel'] in self.indicadores['mejoresResultadosReales']:
                self.indicadores['mejoresResultadosReales'][self.contenedorParametros['nivel']] = {}
            if not idGrupo in self.indicadores['mejoresResultadosReales'][self.contenedorParametros['nivel']]:
                self.indicadores['mejoresResultadosReales'][self.contenedorParametros['nivel']][idGrupo] = []
            self.indicadores['mejoresResultadosReales'][self.contenedorParametros['nivel']][idGrupo].append(np.max(evalsGrupo))
            if not 'mediaResultadosReales' in self.indicadores:
                self.indicadores['mediaResultadosReales'] = {}
            if not self.contenedorParametros['nivel'] in self.indicadores['mediaResultadosReales']:
                self.indicadores['mediaResultadosReales'][self.contenedorParametros['nivel']] = {}
            if not idGrupo in self.indicadores['mediaResultadosReales'][self.contenedorParametros['nivel']]:
                self.indicadores['mediaResultadosReales'][self.contenedorParametros['nivel']][idGrupo] = []
            self.indicadores['mediaResultadosReales'][self.contenedorParametros['nivel']][idGrupo].append(np.mean(evalsGrupo))        
    
    def generarSolucion(self):
        if self.mostrarGraficoParticulas:
            plt.ion()
            plt.show()
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
                self.contenedorParametros['datosNivel'][nivel] = self.agruparNivel(datosNivel, nivel)
                if self.mostrarGraficoParticulas:
                    self.graficarParticulas(datosNivel, nivel)#
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
        self.fin = datetime.now()
        self.indicadores['tiempoEjecucion'] = self.fin-self.inicio
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']
        
    def generarSolucionReducida(self):
        engine = db.create_engine('postgresql://mh:mh@localhost:5432/resultados_mh')
        metadata = db.MetaData()
        connection = engine.connect()
        datosIteracion = db.Table('datos_iteracion', metadata, autoload=True, autoload_with=engine)
        insertDatosIteracion = datosIteracion.insert()
        if 1 in self.contenedorParametros['datosNivel']:
            totalesGrupo = {}
            nivel1 = self.contenedorParametros['datosNivel'][1]
            for idGrupo in nivel1['solPorGrupo']:
                total = np.count_nonzero(nivel1['grupos'] == idGrupo)
                dif = self.contenedorParametros['solPorGrupo'][idGrupo] - total
                if dif != 0:
                    self.agregarEliminarParticulas(dif, idGrupo)
                totalesGrupo[idGrupo] = np.count_nonzero(self.contenedorParametros['datosNivel'][1]['grupos'] == idGrupo)
        nivel = self.contenedorParametros['nivel']
        if not nivel in self.contenedorParametros['datosNivel'] or (self.nivelAnterior == 1 and nivel == 2): 
            self.contenedorParametros['datosNivel'][nivel] = self.generarNivel(nivel)
        datosNivel = self.contenedorParametros['datosNivel'][nivel]
        self.nivelAnterior = nivel
        self.inicio = datetime.now()
        
        if self.mostrarGraficoParticulas and not self.plotShowing:
            plt.ion()
            plt.show()
            self.plotShowing = True
            
        print(f'ACTUALIZANDO NIVEL '+ str(nivel))
        data = []
        for iteracion in range(self.contenedorParametros['numIteraciones']):
            inicio = datetime.now()
            resultadoMovimiento = self.aplicarMovimiento(datosNivel, iteracion, self.contenedorParametros['numIteraciones'])
            strPromedio = f" promedio {np.mean(resultadoMovimiento['evalSoluciones'])} " if self.mostrarPromedio else ''
            string = f'nivel {nivel} iteracion {iteracion} mejor valor encontrado {self.contenedorParametros["mejorEvalGlobal"]} {strPromedio} num particulas {datosNivel["soluciones"].shape[0]}'
            print(string)
            datosNivel['soluciones']     = resultadoMovimiento['soluciones']
            datosNivel['solucionesBin']  = resultadoMovimiento['solucionesBin']
            datosNivel['evalSoluciones'] = resultadoMovimiento['evalSoluciones']
            datosInternos = {
                'datosNivel' : datosNivel,
                'parametros' : self.contenedorParametros
            }
            datosInternos = zlib.compress(pickle.dumps(datosInternos))
            
            if not self.geometric: datosNivel['velocidades']    = resultadoMovimiento['velocidades']
            self.contenedorParametros['datosNivel'][nivel] = self.evaluarGrupos(datosNivel)
            self.calcularEstadoEvolutivo(datosNivel)
            #print(datosNivel['estEvol'])
            if self.mostrarGraficoParticulas:
                self.graficarParticulas(datosNivel, nivel)
            fin = datetime.now()
            data.append({
                'id_ejecucion' : self.idInstancia
                ,'fitness_mejor' : -self.contenedorParametros['mejorEvalGlobal']
                ,'fitness_promedio' : -np.mean(datosNivel['evalSoluciones'])
                ,'fitness_mejor_iteracion' : -np.max(datosNivel['evalSoluciones'])
                ,'inicio' : inicio
                ,'fin' : fin
                ,'parametros_iteracion' : json.dumps({'nivel': nivel})
                ,'datos_internos' : datosInternos})
            #datosConvergencia.append([self.idInstancia, nivel, self.contenedorParametros['mejorEvalGlobal'], np.mean(datosNivel['evalSoluciones']), (fin-inicio).total_seconds()])
        #with open(f"{self.carpetaResultados}{'/autonomo' if self.contenedorParametros['autonomo'] else ''}/convergencia{self.instancia}inercia.csv", "a") as myfile:
        #    for linea in datosConvergencia:
        #        mejorSolStr = ','.join([str(item) for item in linea])
        #        myfile.write(f'{mejorSolStr}\n')
        connection.execute(insertDatosIteracion, data)
        

        self.fin = datetime.now()
        
        self.indicadores['tiempoEjecucion'] = self.fin-self.inicio
        self.indicadores['mejorObjetivo'] = self.contenedorParametros['mejorEvalGlobal']
        self.indicadores['mejorSolucion'] = self.contenedorParametros['mejorSolucionBin']
#        input("Press Enter to continue...")


    def agregarEliminarParticulas(self, dif, idGrupo):
        nivel1 = self.contenedorParametros['datosNivel'][1]
        if dif > 0:            
            solucionesBin, evaluaciones = self.generarSolucionAlAzar(dif)            
            mejoresSoluciones = np.array([self.contenedorParametros['mejorSolGlobal'] for _ in range(dif)])
            grupos = [idGrupo for _ in range(dif)]
            mejoresSolucionesBin, mejoresEvaluaciones = self.evaluarSoluciones(mejoresSoluciones)
            velocidades = np.random.uniform(low=self.contenedorParametros['minVel'], high=self.contenedorParametros['maxVel'], size=(dif, self.problema.getNumDim()))
            mejoresVelocidades = velocidades.copy()
            soluciones = solucionesBin.copy()
            soluciones[soluciones == 0] = self.problema.getRangoSolucion()['min']
            soluciones[soluciones == 1] = self.problema.getRangoSolucion()['max']
            
            mejoresSoluciones = mejoresSolucionesBin.copy()
            mejoresSoluciones[mejoresSoluciones == 0] = self.problema.getRangoSolucion()['min']
            mejoresSoluciones[mejoresSoluciones == 1] = self.problema.getRangoSolucion()['max']
            
            
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
            nsoluciones = list(nivel1['soluciones'])
            nsolucionesBin = list(nivel1['solucionesBin'])
            ngrupos = list(nivel1['grupos'])

            nmejoresEvaluaciones.extend(mejoresEvaluaciones)
            nmejoresSoluciones.extend(mejoresSoluciones)
            nmejoresSolucionesBin.extend(mejoresSolucionesBin)
            nevalSoluciones.extend(evaluaciones)
            nvelocidades.extend(velocidades)
#            nMejoresVelocidades.extend(mejoresVelocidades)
            nsoluciones.extend(soluciones)
            nsolucionesBin.extend(solucionesBin)
            ngrupos.extend(grupos)
            nivel1['mejoresEvaluaciones'] = np.array(nmejoresEvaluaciones)
            nivel1['mejoresSoluciones'] = np.array(nmejoresSoluciones)
            nivel1['mejoresSolucionesBin'] = np.array(nmejoresSolucionesBin)
            nivel1['evalSoluciones'] = np.array(nevalSoluciones)
            nivel1['velocidades']    = np.array(nvelocidades)
            nivel1['soluciones']     = np.array(nsoluciones)
            nivel1['solucionesBin']  = np.array(nsolucionesBin)
            nivel1['grupos']         = np.array(ngrupos)
            
        else:
            while np.count_nonzero(nivel1['grupos'] == idGrupo) > self.contenedorParametros['solPorGrupo'][idGrupo]:
                idxPeor = np.argwhere(nivel1['grupos']==idGrupo)[np.argmin(nivel1['evalSoluciones'][np.argwhere(nivel1['grupos']==idGrupo)])][0]
                if(idGrupo != nivel1['grupos'][idxPeor]): exit()
                mejoresEvaluaciones = nivel1['mejoresEvaluaciones'].tolist()
                mejoresSoluciones = nivel1['mejoresSoluciones'].tolist()
                mejoresSolucionesBin = nivel1['mejoresSolucionesBin'].tolist()
                evalSoluciones = nivel1['evalSoluciones'].tolist()
                velocidades = nivel1['velocidades'].tolist()
                soluciones = nivel1['soluciones'].tolist()
                solucionesBin = nivel1['solucionesBin'].tolist()
                grupos = nivel1['grupos'].tolist()

                del mejoresEvaluaciones[idxPeor]
                del mejoresSoluciones[idxPeor]
                del mejoresSolucionesBin[idxPeor]
                del evalSoluciones[idxPeor]
                del velocidades[idxPeor]
                del soluciones[idxPeor]
                del solucionesBin[idxPeor]
                del grupos[idxPeor]
                nivel1['mejoresEvaluaciones'] = np.array(mejoresEvaluaciones)
                nivel1['mejoresSoluciones'] = np.array(mejoresSoluciones)
                nivel1['mejoresSolucionesBin'] = np.array(mejoresSolucionesBin)
                nivel1['evalSoluciones'] = np.array(evalSoluciones)
                nivel1['velocidades']    = np.array(velocidades)                
                nivel1['soluciones']     = np.array(soluciones)
                nivel1['solucionesBin']  = np.array(solucionesBin)
                nivel1['grupos']         = np.array(grupos)
        self.contenedorParametros['numParticulas'] += dif


    def generarSolucionAlAzar(self, numSols):
        start = datetime.now()
        sols, evals = self.problema.generarSolsAlAzar(numSols, self.contenedorParametros['mejorSolGlobal'])
        end = datetime.now()
        self.guardarIndicadorTiempo('generarSolucionAlAzar', numSols, end-start)
        return sols, evals
        
                        
    def generarNivel(self, nivel):
        start = datetime.now()
        totalNivel = 0
        if nivel == 1:
            totalNivel = self.contenedorParametros['numParticulas']
            solucionesBin, evaluaciones = self.generarSolucionAlAzar(totalNivel)
            mejoresSolucionesBin, mejoresEvaluaciones = self.generarSolucionAlAzar(totalNivel)
            velocidades = np.random.uniform(low=self.contenedorParametros['minVel'], high=self.contenedorParametros['maxVel'], size=(totalNivel, self.problema.getNumDim()))
            soluciones = solucionesBin.copy()
            divisor=1
            soluciones[soluciones == 0] = self.problema.getRangoSolucion()['min']/divisor
            soluciones[soluciones == 1] = self.problema.getRangoSolucion()['max']/divisor
            
            mejoresSoluciones = mejoresSolucionesBin.copy()
            mejoresSoluciones[mejoresSoluciones == 0] = self.problema.getRangoSolucion()['min']/divisor
            mejoresSoluciones[mejoresSoluciones == 1] = self.problema.getRangoSolucion()['max']/divisor
            
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
            soluciones = np.array([nivelAnterior['mejorSolGrupo'][key] for key in nivelAnterior['mejorSolGrupo'] if len(nivelAnterior['mejorSolGrupo'][key]) > 0])
            velocidades = np.array([nivelAnterior['mejoresVel'][key] for key in nivelAnterior['mejoresVel'] if len(nivelAnterior['mejoresVel'][key]) > 0])
            
            grupos = np.array([1 for _ in nivelAnterior['mejorSolGrupo'] ])

            totalNivel = len(soluciones)
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
            datosNivel['solucionesBin']  = solsBin
            datosNivel['grupos'] = grupos
            solPorGrupo = {}
            for grupo in grupos:
                solPorGrupo[grupo] = np.count_nonzero(grupos == grupo)
            datosNivel['solPorGrupo'] = solPorGrupo
            datosNivel = self.evaluarGrupos(datosNivel)
        end = datetime.now()
        self.guardarIndicadorTiempo('generarNivel', totalNivel, end-start)
        for idGrupo in datosNivel['grupos']:
#            AGREGO PARAMETROS POR GRUPO SI NO EXISTEN
            if not nivel in self.contenedorParametros['inercia']:
                self.contenedorParametros['inercia'][nivel] = {}
            if not idGrupo in self.contenedorParametros['inercia'][nivel]:
                self.contenedorParametros['inercia'][nivel][idGrupo] = 0.1
            if not nivel in self.contenedorParametros['accelPer']:
                self.contenedorParametros['accelPer'][nivel] = {}
            if not idGrupo in self.contenedorParametros['accelPer'][nivel]:
                self.contenedorParametros['accelPer'][nivel][idGrupo] = 1.06
            if not nivel in self.contenedorParametros['accelBest']:
                self.contenedorParametros['accelBest'][nivel] = {}
            if not idGrupo in self.contenedorParametros['accelBest'][nivel]:
                self.contenedorParametros['accelBest'][nivel][idGrupo] = 1.06

        return datosNivel
    
    def agruparNivel(self, datosNivel, nivel):
        datosNivel['soluciones'] = datosNivel['soluciones'].astype('float64')
        totalNivel = len(datosNivel['soluciones'])
        start = datetime.now()        
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
                numGrupos = 1
            kmeans = KMeans(n_clusters=numGrupos, init='k-means++')
            grupos = kmeans.fit_predict(datosNivel['evalSoluciones'].reshape(-1,1))
            datosNivel['grupos'] = grupos
        datosNivel['solPorGrupo'] = {}
        for idGrupo in datosNivel['grupos']:
            if not idGrupo in datosNivel['solPorGrupo']:
                datosNivel['solPorGrupo'][idGrupo] = 1
            else:
                datosNivel['solPorGrupo'][idGrupo] += 1
        
        
        self.contenedorParametros['solPorGrupo'] = datosNivel['solPorGrupo']
        
        datosNivel = self.evaluarGrupos(datosNivel)
        end = datetime.now()
        self.guardarIndicadorTiempo('agruparNivel', totalNivel, end-start)
        return datosNivel    
        
    def calcularNumGrupos(self, soluciones):
        wcss = []
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
        return distances.index(max(distances)) + 2

    def calcularEstadoEvolutivo(self, datosNivel):
        if not 'estEvol' in datosNivel: datosNivel['estEvol'] = {}
        grupos = np.unique(datosNivel['grupos'])
        for grupo in grupos:
            if not grupo in datosNivel['estEvol']: datosNivel['estEvol'][grupo] = []
            posGrupo = np.argwhere(datosNivel['grupos'] == grupo)
            solsGrupo = datosNivel['soluciones'][posGrupo]
            if len(solsGrupo) <= 2:
                #print(f"grupo {grupo} tiene {len(solsGrupo)} elementos!!!!!")
                datosNivel['estEvol'][grupo].append(0)
                continue

            evalsGrupo = datosNivel['mejoresEvaluaciones'][posGrupo]
#            print(f"evalsGrupo {evalsGrupo}")
            
            mejorPos = np.argmax(evalsGrupo)
#            print(f"mejorPos {mejorPos}")
#            mejorSol = solsGrupo[mejorPos]
            distMejor = self.calcDistProm(mejorPos, solsGrupo)
#            print(f"distMejor {distMejor}")
            distMin = distMejor
            distMax = distMejor
            
            for idxSol in range(len(solsGrupo)):
                if idxSol == mejorPos: continue
                #print(f'idxSol {idxSol}')
                distSol = self.calcDistProm(idxSol, solsGrupo)
#                print(f"distSol {idxSol} = {distSol}")
                if distMin is None or distSol < distMin:
                    distMin = distSol
                if distMax is None or distSol > distMax:
                    distMax = distSol
            #print(f'grupo {grupo} elementos grupo {len(solsGrupo)} distMejor {distMejor} distMin {distMin} distMax {distMax} ')
            estEvol = (distMejor-distMin)/(distMax-distMin)
#            print(f"distMejor {distMejor}")
#            print(f"distMin {distMin}")
#            print(f"distMax {distMax}")
            if np.isnan(estEvol) or (distMax-distMin) == 0: 
                
                estEvol = 0
#            print(f"estEvol {estEvol}")
            datosNivel['estEvol'][grupo].append(estEvol)

    def calcDistProm(self, idxSol, solsGrupo):
        res = 0
        for idx in range(len(solsGrupo)):
            if idx == idxSol: continue
#            res += np.linalg.norm(solsGrupo[idx]-solsGrupo[idxSol])
            res += np.sum(np.abs(solsGrupo[idx]-solsGrupo[idxSol]))
        return res/len(solsGrupo)-1

            
    def evaluarGrupos(self, datosNivel):
        start = datetime.now()
        mejorSolucionGrupo = {}
        mejorVelGrupo = datosNivel['mejoresVel'] if 'mejoresVel' in datosNivel else {}
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
        if self.problema.getNombre() == "SCP":
            parametrosGso = np.array([self.contenedorParametros['accelPer'],self.contenedorParametros['accelBest'],self.contenedorParametros['inercia']])
            self.problema.graficarSol(datosNivel, parametrosGso, nivel, id = 0)
            return
        if self.mejores is None:
            self.mejores = {}
        if nivel not in self.niveles:
            
            self.niveles[nivel] = {}
            self.mejores[nivel] = {}
#        colores = None
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0,int(0Xcccccc)))
        self.colores = self.scaler.fit_transform(np.array([key for key in datosNivel['grupos']]).reshape((-1,1)))
        cont = 0
        
        for idGrupo in datosNivel['grupos']:
            color = self.colores[idGrupo]
            color = str(hex(int(color)))[2:].upper()
            while len(color) < 6:
                color = '0' + color
            color = '#' + color
            cont += 1
            solsGrupo = datosNivel['soluciones'][np.where(datosNivel['grupos'] == idGrupo)]
            mejor = datosNivel['mejorSolGrupo'][idGrupo]
            
            if idGrupo not in self.niveles[nivel]:
                marker = 'o'
                markerSizeLvl = 5 if nivel == 1 else 14
                
                self.niveles[nivel][idGrupo], = self.ax.plot(solsGrupo[:,0], solsGrupo[:,1], marker=marker, linestyle='None', markersize=markerSizeLvl, color=color)
                self.mejores[nivel][idGrupo], = self.ax.plot(mejor[0], mejor[1], marker=marker, linestyle='None', markersize=10, color=color)
            if self.mejorGlobal is None:
                self.mejorGlobal, = self.ax.plot(self.contenedorParametros['mejorSolGlobal'][0], self.contenedorParametros['mejorSolGlobal'][1], marker = '*', linestyle='None',  markersize=24, color='r')
            
            self.niveles[nivel][idGrupo].set_xdata(solsGrupo[:,0])
            self.niveles[nivel][idGrupo].set_ydata(solsGrupo[:,1])
            self.mejores[nivel][idGrupo].set_xdata(mejor[0])
            self.mejores[nivel][idGrupo].set_ydata(mejor[1])
            self.mejorGlobal.set_xdata(self.contenedorParametros['mejorSolGlobal'][0])
            self.mejorGlobal.set_ydata(self.contenedorParametros['mejorSolGlobal'][1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()