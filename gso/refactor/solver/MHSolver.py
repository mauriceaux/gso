from algoritmos.contenedores.ContenedorResultadosAlgoritmo import ContenedorResultadosAlgoritmo
from agentes.optimizadorParametros import OptimizadorParametros
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class Solver():
    def __init__(self):
        self.resultados = ContenedorResultadosAlgoritmo()
        self.agente = OptimizadorParametros()
        self.autonomo = True
        self.algoritmo = None
    
    def setAlgoritmo(self, algoritmo):
        #print(algoritmo)
        self.algoritmo = algoritmo
        self.algoritmo.contenedorParametros['autonomo'] = self.autonomo
        self.agente.setParamDim(self.algoritmo.getParamDim())
        pass

    def setParamOptimizar(self, paramOptimizar):
        self.agente.setParamOptimizar(paramOptimizar)
    
    def resolverProblema(self):
        start = datetime.now()
        if self.algoritmo is None:
            raise Exception('No se ha definido el algoritmo')
        print(f"resolviendo problema {self.algoritmo.problema.instancia}")
#        parametros = self.algoritmo.getParametros()
#        print(parametros)
#        self.algoritmo.setParametros(parametros)
        if not self.autonomo:
            self.algoritmo.generarSolucion()
        else:
            #30 ejecuciones porque si

            for i in range(900):
    #            indicadores = self.algoritmo.getIndicadores()
#                print(f"solver inicio {self.algoritmo.getParametros()}")
                self.algoritmo.generarSolucionReducida()
                #if self.algoritmo.problema.optimo is not None and self.algoritmo.problema.optimo >= (-self.algoritmo.indicadores["mejorObjetivo"]):
                #    break
                self.algoritmo.setParametros(self.calcularParametrosAlgoritmo())
#                print(f"solver fin {self.algoritmo.getParametros()}")
                
                
                
        end = datetime.now()
        self.tiempoEjec = end-start
#        print(indicadores)
#        self.resultados.agregarResultado(self.algoritmo.getResultado())
        
    def calcularParametrosAlgoritmo(self):
        #datos de entrada
        self.agente.observarResultados(self.algoritmo.getParametros()
                                       ,self.algoritmo.indicadores
                                       )
        return self.agente.mejorarParametros()
            
            
    
    def getMejorResultado(self):
        return self.algoritmo.contenedorParametros['mejorEvalGlobal']
    
    def getMejorSolucion(self):
        return self.algoritmo.indicadores['mejorSolucion']
    
    def getTiempoEjecucion(self):
        return self.tiempoEjec
    
    def graficarConvergencia(self):
        #print(np.array(self.algoritmo.indicadores['mejoresResultadosReales']))
        plt.plot(np.array(self.algoritmo.indicadores['mejoresResultados']))
        plt.plot(np.array(self.algoritmo.indicadores['mejoresResultadosReales']))
        plt.plot(np.array(self.algoritmo.indicadores['mediaResultadosReales']))
        plt.show()
        print("generando grafico")