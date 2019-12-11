from algoritmos.contenedores.ContenedorResultadosAlgoritmo import ContenedorResultadosAlgoritmo

class Solver():
    def __init__(self):
        self.resultados = ContenedorResultadosAlgoritmo()
    
    def setAlgoritmo(self, algoritmo):
        print(algoritmo)
        self.algoritmo = algoritmo
        pass
    
    def resolverProblema(self):
        print("resolviendo problema")
        parametros = self.algoritmo.getParametros()
        print(parametros)
        self.algoritmo.setParametros(parametros)
        self.algoritmo.generarSolucion()
        indicadores = self.algoritmo.getIndicadores()
        print(indicadores)
#        self.resultados.agregarResultado(self.algoritmo.getResultado())
        
    def getMejorResultado(self):
        return 'mejor resultado'
    
    def getMejorSolucion(self):
        return 'mejor solucion'
    
    def getTiempoEjecucion(self):
        return 'tiempo ejecución'
    
    def graficarConvergencia(self):
        print("generando grafico")