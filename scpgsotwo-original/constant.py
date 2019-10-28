class Const():
    def CARPETA_INSTANCIAS(self):
        return "instances"
    def CARPETA_RESULTADOS(self):
        return "resultados"
    def CARPETA_FINAL_RESULTADOS(self):
        return "resultados_final"
    def EPOCH_NUMBER(self):
        #EPmax - Maximum number of epochs
        return 3
    def ITERATION_1(self):
        #L1 - Numero de Iteraciones para el Primer Nivel de PSO
        return 50
    def ITERATION_2(self):
        #L2 - Número de Iteraciones para el Segundo Nivel de PSO
        return 250;
    def POP_SIZE(self):
        #N -  Size of set X_i
        return 10
    def SUB_POP(self):
        #M - Number of partitions of X
        return 5
    def X_MIN(self):
        return -5
    def X_MAX(self):
        return 5
    def TOT_EJEC(self):
        return 1
