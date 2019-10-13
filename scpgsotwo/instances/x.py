# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:08:16 2019

@author: cavasquez
"""

# -*- coding: utf-8 -*-
def LeerInstancia(Instancia):
    EndIf    = None
    EndFor   = None
    EndWhile = None
        
    Archivo       = open(Instancia, "r")
        
    # Leer Dimensión
    Registro           = Archivo.readline().split()
    TotalRestricciones = int(Registro[0])
    TotalVariables     = int(Registro[1])
    
    # Leer Costo
    Costos        = []
    Registro      = Archivo.readline()
    ContVariables = 1
    while Registro != "" and ContVariables <= TotalVariables:
        Valores = Registro.split()
        for Contador in range(len(Valores)):
            Costos.append(int(Valores[Contador]))
            ContVariables = ContVariables + 1
        EndFor
        Registro = Archivo.readline()
    EndWhile
    
    # Preparar Matriz de Restricciones.
    Restricciones = []
    for Fila in range(TotalRestricciones):
        Restricciones.append([])
        for Columna in range(TotalVariables):
            Restricciones[Fila].append(0)
        EndFor
        print(str(len(Restricciones[Fila])))
    EndFor
            
    # Leer Restricciones    
    ContVariables      = 1
    Fila               = 0
    print(Registro)
    while Registro != "":
        CantidadValoresUno = int(Registro)
        ContadorValoresUno = 0
        Registro = Archivo.readline()
        while Registro != "" and ContadorValoresUno < CantidadValoresUno: 
            Columnas = Registro.split() 
            for Contador in range(len(Columnas)):
                Columna = int(Columnas[Contador]) - 1
                Restricciones[Fila][Columna] = 1
                ContadorValoresUno = ContadorValoresUno + 1
            EndFor
            print(str(len(Restricciones[Fila])))
            Registro = Archivo.readline()
        EndWhile   
        Fila = Fila + 1
    EndWhile   
    Archivo.close()
    return Costos, Restricciones


######## MAIN
    
C, R = LeerInstancia("scp41.txt")
#print("Costo:", C)
#print("Restricciones:", R)