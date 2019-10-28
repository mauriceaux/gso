#!/usr/bin/python
# encoding=utf8
import os
import read_instance as r_instance
import gso as _gso
import constant as _cont
import numpy as np
constantes = _cont.Const();
import datetime

def func1_old(pos,costo):
    total=0 
    for i in range(len(pos)):
        if (pos[i]==1):
            total+=costo[i] 
    return total

def func1(pos,costo):
      return np.sum(np.array(pos) * np.array(costo))

dim=1000
tot_ejecuciones = constantes.TOT_EJEC()+1

tTransferencia=[]
tTransferencia.append("sShape1")
#tTransferencia.append("sShape2")
#tTransferencia.append("sShape3")
#tTransferencia.append("sShape4")
#tTransferencia.append("vShape1")
#tTransferencia.append("vShape2")
#tTransferencia.append("vShape3")
#tTransferencia.append("vShape4")

tBinarizacion=[]
tBinarizacion.append("Standar")
#tBinarizacion.append("Complement")
#tBinarizacion.append("StaticProbability")
#tBinarizacion.append("Elitist")

bounds=[(constantes.X_MIN(),constantes.X_MAX())]
#listaArchivos = ['mscp41.txt']
listaArchivos = ['mscpnrh5.txt']

for filename in os.listdir(constantes.CARPETA_INSTANCIAS()):
    if ".txt" in filename and filename in listaArchivos:
        print(filename)
        read_instance = r_instance.Read(constantes.CARPETA_INSTANCIAS() + "/" +filename)
        print('rows: ' + str(read_instance.get_rows()))
        print('columns: ' + str(read_instance.get_columns()))
        #print(read_instance.get_r())
        dim=read_instance.get_columns()
        for tBinary in range(0,len(tBinarizacion)):
            for tTrans in range(0,len(tTransferencia)):
                    
                ruta_resultados_finales = constantes.CARPETA_FINAL_RESULTADOS() + "/" + filename.replace('.txt','') + "_" + tTransferencia[tTrans] + "_" + tBinarizacion[tBinary]
                if os.path.exists(ruta_resultados_finales):
                    os.remove(ruta_resultados_finales)
                filefinal = open(ruta_resultados_finales, "x")
                filefinal.close()
                
                for ejecucion in range(1,tot_ejecuciones):
                    ruta_resultados_ejecucion = constantes.CARPETA_RESULTADOS() + "/" + str(ejecucion)  + "_" + filename.replace('.txt','') + "_" + tTransferencia[tTrans] + "_" + tBinarizacion[tBinary]
                    if os.path.exists(ruta_resultados_ejecucion):
                        os.remove(ruta_resultados_ejecucion)                
                    fileejecucion = open(ruta_resultados_ejecucion, "x")
                    fileejecucion.close()
                    
                    print('INICIO eje: ' + str(ejecucion)  + " insta: " + filename.replace('.txt','') + " tTransf: " + tTransferencia[tTrans] + " tBinary: " + tBinarizacion[tBinary] + " tiempo {0:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))   
                    gso = _gso.GSO(func1,dim, bounds, read_instance.get_c(), read_instance.get_r(),ruta_resultados_ejecucion,ruta_resultados_finales,ejecucion,read_instance.get_rows(),read_instance.get_columns(),tTransferencia[tTrans],tBinarizacion[tBinary])
                    print('FIN eje   : ' + str(ejecucion)  + " insta: " + filename.replace('.txt','') + " tTransf: " + tTransferencia[tTrans] + " tBinary: " + tBinarizacion[tBinary] + " tiempo {0:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())) 
                
                if (tBinarizacion[tBinary]=="Kmeans"):
                    break
