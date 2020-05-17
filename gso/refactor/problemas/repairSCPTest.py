#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:04:23 2020

@author: mauri
"""

from scp.SCPProblem import SCPProblem
from scp.repair.ReparaStrategy2 import ReparaStrategy
import numpy as np




#
#restricciones = np.array([
#            [1,1,0,1]
#            , [1,0,0,0]
#            , [1,1,1,0]
#        ])
#
##expanded = np.expand_dims(matrix, axis=2)
#rexpanded = restricciones.copy()
#rexpanded = np.repeat(rexpanded, matrix.shape[0], axis=0).reshape((restricciones.shape[0],matrix.shape[0],matrix.shape[1]))
##print(rexpanded.shape)
##print(rexpanded)
#mult = rexpanded*matrix
#print(mult.shape)
#print(mult)
#suma = np.sum(mult, axis=2).T
#print(suma)
#producto = np.prod(suma, axis=1)
#print(producto)
#print((producto > 0))
#indicesIncumplidos = np.where(producto == 0)[0]
#print(indicesIncumplidos)
#exit()
#rexpanded = rexpanded.reshape((1, rexpanded.shape[0], rexpanded.shape[1]))


#pesos = np.array([1,2,3,4])
#
#sumaFilasRestriccion = np.sum(matrix, axis=1)
#print(sumaFilasRestriccion)
#ponderacion = matrix.T * sumaFilasRestriccion
#ponderacion = ponderacion.T
#print(ponderacion)
#ponderacion = np.sum(ponderacion, axis=0)
#print(ponderacion)
##ponderacion = 1/ponderacion
##print(ponderacion)
#ponderacion= ponderacion/pesos
#print(ponderacion)
#exit()
#problema = SCPProblem(f'scp/instances/scp410.txt')


















problema = SCPProblem(f'scp/instances/mscp41.txt')
#problema = SCPProblem(f'scp/instances/off/scpnrh5.txt')
#problema = SCPProblem(f'scp/instances/off/scp0.txt')

repara = ReparaStrategy(problema.instance.get_r()
                            ,problema.instance.get_c()
                            ,problema.instance.get_rows()
                            ,problema.instance.get_columns())
#sols = np.zeros((50, problema.getNumDim()), dtype=np.float)
#sols = [ [1.,1.,1.,0.,1.,1.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,1.,1.,1.,1.,1.,0.,1.,0.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,1.,0.,0.,1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,1.,0.,0.,0.,1.,1.,0.,0.,1.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]]
sols = [ [1.,1.,1.,0.,1.,1.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,1.,1.,1.,1.,1.,0.,1.,0.,1.,0.,0.,1.,1.,0.,0.,0.,1.,0.,1.,0.,0.,1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,1.,0.,1.,0.,1.,1.,0.,0.,1.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,1.,0.,0.,1.,0.,0.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]]
#sols = problema.generarSolsAlAzar(50)
sols = np.array(sols)
sols = repara.reparaBatch(sols)
fitness, decoded, _ = problema.evalDecBatch(sols, None)
#fitness, decoded, _ = problema.evalEnc(sols[1])
print(sols[0])
#print(fitness1[1])
print(problema.repair.cumple(sols[0]))
#print(decoded)
print(fitness[0])

exit()








filas = 50
l = 10
blocks = 3
plantilla = np.ones((filas*blocks, l)) * -1
print(f'plantilla.shape {plantilla.shape}')
random = np.random.randint(l, size=(filas, l))
print(f'random.shape {random.shape}')
for i in range(blocks):
    plantilla[i*filas:(i+1)*filas, :(i+1)*blocks] = random[:, :(i+1)*blocks]
    plantilla[i*filas:(i+1)*filas, (i+1)*blocks+1:] = random[:, 0].reshape(-1,1)
    
#print(plantilla[plantilla==-1].shape)
#exit()

print(plantilla)
print(plantilla.shape)
exit()








matrix = np.array([[1,1,1,0]
                    ,[1,1,1,0]
                    ,[0,0,1,0]
                    ,[0,0,1,1]
                    ,[1,0,0,1]])

indicesEliminar = np.array([
            [0,1,2]
            ,[0,1,2]
            ,[0,1,2]
            ,[0,1,2]
            ,[0,1,2]
        ])
blocks = 4
print(np.arange(4).reshape((1,4)))
indices = np.repeat(np.arange(4).reshape((1,4)), blocks, axis=0)
print(indices)
print(indices.shape)


#indicesEliminar = np.arange(matrix.shape[0] * blocks * indicesEliminar.shape[1]).reshape(matrix.shape[0] * blocks, indicesEliminar.shape[1])
indicesEliminar = np.random.randint(4, size=(matrix.shape[0]*blocks, indicesEliminar.shape[1]))

#print(indicesCandidatos)
#exit()


print(indicesEliminar)
print(indicesEliminar.shape)
exit()

expanded = matrix.copy()
expanded = np.repeat(expanded, blocks, axis=0)
#print(expanded)
print(expanded.shape)
#exit()  
#print(np.repeat(indicesEliminar, matrix.shape[0]))
#expanded[np.arange(len(indicesEliminar)).reshape((-1,1)), indicesEliminar] = 1
print(f'corregidas {expanded}')
#print(f'indices eliminar shape {indicesEliminar.reshape((blocks, matrix.shape[0], indicesEliminar.shape[1]))}')
cumple = np.random.randint(2, size=(indicesEliminar.shape[0]))
print(f'cumple {cumple}')
cumple = cumple.reshape( (blocks, matrix.shape[0], 1) )
print(f'cumple {cumple}')
elegidos = np.argmax(cumple, axis=0)
print(f'elegidos {elegidos}')
elegidos = elegidos.reshape((elegidos.shape[0]))
print(f'elegidos {elegidos}')
cumpleElegidos = expanded.copy()
cumpleElegidos = cumpleElegidos.reshape( (blocks, matrix.shape[0], cumpleElegidos.shape[1]) )
print(f'cumpleElegidos {cumpleElegidos}')
#cumpleElegidos = cumpleElegidos.reshape( (matrix.shape[0], blocks, cumpleElegidos.shape[2]) )
cumpleElegidos = cumpleElegidos[elegidos,np.arange(matrix.shape[0]),:]
print(f'cumpleElegidos {cumpleElegidos}')
#print(f'cumpleElegidos {cumpleElegidos[elegidos[0],0,:]}')
#print(f'cumpleElegidos {cumpleElegidos[elegidos[1],1,:]}')
#print(f'cumpleElegidos {cumpleElegidos[elegidos[2],2,:]}')
#print(f'cumpleElegidos {cumpleElegidos[elegidos[3],3,:]}')
#print(f'cumpleElegidos {cumpleElegidos[elegidos[4],4,:]}')
exit()
cumpleElegidos = cumpleElegidos[0, elegidos[0]]
#cumpleElegidos[:, np.arange(matrix.shape[0]).reshape((-1,1)), elegidos]
print(f'cumpleElegidos {cumpleElegidos}')
exit()
