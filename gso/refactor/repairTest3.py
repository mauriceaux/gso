import numpy as np
from datetime import datetime
from problemas.knapsack.knapsack import KP



start = datetime.now()

problema = KP(f'problemas/knapsack/instances/knapPI_3_10000_1000_1')
valores = problema.instance.itemValues
pesos = problema.instance.itemWeights
ponderacion = 1/(valores/(pesos+1))
capacidad = problema.instance.capacidad
NUM_FILAS = 50

print(pesos.shape)

origen = np.ones((NUM_FILAS, problema.instance.numItems))
print(f'origen {origen}')
pesosAplicados = origen * pesos
print(f'pesosAplicados {pesosAplicados}')
print(f'capacidad {capacidad}')

sumaPesos = np.sum(pesosAplicados, axis=1)
print(sumaPesos)
incumplidas = sumaPesos > capacidad
while (incumplidas).any():
    #print(f'incumplidas {incumplidas}')
    #exit()
    solucionesPonderadas = origen[incumplidas] * ponderacion
    #print(solucionesPonderadas)
    k=4
    peoresIndices = np.argpartition(-solucionesPonderadas,k,axis=1)[:,k-1::-1]
    #print(peoresIndices)
    peoresIndicesEliminar = np.random.randint(peoresIndices.shape[1], size=peoresIndices.shape[0])

    #print(peoresIndicesEliminar)

    #print(np.arange(len(peoresIndices)))
    #exit()



    indicesEliminar = peoresIndices[np.arange(len(peoresIndices)), peoresIndicesEliminar]

    #print(indicesEliminar)

    origen[incumplidas,indicesEliminar] = 0
    pesosAplicados = origen * pesos
    #print(f'pesosAplicados {pesosAplicados}')

    sumaPesos = np.sum(pesosAplicados, axis=1)
    #print(sumaPesos)
    incumplidas = sumaPesos > capacidad

print(origen)
print(sumaPesos)
print(f'origen tiene algun 1? {(origen==1).any()}')


end = datetime.now()

print(origen)
print(f'numpy arrays demoro {end-start}')

reparado = []
origen = np.ones((NUM_FILAS, problema.instance.numItems))
start = datetime.now()
for i in range(NUM_FILAS):
    reparado.append(problema.repairStrategy.repara(origen[i])[0])
print(np.array(reparado))
print(f'reparado tiene algun 1? {(np.array(reparado)==1).any()}')
print(f'reparado tiene algun 0? {(np.array(reparado)==0).any()}')
end = datetime.now()
print(f'metodo normal demoro {end-start}')