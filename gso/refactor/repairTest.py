from numba import cuda, float32
import numpy as np
import math
from datetime import datetime


from problemas.knapsack.knapsack import KP
TPB = 2

@cuda.jit
def repara(soluciones, ponderacion, pesos, capacidad, resultado):
    #print('******************************************')
    #print(cuda.grid(2))
    #print(cuda.threadIdx.x, cuda.threadIdx.y)
    numDim  = ponderacion.shape[0]
    sSoluciones = cuda.shared.array(shape=(TPB,numDim), dtype=float32)
    sResultado = cuda.shared.array(shape=(TPB,numDim), dtype=float32)
    sPesos = cuda.shared.array(shape=(numDim), dtype=float32)

    sCumple = cuda.shared.array(shape=(numDim), dtype=float32)
    sPonderacion = cuda.shared.array(shape=(numDim), dtype=float32)
    sPesos = cuda.shared.array(shape=(TPB, numDim), dtype=float32)
    sSumaPesos = cuda.shared.array(shape=(TPB), dtype=float32)
    sCapacidad = cuda.shared.array(shape=(1), dtype=float32)
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    #print(f'tx==x {tx==x}')
    #print(f'ty==y {tx==x}')
    #print(x)
    #return
    bpg = cuda.gridDim.x 
    #print(bpg)
    #return
    #print(cuda.grid(2))
    #print(ponderacion.shape)
    #print(resultado.shape)
    #print(bpg)
    #return
    if x >= resultado.shape[0] or y >= resultado.shape[1]:
        print(f'fuera de rango')
        # Quit if (x, y) is outside of valid C boundary
        return

    #for i in range(bpg):
    sSoluciones[tx, y] = soluciones[x, y]
    if x == 0:
        sPonderacion[y] = ponderacion[y]
        sCapacidad[0] = capacidad[0]

    sPesos[tx, y] = pesos[y] * sSoluciones[tx, y]
    #print(sSoluciones.shape)
    #return
    sResultado[tx,y] = sSoluciones[tx, y]*sPonderacion[y]
    #print(f'sincronizacion 1 hilo {x}, {y}')
    sSumaPesos[tx] += sPesos[tx, y]
    cuda.syncthreads()
    
        #for j in range(numDim):
    
    #cuda.syncthreads()

    if ty == 0:
        #sumaPesos = np.sum(sPesos[tx])
        cont = 0
        while sumaPesos > sCapacidad[0]:
            candEliminar = np.argsort(-sResultado[tx])[:5]
            #candEliminar = sResultado.argsort()[-5:][::-1]
            #print(f'hilo {x}, {y} candEliminar {candEliminar}')
            idxEliminar = np.random.choice(candEliminar)
            #print(f'hilo {x}, {y} idxEliminar {idxEliminar}')
            sSoluciones[tx, idxEliminar] = 0
            sPesos[tx, idxEliminar] = 0
            sResultado[tx, idxEliminar] = 0
            sumaPesos = np.sum(sPesos[tx])
            cuda.syncthreads()
            #print(f'hilo {x}, {y} sumaPesos {sumaPesos} > sCapacidad[0] {sCapacidad[0]} {sumaPesos > sCapacidad[0]}')
    
    #print(f'sincronizacion 2 hilo {x}, {y}')
    cuda.syncthreads()
    

    resultado[x,y] = sSoluciones[tx, ty]



start = datetime.now()
problema = KP(f'problemas/knapsack/instances/off/knapPI_1_5000_1000_1')

valores = problema.instance.itemValues
pesos = problema.instance.itemWeights
ponderacion = 1/(valores/(pesos+1))
#print(ponderacion.shape)

input = np.ones((100, problema.instance.numItems))
#print(input.shape)

A_global_mem = cuda.to_device(input)
B_global_mem = cuda.to_device(ponderacion)
C_global_mem = cuda.to_device(pesos)
D_global_mem = cuda.to_device(np.array([problema.instance.capacidad]))
E_global_mem = cuda.device_array((100, problema.instance.numItems))
#print(A_global_mem)

tb = (TPB, problema.instance.numItems)
bg = int(math.ceil(100 / tb[0]))
#print(bg)
#exit()

repara[bg, tb](A_global_mem, B_global_mem, C_global_mem, D_global_mem, E_global_mem)
res = E_global_mem.copy_to_host()
end = datetime.now()

print(res)
print(f'demoro {end-start}')