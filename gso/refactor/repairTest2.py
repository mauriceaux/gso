from numba import cuda, float32
import numpy as np
import math
from datetime import datetime


from problemas.knapsack.knapsack import KP
TPB = 2

@cuda.reduce
def sum_reduce(a, b):
    return a + b

@cuda.jit
def hacerCero(matrizSols, itemsEliminar, res):
    solucionesc = cuda.shared.array(shape=(cuda.blockDim.x,matrizSols.shape[1]), dtype=float32)
    pondc = cuda.shared.array(shape=pond.shape, dtype=float32)
    resc = cuda.shared.array(shape=(cuda.blockDim.x, 1))

    posGx, posGy = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if posGx > res.shape[0] or posGy > matrizSols.shape[1]:
        return

@cuda.jit
def cumple(matrizSols, capacidad, res):
    solucionesc = cuda.shared.array(shape=(cuda.blockDim.x,matrizSols.shape[1]), dtype=float32)
    capacidadc = cuda.shared.array(shape=capacidad.shape, dtype=float32)
    resc = cuda.shared.array(shape=(cuda.blockDim.x, 1), dtype=float32)

    posGx, posGy = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if posGx > res.shape[0] or posGy > matrizSols.shape[1]:
        return
    
    bloqueX = cuda.blockIdx.x
    dimBloqueX = cuda.blockDim.x

    solucionesc[tx,ty]= matrizSols[posGx,posGy]
    capacidadc[0] = capacidad[0]

    cuda.syncthreads()

    #expect = solucionesc.sum()      # numpy sum reduction
    resc[tx] = sum_reduce(solucionesc[tx])   # cuda sum reduction
    #assert expect == got
    #if ty == 0:
    #    tmp = 0
    #    for i in range(cuda.blockDim.y):
    #        resc[tx] += solucionesc[tx,i]
    cuda.syncthreads()

    

    res[posGx] = resc[tx] <= capacidadc[0]



@cuda.jit
def mMult(matrizSols, pond, res):
    solucionesc = cuda.shared.array(shape=(cuda.blockDim.x,matrizSols.shape[1]), dtype=float32)
    pondc = cuda.shared.array(shape=(matrizSols.shape[1],1), dtype=float32)
    resc = cuda.shared.array(shape=(cuda.blockDim.x,matrizSols.shape[1]), dtype=float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= res.shape[0] or y >= res.shape[1]:
        print(f'fuera de rango')
        # Quit if (x, y) is outside of valid C boundary
        return

    solucionesc[tx,ty] = matrizSols[x,y]
    pondc[ty] = pond[y]

    cuda.syncthreads()

    resc[tx,ty] = pondc[ty] * solucionesc[tx]
    cuda.syncthreads()

    res[x,y] = resc[tx,ty]

@cuda.jit
def selMayorCuda(matrizSols, res):
    solucionesc = cuda.shared.array(shape=(cuda.blockDim.x,matrizSols.shape[1]), dtype=float32)
    resc = cuda.shared.array(shape=(matrizSols.shape[0],1), dtype=float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= res.shape[0] or y >= res.shape[1]:
        print(f'fuera de rango')
        # Quit if (x, y) is outside of valid C boundary
        return

    solucionesc[tx,ty] = matrizSols[x,y]
    resc[tx] = -1
    cuda.syncthreads()

    resc[tx][0] = solucionesc[tx,ty] if solucionesc[tx,ty] > resc[tx][0] else resc[tx][0]
    cuda.syncthreads()

    res[x] = resc[tx]



    #print('******************************************')
    #print(cuda.grid(2))
    #print(cuda.threadIdx.x, cuda.threadIdx.y)
    """
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
    """





def aplicarPonderacion(matriz, ponderacion):
    #MULTIPLICA LA lista de soluciones por la ponderacion
    matrizSols = cuda.to_device(matriz)
    pond = cuda.to_device(ponderacion)
    fls, clms = matriz.shape
    tpb = 100
    res = cuda.device_array((fls, clms))    
    
    tb = (tpb, clms)
    bg = int(math.ceil(fls / tb[0]))
    mMult[bg, tb](matrizSols, pond, res)
    return res.copy_to_host()

def eliminarItems(matriz, itemsEliminar):
    matrizSols = cuda.to_device(matriz)
    items = cuda.to_device(ponderacion)
    fls, clms = matriz.shape
    tpb = 100
    res = cuda.device_array((fls, clms))    
    
    tb = (tpb, clms)
    bg = int(math.ceil(fls / tb[0]))
    hacerCero[bg, tb](matrizSols, itemsEliminar, res)
    return res.copy_to_host()

def selMayorFuzzy(oPonderado):
    #selecciona los bg mayores elementos
    matrizSols = cuda.to_device(oPonderado)
    fls, clms = oPonderado.shape
    tpb = 10
    tb = (fls,tpb)
    bg = int( math.ceil(filas/tpb))
    res = cuda.device_array((fls, bg))
    selMayorCuda[bg, tb](matrizSols, res)
    return res.copy_to_host()


def obtenerIncumplidas(matriz, capacidad):
    #revisa que cada fila de la matriz cumpla
    smatrizSols = cuda.to_device(matriz)
    scapacidad = cuda.to_device(np.array([capacidad]))
    fls, clms = matriz.shape
    tpb = 100
    res = cuda.device_array((fls, 1))    
    
    tb = (tpb, clms)
    bg = int(math.ceil(fls / tb[0])) if fls >= tb[0] else 1
    cumple[bg, tb](smatrizSols, scapacidad, res)
    return res.copy_to_host()

start = datetime.now()
problema = KP(f'problemas/knapsack/instances/knapPI_1_500_1000_1')

valores = problema.instance.itemValues
pesos = problema.instance.itemWeights
ponderacion = 1/(valores/(pesos+1))
capacidad = problema.instance.capacidad


#print(ponderacion.shape)

origen = np.ones((100, problema.instance.numItems))
incumplidas = obtenerIncumplidas(origen, capacidad)
print(incumplidas)
while incumplidas.size > 0:
    restantes = origen[incumplidas]
    oPonderado = aplicarPonderacion(restantes, ponderacion)

    itemsEliminar = selMayorFuzzy(oPonderado)
    restantes = eliminarItems(restantes, itemsEliminar)
    origen[incumplidas] = restantes
    incumplidas = obtenerIncumplidas(restantes, capacidad)
end = datetime.now()

print(origen)
print(f'demoro {end-start}')

"""
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
"""
