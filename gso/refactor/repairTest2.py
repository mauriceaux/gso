import numba
from numba import cuda, float32
import numpy as np
import math
from datetime import datetime


from problemas.knapsack.knapsack import KP
TPB = 2
BD = 1
problema = KP(f'problemas/knapsack/instances/knapPI_1_500_1000_1')
NUM_DIM = problema.getNumDim()
NUM_FILAS = 50
ANCHO_COL = 5
CAPACIDAD = problema.instance.capacidad

@cuda.reduce
def sum_reduce(a, b):
    return a + b

"""
@cuda.jit('void(float32[:,:], float32[:], float32[:,:])', device=False)
def hacerCero(matrizSols, itemsEliminar, res):
    solucionesc = cuda.shared.array(shape=(BD,NUM_DIM), dtype=float32)
    itemsEliminarc = cuda.shared.array(shape=BD, dtype=float32)
#    resc = cuda.shared.array(shape=(cuda.blockDim.x, 1))

    posGx, posGy = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if posGx > res.shape[0] or posGy > matrizSols.shape[1]:
        return
    
    solucionesc[tx,ty] = matrizSols[posGx,posGy]
    itemsEliminarc[tx] = itemsEliminar[posGx]
    
    cuda.syncthreads()
    
    solucionesc[tx,itemsEliminarc[ty]] = 0
    res[posGx,posGy] = solucionesc[tx,ty]
""" 
    

@cuda.jit('void(float32[:,:], float32, boolean[:])', device=False)
def cumple(matrizSols, capacidad, res):
    
    solucionesc = cuda.shared.array(shape=(BD,NUM_DIM), dtype=float32)
    resc = cuda.shared.array(shape=(BD,), dtype=float32)
#    tmp = cuda.shared.array(shape=(BD, 1), dtype=float32)

    posGx, posGy = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if posGx > res.shape[0] or posGy > matrizSols.shape[1]:
        return
    
#    bloqueX = cuda.blockIdx.x
#    dimBloqueX = cuda.blockDim.x

    solucionesc[tx,ty]= matrizSols[posGx,posGy]
#    capacidadc[tx] = capacidad[0]
#    print(capacidad[0])
    resc[tx] = 0

    cuda.syncthreads()

    #expect = solucionesc.sum()      # numpy sum reduction
#    print(resc[tx])
#    print(solucionesc[tx,ty])
    resc[tx] = resc[tx] + solucionesc[tx,ty]   # cuda sum reduction
    cuda.syncthreads()
#    resc[tx] = resc[tx] <= capacidadc[0]
    #assert expect == got
    #if ty == 0:
    #    tmp = 0
    #    for i in range(cuda.blockDim.y):
    #        resc[tx] += solucionesc[tx,i]
#    cuda.syncthreads()

    

    res[posGx] = capacidad - resc[tx] 



@cuda.jit('void(float32[:,:], float32[:], float32[:,:])', device=False)
#@cuda.jit
def mMult(matrizSols, pond, res):
    solucionesc = cuda.shared.array(shape=(BD,NUM_DIM), dtype=float32)
    pondc = cuda.shared.array(shape=(NUM_DIM), dtype=float32)
    resc = cuda.shared.array(shape=(BD,NUM_DIM), dtype=float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= res.shape[0] or y >= res.shape[1]:
        print(f'fuera de rango')
        # Quit if (x, y) is outside of valid C boundary
        return
    #print(f'{x},{y}')
    solucionesc[tx,ty] = matrizSols[x,y]
#    factor = pond[y]
    pondc[ty] = pond[y]

    cuda.syncthreads()

    resc[tx,ty] = (pondc[ty] * solucionesc[tx,ty])
    cuda.syncthreads()

    res[x,y] = resc[tx,ty]

@cuda.jit('void(float32[:,:], float32[:,:])', device=False)
def selMayorCuda(matrizSols, res):
    solucionesc = cuda.shared.array(shape=(NUM_FILAS,ANCHO_COL), dtype=float32)
    resc = cuda.shared.array(shape=(NUM_FILAS,int(math.ceil(NUM_DIM/ANCHO_COL))), dtype=float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bi = cuda.blockIdx.x

    if x >= matrizSols.shape[0] or y >= matrizSols.shape[1] or bi >= res.shape[1]:
        print(f'fuera de rango')
        # Quit if (x, y) is outside of valid C boundary
        return

    print(f'{tx},{ty}')
    solucionesc[tx,ty] = matrizSols[x,y]
    resc[tx,bi] = -1
    cuda.syncthreads()

    resc[tx,bi] = solucionesc[tx,ty] if solucionesc[tx,ty] > resc[tx,bi] else resc[tx,bi]
    cuda.syncthreads()

    res[x,bi] = resc[tx,bi]



def aplicarPonderacion(matriz, ponderacion):
    #MULTIPLICA LA lista de soluciones por la ponderacion
    matrizSols = cuda.to_device(matriz)
    pond = cuda.to_device(ponderacion)
    fls, clms = matriz.shape
    #tpb = 100
    res = cuda.device_array((fls, clms))    
    
    tb = (BD, clms)
    bg = int(math.ceil(fls / tb[0]))
    mMult[bg, tb](matrizSols, pond, res)
    return res.copy_to_host()

def aplicarPesos(matriz, pesos):
    #MULTIPLICA LA lista de soluciones por la ponderacion
    matrizSols = cuda.to_device(matriz)
    pond = cuda.to_device(pesos)
    fls, clms = matriz.shape
    #tpb = 10
    res = cuda.device_array((fls, clms))    
    
    tb = (BD, clms)
    bg = int(math.ceil(fls / tb[0]))
#    print(numba.typeof(matrizSols))
    mMult[bg, tb](matrizSols, pond, res)
    return res.copy_to_host()

def eliminarItems(matriz, itemsEliminar):
    matrizSols = cuda.to_device(matriz)
    items = cuda.to_device(ponderacion)
    fls, clms = matriz.shape
    #tpb = 100
    res = cuda.device_array((fls, clms))    
    
    tb = (BD, clms)
    bg = int(math.ceil(fls / tb[0]))
    hacerCero[bg, tb](matrizSols, itemsEliminar, res)
    return res.copy_to_host()

def selMayorFuzzy(oPonderado):
    #selecciona los bg mayores elementos
    matrizSols = cuda.to_device(oPonderado)
    fls, clms = oPonderado.shape
    #tpb = 10
    tb = (fls,ANCHO_COL)
    bg = int( math.ceil(clms/ANCHO_COL))
    res = cuda.device_array((fls, bg))
    selMayorCuda[bg, tb](matrizSols, res)
    print(res.copy_to_host())
    return res.copy_to_host()


def obtenerIncumplidas(matriz, capacidad):
    #revisa que cada fila de la matriz cumpla
#    print(matriz.shape)
    smatrizSols = cuda.to_device(matriz.astype('float32'))
#    scapacidad = cuda.to_device(capacidad)
    fls, clms = matriz.shape
    #tpb = 1
    res = cuda.device_array((fls))    
    
    tb = (BD, clms)
    bg = int(math.ceil(fls / tb[0])) if fls >= tb[0] else 1
#    print(f'hola')
    #print(f'bg {bg}')
    #print(f'tb {tb}')
    #print(f'smatrizSols {smatrizSols.copy_to_host().shape} {smatrizSols.copy_to_host().dtype}')
#    print(f'scapacidad {scapacidad.copy_to_host()[0]} {scapacidad.copy_to_host().dtype}')
    #print(f'res {res.copy_to_host().shape} {res.copy_to_host().dtype}')
    #print(CAPACIDAD)
    cumple[bg, tb](smatrizSols, CAPACIDAD, res)
#    print(f'chao')
    return res.copy_to_host()

start = datetime.now()


valores = problema.instance.itemValues
pesos = problema.instance.itemWeights
ponderacion = 1/(valores/(pesos+1))
capacidad = problema.instance.capacidad


print(pesos.shape)

origen = np.ones((NUM_FILAS, problema.instance.numItems))
print(f'origen {origen}')
pesosAplicados = aplicarPesos(origen, pesos)
print(f'pesosAplicados {pesosAplicados}')
incumplidas = obtenerIncumplidas(pesosAplicados, capacidad)

#print(incumplidas)
#incumplidas[incumplidas<0.] = False
#incumplidas[incumplidas>=0.] = True
print(f'incumplidas {incumplidas<0.}')
while incumplidas.size > 0:
    restantes = origen[incumplidas<0]
    oPonderado = aplicarPonderacion(restantes, ponderacion)
    print(f'oPonderado {oPonderado}')
    itemsEliminar = selMayorFuzzy(oPonderado)
    print(f'itemsEliminar {itemsEliminar}')
    exit()
    #restantes = eliminarItems(restantes, itemsEliminar)
    origen[incumplidas] = restantes
    incumplidas = obtenerIncumplidas(restantes, capacidad)
    #incumplidas[incumplidas==0.] = False
    #incumplidas[incumplidas==1.] = True
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
