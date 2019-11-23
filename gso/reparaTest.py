#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:05:46 2019

@author: mauri
"""

import reparastrategy as _repara
import repair.ReparaStrategy as _repara2
import read_instance as r_instance
from datetime import datetime
import numpy as np

def fObj(pos,costo):
    return np.sum(np.array(pos) * np.array(costo))

instance = r_instance.Read("instancesFinal/scpnrh5.txt")

repair = _repara.ReparaStrategy(instance.get_r()
                                    ,instance.get_c()
                                    ,instance.get_rows()
                                    ,instance.get_columns())

repair2 = _repara2.ReparaStrategy(instance.get_r()
                                    ,instance.get_c()
                                    ,instance.get_rows()
                                    ,instance.get_columns())

solucion = np.zeros(instance.get_columns())
#solucion[0] = 1

start = datetime.now()
reparada = solucion.copy()
#reparada = repair.repara_oneModificado(reparada)
reparada = repair.repara_one(reparada)

end = datetime.now()
obj = fObj(reparada, instance.get_c())
print(solucion)
print(reparada)
print(f'repara camilo demoro {end-start} evaluacion {obj} son iguales? {(solucion==reparada).all()}')

start = datetime.now()
reparada = solucion.copy().tolist()
print(f"cumple {repair2.cumple(reparada)}")
reparada = repair2.repara(reparada)
print(f"cumple {repair2.cumple(reparada)}")
end = datetime.now()
obj = fObj(reparada, instance.get_c())
#print(solucion)
#print(reparada)
print(f'repara lemus demoro {end-start} evaluacion {obj} son iguales? {(solucion==reparada).all()}')