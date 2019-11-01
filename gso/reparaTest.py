#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:05:46 2019

@author: mauri
"""

import reparastrategy as _repara
import read_instance as r_instance
from datetime import datetime
import numpy as np

instance = r_instance.Read("instances/mscpnrh5.txt")

repair = _repara.ReparaStrategy(instance.get_r()
                                    ,instance.get_c()
                                    ,instance.get_rows()
                                    ,instance.get_columns())

solucion = np.zeros(instance.get_columns())
solucion[0] = 1

start = datetime.now()
reparada = solucion.copy()
reparada = repair.repara_one(reparada)
end = datetime.now()
print(solucion)
print(reparada)
print(f'demoro {end-start} son iguales? {(solucion==reparada).all()}')