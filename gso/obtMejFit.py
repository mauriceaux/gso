#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:38:08 2019

@author: mauri
"""

import pandas as pd
import os
import numpy as np

directory = 'resultadosFinal/'
res = []
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".csv"):
        
        data = pd.read_csv(directory+filename, header=None)
        res.append([np.max(data.iloc[:,0].values),filename])
#        print(np.max(data.iloc[:,0].values))
#        exit()
        
for item in res:
    print(f'{item[0]}\t{item[1]}')