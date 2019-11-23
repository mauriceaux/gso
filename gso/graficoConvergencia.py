#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:21:32 2019

@author: mauri
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

folder ='mejoresResultados/'
#filename = '234549.scpnrg1.txt' 
#filename = '220516.scpnre1.txt'
for filename in os.listdir(folder):
    print(filename)
    if filename.endswith(".txt"):
        data = pd.read_csv(folder+filename, header=None)
        sorted_data = data.sort_values(0)
        print(sorted_data)
        
        
        plt.plot(sorted_data.iloc[:,1].values)
        plt.plot(sorted_data.iloc[:599,1].values)
        plt.title(f'Convergencia instancia {filename[7:]}')
        plt.show()