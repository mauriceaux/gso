#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:43:44 2019

@author: mauri
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import os

directory = 'resultados/'
path = []
for filename in os.listdir(directory):
    if not filename.endswith(".csv"): continue
    path.append(directory + filename)
handles = []
for filename in path:
#    print(filename)
#    file = filename
    df = pd.read_csv(filename, header=None)
#    print(df.iloc[:,0])
#    exit()
    l = plt.plot(df.iloc[:,0], label=f'swarm {filename}')
    handles.append(l)
#path2 = f'resultados/swarmL1S0.csv'
#df2 = pd.read_csv(path2, header=None)
#
#path3 = f'resultados/swarmL0S1.csv'
#df3 = pd.read_csv(path3, header=None)
#
#path4 = f'resultados/swarmL0S2.csv'
#df4 = pd.read_csv(path4, header=None)
#
#path5 = f'resultados/swarmL0S3.csv'
#df5 = pd.read_csv(path5, header=None)

#print(df1.iloc[:,0])
#level0, = plt.plot(df1.iloc[:,0], label='swarm level 0 0')

#level2, = plt.plot(df3.iloc[:,0], label='swarm level 0 1')
#level3, = plt.plot(df4.iloc[:,0], label='swarm level 0 2')
#level4, = plt.plot(df5.iloc[:,0], label='swarm level 0 3')
#level1, = plt.plot(df2.iloc[:,0], label='swarm level 1')
#plt.legend(handles=[level0, level1, level2, level3, level4])
#plt.legend(handles=handles)
#level0.set_label('swarm level 0')
#level1.set_label('swarm level 1')
#plt.plot(df3.iloc[:,0])
#plt.plot(df4.iloc[:,0])
#plt.plot(df5.iloc[:,0])
plt.show()
