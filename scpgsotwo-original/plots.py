#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:43:44 2019

@author: mauri
"""

import pandas as pd
import matplotlib.pyplot as plt

path1 = f'resultados/swarmL0S0.csv'
df1 = pd.read_csv(path1, header=None)

path2 = f'resultados/swarmL1S0.csv'
df2 = pd.read_csv(path2, header=None)

path3 = f'resultados/swarmL0S1.csv'
df3 = pd.read_csv(path3, header=None)

path4 = f'resultados/swarmL0S2.csv'
df4 = pd.read_csv(path4, header=None)

path5 = f'resultados/swarmL0S3.csv'
df5 = pd.read_csv(path5, header=None)

#print(df1.iloc[:,0])
level0, = plt.plot(df1.iloc[:,0], label='swarm level 0 0')

level2, = plt.plot(df3.iloc[:,0], label='swarm level 0 1')
#level3, = plt.plot(df4.iloc[:,0], label='swarm level 0 2')
#level4, = plt.plot(df5.iloc[:,0], label='swarm level 0 3')
level1, = plt.plot(df2.iloc[:,0], label='swarm level 1')
#plt.legend(handles=[level0, level1, level2, level3, level4])
plt.legend(handles=[level0, level1, level2])
#level0.set_label('swarm level 0')
#level1.set_label('swarm level 1')
#plt.plot(df3.iloc[:,0])
#plt.plot(df4.iloc[:,0])
#plt.plot(df5.iloc[:,0])
plt.show()