#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 23:52:56 2020

@author: mauri
"""

from Graficador import Graficador
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import time

#graficador = Graficador()
##graficador.setTiempo(1000)
##graficador.inicio()
#graficador.add_plot("plot1")
##graficador.createPlot()
#line = None
#for i in range(100):
#    ydata = np.array([math.sin(i),math.cos(i),math.tan(i)])
#    line = graficador.live_plotter(np.arange(ydata.shape[0]),ydata,"plot1")
    
    
#from pylive import live_plotter
#import numpy as np
#
#size = 100
#x_vec = np.linspace(0,1,size+1)[0:-1]
#y_vec = np.random.randn(len(x_vec))
#line1 = []
#while True:
#    rand_val = np.random.randn(1)
#    y_vec[-1] = rand_val
#    line1 = live_plotter(x_vec,y_vec,line1)
#    y_vec = np.append(y_vec[1:],0.0)

ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line, = axes.plot(xdata, ydata, 'r-')
 
for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
 
# add this if you don't want the window to disappear at the end
plt.show()