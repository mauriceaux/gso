#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 23:58:59 2020

@author: mauri
"""
import matplotlib.pyplot as plt
import numpy as np
import time
#from time import datetime

# use ggplot style for more sophisticated visuals
#plt.style.use('ggplot')

class Graficador:
    def __init__(self):
#        self.fig = plt.figure(figsize=(13,6))
#        self.fig = plt.figure()
#        self.ax = self.fig.add_subplot(111)
#        self.lines1  = None
#        self.ax = self.fig.add_subplot(111)
#        self.lines1  = None
        self.axs = []
        self.axsIds = []
        self.lines = []
        
        
    def add_plot(self,id):
        
        self.axsIds.append(id)
        self.lines.append(None)
#        self.lines1  = None
        
    def createPlot(self):
        self.fig, self.axs = plt.subplots(len(self.axsIds))
        for i in range(len(self.axsIds)):
            self.axs[i].set_title(self.axsIds[i])
            
    
    def live_plotter(self, x_vec,y1_data,identifier='',dotSize=1,marker='o',pause_time=0.1):
        limite=1000
        y1_data = y1_data[:limite]
        x_vec = x_vec[:limite]
#        print(f"np.where(np.array(self.axsIds) == identifier)[0][0] {np.where(np.array(self.axsIds) == identifier)[0].shape[0]}") 
#        exit()
        line1 = None
        if np.where(np.array(self.axsIds) == identifier)[0].shape[0] > 0:
            
            idLine1 = np.where(np.array(self.axsIds) == identifier)[0][0]
    #        exit()
    #        print(idLine1)
    #        exit()
            line1 = self.lines[idLine1]
            ax = self.axs[idLine1]
        if line1 is None:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
#            fig = plt.figure(figsize=(13,6))
#            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec,y1_data,'.')        
            #update plot label/title
#                plt.ylabel('Y Label')
            plt.title('{}'.format(self.axsIds[idLine1]))
            plt.show()
        
        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_ydata(y1_data)
        self.lines[idLine1] = line1
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
        
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        self.fig.canvas.flush_events() 
        time.sleep(pause_time)
        
        # return line so we can update it again in the next iteration
#        return line1
#
#def setData(data, line):
#    x = np.arange(data.shape[0])
#    y = data
#    if line is None:
#        plt.ion()
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        line, = ax.plot(x,y,marker='.',linestyle='None')
#        
#        plt.show()
#        
##        self.ax.set_ylim(np.min(data)-1,np.max(data)+1)
#    line.set_ydata(y)
##        print(y)
##        exit()
##        plt.cla()
#    
##        self.fig.canvas.draw()
#    plt.pause(0.1)
#    return line