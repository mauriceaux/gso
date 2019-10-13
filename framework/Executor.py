#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:30:54 2019

@author: mauri
"""
from multiprocessing import Process, Queue
import multiprocessing
import numpy as np
from collections import OrderedDict

class Executor():
    def __init__(self):
        self.pid = 0
        pass
    
    def _genPrss(self, solVec, fns, return_dict):
        ps = []
        prcMap = {}
        cont = 0
        for vec in solVec:
            for fn in fns:
                ps.append(Process(target=fn, args=[vec, return_dict, self.pid]))
                prcMap[self.pid] = cont
                self.pid+=1
            cont += 1
        return np.array(ps), prcMap
    
    def evalFns(self, solVecs, fns):
        ret = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes, prcMap = self._genPrss(solVecs, fns, return_dict)
        self.process(processes)
        items_returned = OrderedDict(sorted(return_dict.items()))
        for key, value in items_returned.items():
            
            if key in prcMap:
                try:
                    ret[prcMap[key]].append(value)
                except:
                    ret.insert(prcMap[key], [])
                    ret[prcMap[key]].append(value)
        return ret
        
    
    def process(self, prcsList):
        prcsNp = np.array(prcsList)
        
        shape = prcsNp.shape
        prcsNp = prcsNp.reshape((np.prod(shape)))
        
        
#        print(prcsNp.shape)
#        exit()
        
#        print(prcsNp)
#        exit()
        proc = []
        for fn in prcsNp:
#            print(fn.shape)
#            exit()
            
#            print(1)
            p = fn
#            print(2)
            p.start()
            #                exit()
#            print(3)
            proc.append(p)
#    print(4)
        for p in proc:
            p.join()
        procNp = np.array(proc).reshape(shape)
        print(f'multithread execution end')
#        print(procNp)
        return procNp
