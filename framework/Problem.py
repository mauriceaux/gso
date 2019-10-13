#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:38:08 2019

@author: mauri
"""
import os
import pandas as pd
import multiprocessing
from multiprocessing import Process
from Executor import Executor
import numpy as np
from collections import OrderedDict


class Problem:
    def __init__(self):
        self.objs = []
        self.constrs = []
        self.Executor = Executor()
        self.objMap = {}
        self.ctrMap = {}
        pass
    
    def setDatos(self, path):
        if not os.path.exists(path):
            raise Exception(f"invalid path! {path}")
        self.data = pd.read_csv(path)
        
    def setObjs(self, objs):
        if not isinstance(objs, list):
            raise Exception(f"{objs} if not a python list!")
        self.objs = objs
    
    def addObj(self, obj):
        if not callable(obj):
            raise Exception(f"{obj} if not a callable!")
        self.objs.append(obj)
        
    def setConstr(self, costr):
        if not isinstance(costr, list):
            raise Exception(f"{costr} if not a python list!")
        self.constrs = costr
    
    def addConstr(self, constr):
        if not callable(constr):
            raise Exception(f"{constr} if not a callable!")
        self.constrs.append(constr)    
        
    def evalSolutions(self, solutions):
        solVecs = []
        for solution in solutions:
            solVecs.append(solution.getSolVec())
        executor = Executor()
        objs = executor.evalFns(solVecs, self.objs)
        constr = executor.evalFns(solVecs, self.constrs)
        idx = 0
        for solution in solutions:
            solution.setFitness(objs[idx])
            solution.setInfact(constr[idx])
            idx += 1
        return solutions
