#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:00:16 2019

@author: mauri
"""

from Problem import Problem
from Solution import Solution
import numpy as np

def obj1(vecSol, return_dict, pid):
    import time
    time.sleep(1)
    print('obj1')
    obj = 0
    for sol in vecSol:
        obj+= sol
    return_dict[pid] = obj
    return return_dict

def obj2(vecSol, return_dict, pid):
    print('obj2')
    obj = 0
    for sol in vecSol:
        obj-= sol
    return_dict[pid] = obj
    return return_dict
    
def obj3(vecSol, return_dict, pid):
    print('obj3')
    obj = 1
    for sol in vecSol:
        obj*= sol
    return_dict[pid] = obj
    return return_dict

def obj4(vecSol, return_dict, pid):
    print('obj4')
    obj = 1
    for sol in vecSol:
        obj/= sol
    return_dict[pid] = obj
    return return_dict

def c1(vecSol, return_dict, pid):
    print('c1')
    c = 0
    for sol in vecSol:
        c+= sol
    
    return_dict[pid] = c < 50
    return return_dict

def c2(vecSol, return_dict, pid):
    print('c2')
    c = 0
    for sol in vecSol:
        c+= sol
    
    return_dict[pid] = c > 0
    return return_dict

def c3(vecSol, return_dict, pid):
    print('c3')
    c = 0
    for sol in vecSol:
        c+= sol
    
    return_dict[pid] = c != 20
    return return_dict

def c4(vecSol, return_dict, pid):
    print('c4')
    c = 0
    for sol in vecSol:
        c+= sol
#    return c != 10
    return_dict[pid] = c != 10
    return return_dict

p = Problem()
objs = []
objs.append(obj1)
objs.append(obj2)
objs.append(obj3)
objs.append(obj4)
constrs = []
constrs.append(c1)
constrs.append(c2)
constrs.append(c3)
constrs.append(c4)
p.setObjs(objs)
p.setConstr(constrs)
solVec = []
for i in range(10):
    solution = Solution()
    solution.solVec = np.random.randint(low=1, high=100, size=10)
    solVec.append(solution)
#print(np.array(solVec).shape)
#exit()
solVec = p.evalSolutions(solVec)
for solution in solVec:
    print(f'solution {solution.solVec} has fittnes {solution.f} and infactibility {solution.i}')