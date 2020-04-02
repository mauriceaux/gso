#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:01:54 2020

@author: mauri
"""

from collections import Counter
import numpy as np
import math


class RankPerm():
    
    def rankperm(self,perm):
        perm = np.array(perm)
        rank = 1
        suffixperms = 1
        ctr = Counter()
        for i in range(perm.shape[0]):
            x = perm[((perm.shape[0] - 1) - i)]
            ctr[x] += 1
            for y in ctr:
                if (y < x):
                    rank += ((suffixperms * ctr[y]) // ctr[x])
            suffixperms = ((suffixperms * (i + 1)) // ctr[x])
        return rank
    
    def unrankperm(self,arr, rank):
        ctr = Counter()
        permcount = 1
        unos = np.count_nonzero(arr > 0)
        original = np.zeros(arr.shape)
        original[:unos] = 1
    #    totalPerm = math.factorial(arr.shape[0])/(math.factorial(unos))*math.factorial(arr.shape[0]-unos)
        
        totalPerm = self.rankperm(original)
        rank = rank if rank <= totalPerm else (rank % totalPerm)
        
        for i in range(arr.shape[0]):
            x = arr[i]
            ctr[x] += 1
            permcount = (permcount * (i + 1)) // ctr[x]
        # ctr is the histogram of letters
        # permcount is the number of distinct perms of letters
        perm = []
        for i in range(arr.shape[0]):
            for x in sorted(ctr.keys()):
                # suffixcount is the number of distinct perms that begin with x
                suffixcount = permcount * ctr[x] // (arr.shape[0] - i)
                if rank <= suffixcount:
                    perm.append(x)
                    permcount = suffixcount
                    ctr[x] -= 1
                    if ctr[x] == 0:
                        del ctr[x]
                    break
                rank -= suffixcount
        return np.array(perm)
    
    #tam = 10000
    #arr = np.random.randint(low=0,high=2, size=(tam))
    #print(f"caso {arr}")
    #unos = np.count_nonzero(arr > 0)
    #print(f"num unos {unos}")
    #original = np.zeros((tam))
    ##print(original)
    #original[-unos:] = 1
    #print(f"generador {original}")
    ##exit()
    #ranking = rankperm(arr)
    ##print(ranking)
    ##print(arr)
    #for i in range(100):
    ##    print(f"ranking {ranking+i}")
    #    print(unrankperm(original,ranking+i))
    ##print(rankperm('0001100011000110001100011000110001100011000110001100011000110001100011000110001100011000110001100011000110001100011000110001100011000110001100011000110001100011'))
    ##print(rankperm('00101'))
    ##print(rankperm('01001'))
    ##print(rankperm('00110'))
    ##print(rankperm('01010'))
    ##print(rankperm('01100'))
    ##print(unrankperm('00011',4))