#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 23:33:59 2018

@author: cs
"""

import numpy as np
#DTYPE = np.double
#ctypedef np.double DTYPE_t

def SVMP(a, b):
    #cdef int rowCSR_Sum = <int*>malloc(sizeof(int))
    cdef float rowCSR_Sum = 0
    #print(a[1].getformat())
    
    resultCSR = np.empty_like(np.zeros(a.indptr.shape[0] - 1), dtype = 'd')
    #print(resultCSR)
    #print(str(len(a.indptr - 1)))
    cdef int i
    cdef int j
    for i in range(len(a.indptr) - 1):
        #print(len(a.indptr))
        for j in range(a.indptr[i], a.indptr[i + 1]):
            rowCSR_Sum += a.data[j] * b[a.indices[j]]
        resultCSR[i] = rowCSR_Sum
        rowCSR_Sum = 0
    return resultCSR



