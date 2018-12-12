#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:46:33 2018

@author: cs
"""


import numpy as np
#cimport numpy as np
from cython.parallel import prange
from threading import Lock, Thread
#from libc.math import fabs

#DTYPE = np.float

lock = Lock()
g = 0

T = []
for k in range(4):
    t = Thread(name = 'thread' + str(k)).start()
    

def SVMP(a, b):
    #cdef int rowCSR_Sum = <int*>malloc(sizeof(int))
    
             
    cdef float rowCSR_Sum = 0
    #print(a[1].getformat())
    
    resultCSR = np.empty_like(np.zeros(a.indptr.shape[0] - 1), dtype = 'd')
    #cdef np.ndarray resultCSR = np.zeros(a.indptr.shape[0] -1)
    #print(resultCSR)
    #print(str(len(a.indptr - 1)))
    cdef int i
    cdef int j
    cdef int N = len(a.indptr) - 1
    with nogil, parallel(num_threads = 4):
        for i in prange(N, schedule = "dynamic"):
        #print(i)
        #print(len(a.indptr))
        
            #print("Reached here")
            c = a.indptr[i]
            d = a.indptr[i + 1]
            for j in range(c, d):
                rowCSR_Sum += a.data[j] * b[a.indices[j]]    
            resultCSR[i] = rowCSR_Sum
            rowCSR_Sum = 0

    return resultCSR


#for x in range(len(T)):
    #T[x].join()
    




