# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:02:23 2018

@author: Rohit
"""
import numpy as np

# Scenario 1: 
a = np.matrix('1 1 0 0 0 0 0 0 ; 1 1 1 1 1 1 0 0 ; 0 0 0 2 1 0 2 1')
b = np.matrix('1 1 1 1 1 1 1 1 ').T

ab = a*b

rowSum = 0
result = []
for i in range(a.shape[0]):
    for j in range(b.shape[0]):
        rowSum += a[i,j]*b[j]
    result.append(rowSum)
    rowSum = 0

I = [0,1,0,1,2,3,4,5,3,4,6,7]
V = [1,1,1,1,1,1,1,1,2,1,2,1]
P = [0,2,8,12]

rowCSR_Sum = 0
resultCSR = []
for i in range(len(P)-1):
    pointerList = list(range(P[i], P[i+1]))    
    for j in pointerList:
        rowCSR_Sum += V[j]*b[I[j]]
    resultCSR.append(rowCSR_Sum)
    rowCSR_Sum = 0
 
# Scenario 2: 
a = np.matrix('1 1 0 0 0 0 0 0 2; 1 1 1 1 1 1 0 0 2; 0 0 0 2 1 0 2 1 2')
b = np.matrix('1 1 1 1 1 1 1 1 2').T

I = [0,1,8,0,1,2,3,4,5,8,3,4,6,7,8]
V = [1,1,2,1,1,1,1,1,1,2,2,1,2,1,2]
P = [0,3,10,15]

# Scenario 2:
rowCSR_Sum = 0
resultCSR = []
for i in range(len(P)-1):
    pointerList = list(range(P[i], P[i+1]))    
    for j in pointerList:
        rowCSR_Sum += V[j]*b[I[j]]
    resultCSR.append(rowCSR_Sum)
    rowCSR_Sum = 0
        
  
#pointerDict = dict(zip(list(range(P[i], P[i+1])), list(range(P[i+1] - P[i]))))     
    
#dictCSR = dict(zip([0,1,0,1,2,3,4,5,3,4,6,7], [1,1,1,1,1,1,1,1,2,1,2,1]))

# =============================================================================
# list1 = [1,2,3,4]
# list2 = [1,2,3,4]
# list3 = ['a','b','c','d']
# =============================================================================
# CSR_Dict = dict(zip(zip(I,V),P))

# create dictioanary between I and V
# multiply dict.value(for k in I) and b(j)

# =============================================================================
# #for i in V:
# rowCSR_Sum = 0
# resultCSR = []
# for i in range(len(P)-1):
#     V_ = I[P[i]:P[i+1]]
#     for j in V_:
#         rowCSR_Sum += V[j]"""dict.value(for j in I.keys)"""*b[j]
#     resultCSR.append(rowCSR_Sum)
#     rowCSR_Sum = 0
#         #rowSum = a[j:]
# =============================================================================
        
        
#for i in V: