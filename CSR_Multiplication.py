# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:02:23 2018

@author: Rohit
"""
import numpy as np

# Normal multiplication with '*' function 
def python_multiplication(a, b):
    return a*b

def loop_multiplication(a, b):
    rowSum = 0
    result = []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            rowSum += a[i,j]*b[j]
        result.append(rowSum)
        rowSum = 0
    return result
        
def csr_Multiplication(a, b, I, V, P):
    rowCSR_Sum = 0
    resultCSR = []
    for i in range(len(P)-1):
        pointerList = list(range(P[i], P[i+1]))    
        for j in pointerList:
            rowCSR_Sum += V[j]*b[I[j]]
        resultCSR.append(rowCSR_Sum)
        rowCSR_Sum = 0
    return resultCSR

# =============================================================================
# # Testing scenario 1: 
# =============================================================================
input_1 = np.matrix('1 1 0 0 0 0 0 0 ; 1 1 1 1 1 1 0 0 ; 0 0 0 2 1 0 2 1')
input_2 = np.matrix('1 1 1 1 1 1 1 1 ').T

Index1 = [0,1,0,1,2,3,4,5,3,4,6,7]
Value1 = [1,1,1,1,1,1,1,1,2,1,2,1]
Pointer1 = [0,2,8,12]

pythonProduct1 = python_multiplication(input_1, input_2)

loopProduct1 = loop_multiplication(input_1, input_2)

csr_Product1 = csr_Multiplication(input_1, input_2, Index1, Value1, Pointer1)

# =============================================================================
# # Testing Scenario 2:
# =============================================================================
input_3 = np.matrix('1 1 0 0 0 0 0 0 2; 1 1 1 1 1 1 0 0 2; 0 0 0 2 1 0 2 1 2')
input_4 = np.matrix('1 1 1 1 1 1 1 1 2').T

Index2 = [0,1,8,0,1,2,3,4,5,8,3,4,6,7,8]
Value2 = [1,1,2,1,1,1,1,1,1,2,2,1,2,1,2]
Pointer2 = [0,3,10,15]

pythonProduct2 = python_multiplication(input_3, input_4)

loopProduct2 = loop_multiplication(input_3, input_4)

csr_Product2 = csr_Multiplication(input_3, input_4, Index2, Value2, Pointer2)
    

 # =============================================================================
# rowSum = 0
# result = []
# for i in range(a.shape[0]):
#     for j in range(b.shape[0]):
#         rowSum += a[i,j]*b[j]
#     result.append(rowSum)
#     rowSum = 0
# =============================================================================       
  
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