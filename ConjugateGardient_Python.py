# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:36:02 2018

@author: Rohit
"""

# =============================================================================
# import numpy as np
# a = np.array([5,4])[np.newaxis]
# print(a)
# print(a.T)
# 
# function [x] = conjgrad(A, b, x)
#     r = b - A * x;
#     p = r;
#     rsold = r' * r;
# 
#     for i = 1:length(b)
#         Ap = A * p;
#         alpha = rsold / (p' * Ap);
#         x = x + alpha * p;
#         r = r - alpha * Ap;
#         rsnew = r' * r;
#         if sqrt(rsnew) < 1e-10
#               break;
#         end
#         p = r + (rsnew / rsold) * p;
#         rsold = rsnew;
#     end
# end
# =============================================================================

import numpy as np

def ConjGrad(a, b, x):
    r = (b - np.dot(a, x));
    p = r;
    rsold = np.dot(r.T, r);
    
    for i in range(len(b)):
        a_p = np.dot(a, p);
        alpha = rsold / np.dot(p.T, a_p);
        x = x + (alpha * p);
        r = r - (alpha * a_p);
        rsnew = np.dot(r.T, r);
        if (np.sqrt(rsnew) < (10 ** -5)):
            break;
        p = r + ((rsnew / rsold) * p);
        rsold = rsnew;
        
    return p

a = np.array([[3, 2, -1], [2, -1, 1], [-1, 1, -1]]) # 3X3 symmetric matrix
b = (np.array([1, -2, 0])[np.newaxis]).T  # 3X1 matrix
x = (np.array([0, 1, 2])[np.newaxis]).T
    
val = ConjGrad(a, b, x);
print(val)



    
    