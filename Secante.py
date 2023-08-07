# -*- coding: utf-8 -*-
"""
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Busca Unidimensional Secante
"""
import numpy as np

def secante(g, d, x0, b=0.1, maxiter=1000):
    
    a0 = 0
    a = 1
    di = d

    g0 = g(*x0).dot(np.transpose(di))
    gin = g0
    
    res = 0
    
    while res==0:
        a = 2*a
        x1 = x0 + a*di
        g1 = g(*x1).dot(np.transpose(di))
        if ((g1*g0)<0) == True:
            res = 1
        
    k=1
    g2 = gin
    
    while (((abs(g2)) > (abs(b*gin))) == True) and k<maxiter:
        a2 = a0 - ((a - a0)/(g1 - g0))*g0
        x2 = x0 + a2*di
        g2 = g(*x2).dot(np.transpose(di))
        if ((g0*g2)<0) == True:
            a = a2
            g1 = g2
        else:
            a0 = a2
            g0 = g2
        
        k=k+1
        
    return a2    
