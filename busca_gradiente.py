# -*- coding: utf-8 -*-
"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Método do Gradiente
"""
import numpy as np
from alpha import dicotomica

def gradiente(f,g,x0,epsilon,maxiter=5000):
    
    k=1
    xi = np.array(x0) #ponto de partida
    
    while k < maxiter:
                
        fxi = f(*xi)
        
        grad = g(*xi)
        
        di = -grad
        
        def fobj(alpha):
            xaux = xi + alpha*-grad
            return f(*xaux)
        
        alpha_opt = dicotomica (fobj, 0, 1, 0.005)
        
        x_opt = xi + alpha_opt * di
        
        fxi2 = f(*x_opt)
        
        if abs(fxi2-fxi)<epsilon:
            return x_opt

        print("%d\t" %(k), (*x_opt))
        
        xi = np.array([*x_opt])

        k=k+1
        
    print ("ERRO: Número máximo de iterações atingido!")
    return x_opt
    

