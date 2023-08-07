# -*- coding: utf-8 -*-
"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Método Quase Newton com Busca Unidimensional Secante 
"""
import numpy as np
from Secante import secante

def q_newton(f,g,x0,epsilon,maxiter=5000):
    
    print("k\t x_opt\t\t")

    k=1
    xi = np.array(x0) #ponto de partida
    d_i = np.identity(len(x0))
    
    while k < maxiter:
                
        fxi = f(*xi)
        
        grad = g(*xi)
        
        di = grad.dot(-d_i)
                    
        alpha_opt = secante(g,di,xi)
        
        x_opt = xi + alpha_opt*di
        
        gd = g(*x_opt)
        
        fxi2 = f(*x_opt)
        
        if abs(fxi2-fxi)<epsilon:
            return x_opt
        
        s_i = x_opt - xi
        y_i = gd - grad
        
        si = np.reshape(s_i,(1,len(x0)))
        yi = np.reshape(y_i,(1,len(x0)))

        if si.dot(np.transpose(yi)) > 0:
            j = np.transpose(si).dot(si)/si.dot(np.transpose(yi))
            l = ((d_i.dot(np.transpose(yi))).dot(yi)).dot(d_i)/(yi.dot(d_i)).dot(np.transpose(yi))
            d_i2 = d_i+j-l
        else :
            d_i2 = np.identity(len(x0))
        
        #print(d_i2)
        print("%d\t" %(k), (*x_opt))
        
        d_i = d_i2
        xi = np.array([*x_opt]) #ponto de partida

        k=k+1
        
    print ("ERRO: Número máximo de iterações atingido!")
    return x_opt

