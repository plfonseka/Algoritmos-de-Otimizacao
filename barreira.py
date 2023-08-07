# -*- coding: utf-8 -*-
"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Método da Barreira
"""
import numpy as np
from quase_newton import q_newton

def barreira(f, x0, epsilon=1E-10, maxiter=1000):

    print("i\t x_opt\t\t")
    
    c = 10 #parâmetro penalizador

    xi = np.array(x0) #ponto de partida
    
    i = 1
    while i <= maxiter:
            
        def B(x1,x2):
            return 1/((x1**2)-x2)

        def fp(x1,x2): #função penalizada
            return f(x1,x2) - c*B(x1,x2)
        
        def g(x1,x2): #gradiente da função penalizada
        
            g_p1 = np.array([(2*x1/(((x1**2)-x2)**2)),
                             (-1/(((x1**2)-x2)**2))])
            u1 = 2*x1 - 4*x2 + 4*((x1 - 2)**3)
            u2 = -4*x1 + 8*x2
            g1 = u1 + c*g_p1[0]
            g2 = u2 + c*g_p1[1] 
            return np.array([g1,g2])
        
        fxi = fp(*xi)
        
        xi2 = q_newton(fp,g,B,xi) #aplicação de minimização irrestrita
               
        fxi2 = fp(*xi2)
        
        if abs(fxi2-fxi)<epsilon: #teste de convergência
            return xi2
        else:
            xi = xi2
            c = c/10
        print("%d\t" %(i), (*xi2))
        i = i+1
    
    print ("ERRO: Número máximo de iterações atingido!")
    return xi2
    
def f(x1,x2): #função analisada
    return ((x1-2)**4)+((x1-(2*x2))**2)

x0 = [-1,0]

solucao = barreira(f, x0)
solut = np.reshape(solucao,(len(x0),))
sol = [round(num,3) for num in solut]


print(sol)