# -*- coding: utf-8 -*-
"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Método de Penalidades para a função P(1)
"""
import numpy as np
from quase_newton import q_newton

def penalidade_1(f, x0, epsilon=1E-10, maxiter=1000):

    print("k\t xi\t\t")
    
    c = 0.1 #parâmetro penalizador
    a = 96 #ajuste do parâmetro penalizador
    xi = np.array(x0) #ponto de partida
    
    i = 1
    while i <= maxiter:
            
        def P(x1,x2): #definição do quadrado da restrição
            if (x1+x2-1)**2 <= 0:
                return 0
            else:
                return (x1+x2-1)**2
            
        def fp(x1,x2): #função penalizada
            return f(x1,x2) + c*P(x1,x2)
        
        def g(x1,x2): #gradiente da função penalizada
            if P(x1,x2) <= 0:
                g1 = 1*x1
                g2 = 1/3*x2
                return np.array([g1,g2])
            else:
                g1 = x1 + c*(2*(x1+x2-1))
                g2 = 1/3*x2 + c*(2*(x1+x2-1))
                return np.array([g1,g2])
        
        fxi = fp(*xi)
        
        xi2 = q_newton(fp,g,xi) #aplicação de minimização irrestrita
               
        fxi2 = fp(*xi2)
                
        if abs(fxi2-fxi)<=epsilon: #atualização dos parâmentros
            return xi2
        else:
            xi = xi2
            c = c*a
        print("%d\t" %(i), (*xi2))
        i = i+1
    
    print ("ERRO: Número máximo de iterações atingido!")
    return xi2
    
def f(x1,x2): #função analisada
    return 1/2*(x1**2+((x2**2)/3))

x0 = [0,0]

solucao = penalidade_1(f, x0)
solut = np.reshape(solucao,(len(x0),))
sol = [round(num,3) for num in solut]


print(sol)