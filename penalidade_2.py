# -*- coding: utf-8 -*-
"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Método de Penalidades para a função P(2)
"""
import numpy as np
from quase_newton import q_newton

def penalidade_2(f, x0, epsilon=1E-10, maxiter=1000):

    print("k\t xi\t\t")
    
    c = 0.1 #parâmetro penalizador
    a = 20  #ajuste do parâmetro penalizador
    xi = np.array(x0) #ponto de partida
        
    i = 1
    while i <= maxiter:
        
        def P(x1,x2,x3,x4,x5): #definição do quadrado das restrições
            
            if (x1**2+x2**2+x3**2+x4**2+x5**2-10)**2 <= 0:
                r1 = 0
            else:
                r1 = (x1**2+x2**2+x3**2+x4**2+x5**2-10)**2
    
            if (x2*x3 - 5*x4*x5)**2 <= 0:
                r2 = 0
            else:
                r2 = (x2*x3-5*x4*x5)**2

            if (x1**3+x2**3+1)**2 <= 0:
                r3 = 0
            else:
                r3 = (x1**3+x2**3+1)**2
            return r1+r2+r3

        def fp(x1,x2,x3,x4,x5): #função penalizada
            return f(x1,x2,x3,x4,x5) + c*P(x1,x2,x3,x4,x5)
        
        def g(x1,x2,x3,x4,x5): #gradiente da função penalizada
            
            p1 = x1**2+x2**2+x3**2+x4**2+x5**2-10
            p2 = x2*x3-5*x4*x5
            p3 = x1**3+x2**3+1
            
            g_p1 = np.array([2*x1,2*x2,2*x3,2*x4,2*x5])
            g_p2 = np.array([0,x3,x2,-5*x5,-5*x4])
            g_p3 = np.array([3*x1**2,3*x2**2,0,0,0])
            
            
            if P(x1,x2,x3,x4,x5) <= 0:
                g1 = (x2*x3*x4*x5)*np.exp(x1*x2*x3*x4*x5)
                g2 = (x1*x3*x4*x5)*np.exp(x1*x2*x3*x4*x5)
                g3 = (x1*x2*x4*x5)*np.exp(x1*x2*x3*x4*x5)
                g4 = (x1*x2*x3*x5)*np.exp(x1*x2*x3*x4*x5)
                g5 = (x1*x2*x3*x4)*np.exp(x1*x2*x3*x4*x5)
                return np.array([g1,g2,g3,g4,g5])
            else:
                u1 = (x2*x3*x4*x5)*np.exp(x1*x2*x3*x4*x5)
                u2 = (x1*x3*x4*x5)*np.exp(x1*x2*x3*x4*x5)
                u3 = (x1*x2*x4*x5)*np.exp(x1*x2*x3*x4*x5)
                u4 = (x1*x2*x3*x5)*np.exp(x1*x2*x3*x4*x5)
                u5 = (x1*x2*x3*x4)*np.exp(x1*x2*x3*x4*x5)
                g1 = u1 + c*(2*(p1*g_p1[0]+p2*g_p2[0]+p3*g_p3[0]))
                g2 = u2 + c*(2*(p1*g_p1[1]+p2*g_p2[1]+p3*g_p3[1]))
                g3 = u3 + c*(2*(p1*g_p1[2]+p2*g_p2[2]+p3*g_p3[2]))
                g4 = u4 + c*(2*(p1*g_p1[3]+p2*g_p2[3]+p3*g_p3[3]))
                g5 = u5 + c*(2*(p1*g_p1[4]+p2*g_p2[4]+p3*g_p3[4]))
                return np.array([g1,g2,g3,g4,g5])
        
        fxi = fp(*xi)
        
        xi2 = q_newton(fp,g,xi) #aplicação de minização irrestrita
               
        fxi2 = fp(*xi2)
                
        if abs(fxi2-fxi)<=epsilon: #atualização dos parâmetros
            return xi2
        else:
            xi = xi2
            c = c*a
        print("%d\t" %(i), (*xi2))
        i = i+1
    
    print ("ERRO: Número máximo de iterações atingido!")
    return xi2
    
def f(x1,x2,x3,x4,x5):
    return 1*np.exp(x1*x2*x3*x4*x5) #função analisada

x0 = [-2,2,2,-1,-1]

solucao = penalidade_2(f, x0)
solut = np.reshape(solucao,(len(x0),))
sol = [round(num,3) for num in solut]


print(sol)
