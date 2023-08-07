# -*- coding: utf-8 -*-
"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Método do Lagrangiano Aumentado para a função P(1)
"""
import numpy as np
from quase_newton import q_newton

def lagrangiano_1(f, x0, epsilon=1E-10, maxiter=1000):

    print("k\t xi\t\t")

########### Parâmetros Penalizadores ###############    
    ri = np.array([0.1])
    rj = np.array([0.1])
    
########### Multiplicadores de Lagrange ############ 
    lamb = np.array([0.1])
    nu = np.array([0.1])
    
    a = 100000 #ajuste dos parâmetros penalizadores
    xi = np.array(x0) #ponto de partida
        
    i = 1
    while i <= maxiter:
        
############## Restrições c(x) ##################
        
        def c_1(x1,x2):
            r1 = (x1+x2-1)
            return (nu[0]*r1)            
        
        def c_2(x1,x2):
            r1 = (x1+x2-1)**2
            return (ri[0]*r1)
        
############## Restrições h(x) ##################

        def h_1(x1,x2):
            r1 = 0
            return (lamb[0]*r1)
        
        def h_2(x1,x2):
            r1 = 0
            return (rj[0]*r1)
            
        def fp(x1,x2): #função lagrangiana aumentada
            return f(x1,x2)+c_1(x1,x2)+h_1(x1,x2)+1/2*c_2(x1,x2)+1/2*h_2(x1,x2)

        def g(x1,x2): #gradiente da função lagrangiana aumentada
            
            c1 = x1+x2-1
            
            g_c1 = np.array([1,1])
                       
            u1 = 1*x1
            u2 = 1/3*x2
            g1 = u1 + ((nu[0]+ri[0]*c1)*g_c1[0])
            g2 = u2 + ((nu[0]+ri[0]*c1)*g_c1[1])
      
            return np.array([g1,g2])
        
        def r_1(x1,x2):
            return x1+x2-1
        
        fxi = fp(*xi)
        
        xi2 = q_newton(fp,g,xi) #aplicação de minimização irrestrita
               
        fxi2 = fp(*xi2)
                
        if abs(fxi2-fxi)<=epsilon: #atualização dos parâmentros
            return xi2
        else:
            xi = xi2
            
################ Atualização dos parâmetros para c(x) #######################
            def r_c(x1,x2): 
                c1 = x1+x2-1
                rt = np.array([c1])
                if rt[0] != 0:
                    nu[0] = nu[0]+ri[0]*r_1(*xi2)                    
                    ri[0] = ri[0]*a
                    
            r_c(*xi2)                
        print("%d\t" %(i), (*xi2))
        i = i+1
    
    print ("ERRO: Número máximo de iterações atingido!")
    return xi2
    
def f(x1,x2): #função analisada
    return 1/2*(x1**2+((x2**2)/3))

x0 = [1,0]

solucao = lagrangiano_1(f, x0)
solut = np.reshape(solucao,(len(x0),))
sol = [round(num,3) for num in solut]


print(sol)