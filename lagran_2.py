# -*- coding: utf-8 -*-
"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Método do Lagrangiano Aumentado para a função P(2)
"""
import numpy as np
from quase_newton import q_newton

def lagrangiano_2(f, x0, epsilon=1E-10, maxiter=1000):

    print("k\t xi\t\t")

########### Parâmetros Penalizadores ###############    
    ri = np.array([0.1,0.1,0.1])
    rj = np.array([0.1,0.1,0.1])
    
########### Multiplicadores de Lagrange ############ 
    lamb = np.array([0.1,0.1,0.1])
    nu = np.array([0.1,0.1,0.1])
    
    a = 15 #ajuste dos parâmetros penalizadores
    xi = np.array(x0) #ponto de partida
        
    i = 1
    while i <= maxiter:
        
############## Restrições c(x) ##################
        
        def c_1(x1,x2,x3,x4,x5):
            r1 = (x1**2+x2**2+x3**2+x4**2+x5**2-10)
            r2 = (x2*x3-5*x4*x5)
            r3 = (x1**3+x2**3+1)
            return (nu[0]*r1)+(nu[1]*r2)+(nu[2]*r3)            
        
        def c_2(x1,x2,x3,x4,x5):
            r1 = (x1**2+x2**2+x3**2+x4**2+x5**2-10)**2
            r2 = (x2*x3-5*x4*x5)**2
            r3 = (x1**3+x2**3+1)**2
            return (ri[0]*r1)+(ri[1]*r2)+(ri[2]*r3)

############## Restrições h(x) ##################

        def h_1(x1,x2,x3,x4,x5):
            r1 = 0
            r2 = 0
            r3 = 0
            return (lamb[0]*r1)+(lamb[1]*r2)+(lamb[2]*r3)
        
        def h_2(x1,x2,x3,x4,x5):
            r1 = 0
            r2 = 0
            r3 = 0
            return (rj[0]*r1)+(rj[1]*r2)+(rj[2]*r3)

        def fp(x1,x2,x3,x4,x5): #função penalizada
            return f(x1,x2,x3,x4,x5)+c_1(x1,x2,x3,x4,x5)+h_1(x1,x2,x3,x4,x5)+1/2*c_2(x1,x2,x3,x4,x5)+1/2*h_2(x1,x2,x3,x4,x5)
            
        def g(x1,x2,x3,x4,x5): #gradiente da função penalizada
            
            c1 = x1**2+x2**2+x3**2+x4**2+x5**2-10
            c2 = x2*x3-5*x4*x5
            c3 = x1**3+x2**3+1
            
            g_c1 = np.array([2*x1,2*x2,2*x3,2*x4,2*x5])
            g_c2 = np.array([0,x3,x2,-5*x5,-5*x4])
            g_c3 = np.array([3*x1**2,3*x2**2,0,0,0])
                       
            u1 = (x2*x3*x4*x5)*np.exp(x1*x2*x3*x4*x5)
            u2 = (x1*x3*x4*x5)*np.exp(x1*x2*x3*x4*x5)
            u3 = (x1*x2*x4*x5)*np.exp(x1*x2*x3*x4*x5)
            u4 = (x1*x2*x3*x5)*np.exp(x1*x2*x3*x4*x5)
            u5 = (x1*x2*x3*x4)*np.exp(x1*x2*x3*x4*x5)
            g1 = u1 + ((nu[0]+ri[0]*c1)*g_c1[0]+(nu[1]+ri[1]*c2)*g_c2[0]+(nu[2]+ri[2]*c3)*g_c3[0])
            g2 = u2 + ((nu[0]+ri[0]*c1)*g_c1[1]+(nu[1]+ri[1]*c2)*g_c2[1]+(nu[2]+ri[2]*c3)*g_c3[1])
            g3 = u3 + ((nu[0]+ri[0]*c1)*g_c1[2]+(nu[1]+ri[1]*c2)*g_c2[2]+(nu[2]+ri[2]*c3)*g_c3[2])
            g4 = u4 + ((nu[0]+ri[0]*c1)*g_c1[3]+(nu[1]+ri[1]*c2)*g_c2[3]+(nu[2]+ri[2]*c3)*g_c3[3])
            g5 = u5 + ((nu[0]+ri[0]*c1)*g_c1[4]+(nu[1]+ri[1]*c2)*g_c2[4]+(nu[2]+ri[2]*c3)*g_c3[4])
            
            return np.array([g1,g2,g3,g4,g5])
        
        def r_1(x1,x2,x3,x4,x5):
            return x1**2+x2**2+x3**2+x4**2+x5**2-10
        def r_2(x1,x2,x3,x4,x5):
            return x2*x3-5*x4*x5
        def r_3(x1,x2,x3,x4,x5):
            return x1**3+x2**3+1
        
        fxi = fp(*xi)
        
        xi2 = q_newton(fp,g,xi) #aplicação de minização irrestrita
               
        fxi2 = fp(*xi2)
                
        if abs(fxi2-fxi)<=epsilon: #Teste de convergência
            return xi2
        else:
            xi = xi2  
            
################ Atualização dos parâmetros para c(x) #######################
            def r_c(x1,x2,x3,x4,x5): 
                c1 = x1**2+x2**2+x3**2+x4**2+x5**2-10
                c2 = x2*x3-5*x4*x5
                c3 = x1**3+x2**3+1
                rt = np.array([c1,c2,c3])
                if rt[0] != 0:
                    nu[0] = nu[0]+ri[0]*r_1(*xi2)                    
                    ri[0] = ri[0]*a
                if rt[1] != 0:
                    nu[1] = nu[1]+ri[1]*r_2(*xi2)        
                    ri[1] = ri[1]*a
                if rt[2] != 0:
                    nu[2] = nu[2]+ri[2]*r_3(*xi2)                            
                    ri[2] = ri[2]*a 
                    
################ Atualização dos parâmetros para h(x) #######################            
            def r_h(x1,x2,x3,x4,x5):
                h = 0
                rt = np.array([h])                    
                if rt[0] > 0:
                    lamb[0] = lamb[0]+rj[0]*h                   
                    rj[0] = rj[0]*a
                   
            r_c(*xi2)
        print("%d\t" %(i), (*xi2))
        i = i+1
    
    print ("ERRO: Número máximo de iterações atingido!")
    return xi2
    
def f(x1,x2,x3,x4,x5):
    return 1*np.exp(x1*x2*x3*x4*x5) #função analisada

x0 = [-2,2,2,-1,-1]

solucao = lagrangiano_2(f, x0)
solut = np.reshape(solucao,(len(x0),))
sol = [round(num,3) for num in solut]


print(sol)
