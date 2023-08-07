"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Busca Dicotômica
"""

def f(x):
    return x**2 + 2*x #função a ser avaliada

#Rotina para o ajuste da solução ótima

def dicotomica (a, b, eps, l=0.0001, maxiter=100):
    xi = (a+b)/2
    
    if abs(b-a)<l:
        return xi
    print("k\t xi\t\t")
    
    k = 1 #contador
    while k < maxiter:
        xi = (a+b)/2
        lamb = xi - eps
        mi = xi + eps
        if f(lamb) < f(mi):
            b = mi
        elif f(lamb) > f(mi):
            a = lamb
        else:
            return xi
        print("%d\t %e\t" %(k, xi))
        if abs(b-a)<l:
            return xi
        k = k+1
        
    print ("ERRO: Número máximo de iterações atingido!")
    return xi
        
solucao = dicotomica(-3, 5, 0.001)
print(solucao) 
