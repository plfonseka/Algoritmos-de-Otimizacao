"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Alpha
"""
#Rotina para o ajuste da solução ótima

def dicotomica (f, a, b, eps, l=1e-6, maxiter=5000):
    
    xi = (a+b)/2
     
    if abs(b-a)<l:
        return xi
    
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

        if abs(b-a)<l:
            return xi
        k = k+1
    return xi

