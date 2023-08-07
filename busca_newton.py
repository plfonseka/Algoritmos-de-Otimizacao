"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Busca por Método de Newton
"""

def f(x):
    return x**2 + 2*x
def df(x):
    return 2*x + 2
def d2f(x):
    return 2

def newton (f, df, d2f, x0, eps, maxiter=100):
    if abs(f(x0))<=eps:
        return x0
    print("k\t xi\t\t")
    
    k = 1
    while k < maxiter:
        xi = x0 - (df(x0)/d2f(x0))
        if abs(f(xi) <= eps):
            return xi
        print("%d\t %e\t %e" %(k, xi, f(xi)))
        x0 = xi
        k=k+1
    print ("ERRO: Número máximo de iterações atingido!")
    return xi
        
solucao = newton(f, df, d2f, -3, 0.0001)
print(solucao)

