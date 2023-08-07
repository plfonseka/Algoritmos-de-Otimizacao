"""
Universidade Estadual de Campinas
IM 380 - Otimização de Sistemas Mecânicos
Pedro Lucas - Ra 263117

Busca por Fibonacci
"""

fb = [1,1,2,3,5,8,13,21,34,55,89,144,233,377] #parte da sequência de Fibonacci

#n = 6 #contante que determina o valor de Fn
#a = -3 #limite inferior do intervalo de incerteza
#b = 5 #limite superior do intervalo de incerteza

def f(x):
    return x**2 + 2*x #função a ser avaliada

#Rotina para a solução por fibonacci
def fibonacci (a, b, n, maxiter=100):
    l = (a+b)/ fb[n]
    xi = (a+b)/ 2
    
    if (b-a) < l:
        return xi
    print("k\t xi\t\t")

    k = 1
    while k < maxiter:
        xi = (a+b)/2
        lamb = a + (1 - (fb[n-k+1]/fb[n-k+2])) * (b - a)
        mi = b - (1 - (fb[n-k+1]/fb[n-k+2])) * (b - a)
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
        
solucao = fibonacci(-3, 5, 10)
print(solucao)