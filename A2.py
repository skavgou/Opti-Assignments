import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sympy.benchmarks.bench_meijerint import alpha
from sympy.physics.paulialgebra import delta
import decimal
import math

from sympy.plotting import plot3d

# from A1 import num_iters

decimal.getcontext().prec = 100

def polyak_alpha(fn, df, x, f_star, e):
    numer = fn(*x) - f_star
    slope = np.array(df(*x), dtype=float)
    denom = np.sum( slope**2 ) + e
    return numer / denom


def gradDescent2InputPolyak(fn,df,num_iters,x0_start,x1_start,f_star,e):
    x0=x0_start
    x1=x1_start
    X=np.array([x0,x1]); F = np.array([fn(x0, x1)], dtype=float)
    for k in range(num_iters):
        alpha = polyak_alpha(fn,df,(x0, x1),f_star,e)
        print(alpha)
        step = alpha * np.array(df(x0, x1), dtype=float)
        x0 = x0 - step[0]
        x1 = x1 - step[1]
        X=np.append(X,[x0,x1],axis=0); F = np.append(F, fn(x0, x1))
    return (X,F)

def gradDescent2InputRMSProp(fn,df,num_iters,x0_start,x1_start,alpha_start,beta,e):
    x0=x0_start
    x1=x1_start
    sum = 0
    X=np.array([x0,x1]); F = np.array([fn(x0, x1)], dtype=float)
    for k in range(num_iters):
        sum = beta * sum + ((1 - beta) * (df(x) ^ 2))
        alpha = alpha_start / ( math.sqrt(sum) + e)
        step = alpha * np.array(df(x0, x1), dtype=float)
        x0 = x0 - step[0]
        x1 = x1 - step[1]
        X=np.append(X,[x0,x1],axis=0); F = np.append(F, fn(x0, x1))
    return (X,F)

x = sp.symbols('x')
y = sp.symbols('y')
q1fx = x**2 + 100*y**2
q1fx_func = sp.lambdify((x,y),q1fx)
print(q1fx_func(1,2))

q1df = [sp.diff(q1fx, x), sp.diff(q1fx, y)]
q1df_func = sp.lambdify((x, y), q1df)
vec = q1df_func(1,1)
np.sum(vec)
X,F = gradDescent2InputPolyak(q1fx_func,q1df_func,200,2,2,1e-8,0)
num_iters = len(F) + 1
plt.plot(list(range(1, num_iters)), F, color = 'blue', label='Aplha = 0.05')
plt.legend()
plt.title("xk vs iteration")
plt.ylabel("f(xk)")
plt.xlabel("iteration")
plt.yscale("log")
plt.show()