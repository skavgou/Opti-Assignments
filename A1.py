import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sympy.physics.paulialgebra import delta
import decimal
decimal.getcontext().prec = 100

x = sp.symbols('x')
fx = x**4
dfdx_of_fx = sp.diff(fx, x)
df_func = sp.lambdify(x, dfdx_of_fx, 'numpy')

x_el_of = np.linspace(-2, 2, 100)
exact_points = df_func(x_el_of)

def non_sympy_func(x):
    return x**4

def forward_finite_approx_func(delta):
    return (non_sympy_func(x_el_of + delta) - non_sympy_func(x_el_of)) / delta

forward_finite_approx = forward_finite_approx_func(0.01)

deltas = np.linspace(0.001, 1, 100)

mean_absolute_errors = []

for delta in deltas:
    xs = forward_finite_approx_func(delta)
    mean_absolute_errors.append(mean_absolute_error(exact_points,xs))

plt.plot(x_el_of, exact_points, color='blue', label='Exact')
plt.plot(x_el_of, forward_finite_approx, color='red', label='Forward')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

plt.plot(deltas, mean_absolute_errors, color='blue', label='Exact')
plt.xlabel('delta')
plt.ylabel('MAE')
plt.show()

class myFun():
    def f(self, x):
        return float(decimal.Decimal(x)**4) # function value f(x)
    def df(self, x):
        return float(4*(decimal.Decimal(x))**3) # derivative of f(x)

my_fun = myFun()

num_iters = 50

def gradDescent(fn,num_iters,x0,alpha):
    x=x0;
    X=np.array([x]); F=np.array(fn.f(x));
    for k in range(num_iters):
        step = alpha*fn.df(x)
        x = x - step
        X=np.append(X,[x],axis=0); F=np.append(F,fn.f(x))
    return (X,F)

(X1, F1) = gradDescent(my_fun,num_iters,x0=1,alpha=0.05)
(X2, F2) = gradDescent(my_fun,num_iters,x0=1,alpha=0.5)
(X3, F3) = gradDescent(my_fun,num_iters,x0=1,alpha=1.2)

num_iters+=2
plt.plot(list(range(1, num_iters)), X1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), X2, color = 'green', label='Aplha = 0.5')
#plt.plot(list(range(1, num_iters)), X3, color = 'red', label='Aplha = 1.2')
plt.legend()
plt.show()

plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), F2, color = 'green', label='Aplha = 0.5')
#plt.plot(list(range(1, num_iters)), F3, color = 'red', label='Aplha = 1.2')
plt.legend()
plt.show()

