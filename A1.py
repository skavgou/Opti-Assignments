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
print(dfdx_of_fx)

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
    x=x0
    X=np.array([x]); F=np.array(fn.f(x));
    for k in range(num_iters):
        step = alpha*fn.df(x)
        x = x - step
        X=np.append(X,[x],axis=0); F=np.append(F,fn.f(x))
    return (X,F)

(X1, F1) = gradDescent(my_fun,num_iters,x0=1,alpha=0.05)
(X2, F2) = gradDescent(my_fun,num_iters,x0=1,alpha=0.5)
(X3, F3) = gradDescent(my_fun,num_iters,x0=1,alpha=1.2)
print(F1)

num_iters = len(X1) + 1
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

def q2_function(x0, x1):
    return (x0**2 + 10*x1**2) / 2

x1_vals = np.linspace(-2, 2, 100)
x0_vals = np.linspace(-2, 2, 100)
num_iters = 100
x0_meshed, x1_meshed = np.meshgrid(x0_vals, x1_vals)
y_values = q2_function(x0_meshed, x1_meshed)
plt.contour(x0_meshed, x1_meshed, y_values)
plt.title("Q2 I Contour plot")
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()

x0_sym = sp.symbols('x0')
x1_sym = sp.symbols('x1')
q2f = (x0_sym**2 + 10*x1_sym**2) / 2
q2f_func = sp.lambdify((x0_sym, x1_sym),q2f, 'numpy')
df0 = sp.diff(q2f, x0_sym)
df1 = sp.diff(q2f, x1_sym)
df0_func = sp.lambdify(x0_sym, df0, 'numpy')
df1_func = sp.lambdify(x1_sym, df1, 'numpy')


def gradDescent2Input(fn,df0_func,df1_func,num_iters,x0_start,x1_start,alpha):
    x0=x0_start
    x1=x1_start
    X=np.array([x0,x1]); F = np.array([fn(x0, x1)], dtype=float)
    for k in range(num_iters):
        #print("Iter = " + str(k))
        step0 = alpha*df0_func(x0)
        step1 = alpha*df1_func(x1)
        x0 = x0 - step0
        x1 = x1 - step1
        X=np.append(X,[x0,x1],axis=0); F = np.append(F, fn(x0, x1))
    return (X,F)

def gradDescent2InputFast(fn,df0_func,df1_func,num_iters,x0_start,x1_start,alpha):
    x0=x0_start
    x1=x1_start
    X=np.zeros((num_iters + 1, 2))
    F=np.zeros(num_iters + 1)
    X[0]=[x0,x1]; F[0] = fn(x0, x1)
    for k in range(num_iters):
        #print("Iter = " + str(k))
        step0 = alpha*df0_func(x0,x1)
        step1 = alpha*df1_func(x0,x1)
        x0 = x0 - step0
        x1 = x1 - step1
        val = fn(x0, x1)
        X[k+1] = [x0,x1]; F[k + 1] = float(fn(x0, x1))
    return (X,F)

(X1, F1) = gradDescent2Input(q2f_func,df0_func,df1_func,num_iters,x0_start=1.5,x1_start=1.5,alpha=0.05)
print(F1)
(X2, F2) = gradDescent2Input(q2f_func,df0_func,df1_func,num_iters,x0_start=1.5,x1_start=1.5,alpha=0.2)
num_iters = len(X1) + 1
plt.plot(list(range(1, num_iters)), X1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), X2, color = 'red', label='Aplha = 0.2')
plt.legend()
plt.show()
num_iters = len(F1) + 1
plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), F2, color = 'red', label='Aplha = 0.2')
plt.legend()
plt.show()

q2f2 = x**4 - 2*x**2 + 0.1*x
dfq2f2 = sp.diff(q2f2, x)
q2f2_func = sp.lambdify(x, q2f2, 'numpy')
dfq2f2_func = sp.lambdify(x, dfq2f2, 'numpy')
x_vals = np.linspace(-2, 2, 100)
y_vals = q2f2_func(x_vals)
plt.plot(x_vals, y_vals, color = 'blue', label='Aplha = 0.05')
plt.legend()
plt.show()

def gradDescent1Input(fn,fndf,num_iters,x0,alpha):
    x=x0
    X=np.array([x]); F=np.array(fn(x));
    for k in range(num_iters):
        step = alpha*fndf(x)
        x = x - step
        X=np.append(X,[x],axis=0); F=np.append(F,fn(x))
    return (X,F)

num_iters = 100
(X1, F1) = gradDescent1Input(q2f2_func,dfq2f2_func,num_iters,x0=1.5,alpha=0.05)
(X2, F2) = gradDescent1Input(q2f2_func,dfq2f2_func,num_iters,x0=-1.5,alpha=0.05)
print(X1)
print(X2)
num_iters = len(X1) + 1
plt.plot(list(range(1, num_iters)), X1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), X2, color = 'red', label='Aplha = 0.2')
plt.legend()
plt.show()
plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), F2, color = 'red', label='Aplha = 0.2')
plt.legend()
plt.show()

# Question 3

q3fx = x**2
dfq3fx = sp.diff(q3fx, x)
q3fx_func = sp.lambdify(x, q3fx, 'numpy')
dfq3fx_func = sp.lambdify(x, dfq3fx, 'numpy')

num_iters = 100
(X1, F1) = gradDescent1Input(q3fx_func,dfq3fx_func,num_iters,x0=1,alpha=0.05)
(X2, F2) = gradDescent1Input(q3fx_func,dfq3fx_func,num_iters,x0=1,alpha=0.01)
(X3, F3) = gradDescent1Input(q3fx_func,dfq3fx_func,num_iters,x0=1,alpha=1.01)

num_iters = len(X1) + 1
plt.plot(list(range(1, num_iters)), X1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), X2, color = 'red', label='Aplha = 0.01')
plt.plot(list(range(1, num_iters)), X3, color = 'orange', label='Aplha = 1.01')
plt.legend()
plt.title("xk vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("xk")
plt.yscale("log")
plt.show()

plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='Aplha = 0.05')
plt.plot(list(range(1, num_iters)), F2, color = 'red', label='Aplha = 0.01')
plt.plot(list(range(1, num_iters)), F3, color = 'orange', label='Aplha = 1.01')
plt.legend()
plt.title("f(xk) vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("f(xk)")
plt.yscale("log")
plt.show()

γ = sp.symbols('γ')
q3fx2 = γ*x**2
q3fx2_1 = 0.5*x**2
q3fx2_2 = 1*x**2
q3fx2_3 = 2*x**2
q3fx2_4 = 5*x**2
dfq3fx2 = sp.diff(q3fx2, x)
dfq3fx2_1 = sp.diff(q3fx2_1, x)
dfq3fx2_2 = sp.diff(q3fx2_2, x)
dfq3fx2_3 = sp.diff(q3fx2_3, x)
dfq3fx2_4 = sp.diff(q3fx2_4, x)
q3fx2_func = sp.lambdify(x, q3fx2, 'numpy')
dfq3fx2_func = sp.lambdify(x, dfq3fx2, 'numpy')

num_iters = 200
(X1, F1) = gradDescent1Input(sp.lambdify(x, q3fx2_1, 'numpy'),
                             sp.lambdify(x, dfq3fx2_1, 'numpy'),num_iters,x0=1,alpha=0.1)
(X2, F2) = gradDescent1Input(sp.lambdify(x, q3fx2_2, 'numpy'),
                             sp.lambdify(x, dfq3fx2_2, 'numpy'),num_iters,x0=1,alpha=0.1)
(X3, F3) = gradDescent1Input(sp.lambdify(x, q3fx2_3, 'numpy'),
                             sp.lambdify(x, dfq3fx2_3, 'numpy'),num_iters,x0=1,alpha=0.1)
(X4, F4) = gradDescent1Input(sp.lambdify(x, q3fx2_4, 'numpy'),
                             sp.lambdify(x, dfq3fx2_4, 'numpy'),num_iters,x0=1,alpha=0.1)

num_iters = len(X1) + 1
plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='γ = 0.5')
plt.plot(list(range(1, num_iters)), F2, color = 'red', label='γ = 1')
plt.plot(list(range(1, num_iters)), F3, color = 'orange', label='γ = 2')
plt.plot(list(range(1, num_iters)), F4, color = 'green', label='γ = 5')
plt.legend()
plt.title("f(xk) vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("f(xk)")
plt.yscale("log")
plt.show()

# Q3 part 3

q3fx3 = sp.Abs(x)
dfq3fx3 = sp.sign(x)
q3fx3_func = sp.lambdify(x, q3fx3, 'numpy')
dfq3fx3_func = sp.lambdify(x, dfq3fx3, 'numpy')

num_iters = 60
(X1, F1) = gradDescent1Input(q3fx3_func, dfq3fx3_func,num_iters,x0=1,alpha=0.1)
num_iters = len(X1) + 1
plt.plot(list(range(1, num_iters)), X1, color = 'blue', label='γ = 5')
plt.title("xk vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("xk")
plt.legend()
plt.show()

plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='γ = 5')
plt.title("f(xk) vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("f(xk)")
plt.legend()
plt.show()

x2_sym = sp.symbols('x2')
q4fx = x1_sym**2 + γ*x2_sym**2
q4fx_func = sp.lambdify((x1_sym, x2_sym, γ), q4fx, 'numpy')
q4fx_1 = x1_sym**2 + 1*x2_sym**2
q4fx_2 = x1_sym**2 + 4*x2_sym**2
q4fx_func1 = sp.lambdify((x1_sym, x2_sym), q4fx_1, 'numpy')
q4fx_func2 = sp.lambdify((x1_sym, x2_sym), q4fx_2, 'numpy')
dfq4fx_x1_1 = sp.diff(q4fx_1, x1_sym)
dfq4fx_x2_1 = sp.diff(q4fx_1, x2_sym)
dfq4fx_x1_2 = sp.diff(q4fx_2, x1_sym)
dfq4fx_x2_2 = sp.diff(q4fx_2, x2_sym)
dfq4fx_x1_1_func = sp.lambdify(x1_sym, dfq4fx_x1_1, 'numpy')
dfq4fx_x2_1_func = sp.lambdify(x2_sym, dfq4fx_x2_1, 'numpy')
dfq4fx_x1_2_func = sp.lambdify(x1_sym, dfq4fx_x1_2, 'numpy')
dfq4fx_x2_2_func = sp.lambdify(x2_sym, dfq4fx_x2_2, 'numpy')

num_iters = 100
(X1, F1) = gradDescent2Input(q4fx_func1,dfq4fx_x1_1_func,dfq4fx_x2_1_func,num_iters,x0_start=1,x1_start=1,alpha=0.1)
(X2, F2) = gradDescent2Input(q4fx_func2,dfq4fx_x1_2_func,dfq4fx_x2_2_func,num_iters,x0_start=1,x1_start=1,alpha=0.1)

x1_vals = np.linspace(-1.5, 1.5, 100)
x2_vals = np.linspace(-1.5, 1.5, 100)
num_iters = 100
x1_meshed, x2_meshed = np.meshgrid(x1_vals, x2_vals)
y_values1 = q4fx_func(x1_meshed, x2_meshed, 1)
y_values2 = q4fx_func(x1_meshed, x2_meshed, 4)
plt.contour(x1_meshed, x2_meshed, y_values1, label='γ = 1')
# plt.title("Q2 I Contour plot - γ = 1")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.legend()
# plt.show()
plt.contour(x1_meshed, x2_meshed, y_values2, label='γ = 4')
plt.title("Q2 I Contour plot - γ = 4")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

num_iters = len(X1) + 1
plt.plot(list(range(1, num_iters)), X1, color = 'blue', label='γ = 1')
plt.plot(list(range(1, num_iters)), X2, color = 'red', label='γ = 4')
plt.legend()
plt.show()

num_iters = len(F1) + 1
plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='γ = 1')
plt.plot(list(range(1, num_iters)), F2, color = 'red', label='γ = 4')
plt.legend()
plt.show()

q4f2 = (1 - x1_sym)**2 + 100*(x2_sym - x1_sym**2)**2
q4f2_df1 = sp.diff(q4f2, x1_sym)
q4f2_df2 = sp.diff(q4f2, x2_sym)

q4f2_func = sp.lambdify((x1_sym, x2_sym), q4f2, 'numpy')
q4f2_df1_func = sp.lambdify((x1_sym, x2_sym), q4f2_df1, 'numpy')
q4f2_df2_func = sp.lambdify((x1_sym, x2_sym), q4f2_df2, 'numpy')

x1_vals = np.linspace(-2, 2, 100)
x2_vals = np.linspace(-1, 3, 100)
num_iters = 100
x1_meshed, x2_meshed = np.meshgrid(x1_vals, x2_vals)
y_values = q4f2_func(x1_meshed, x2_meshed)
plt.contour(x1_meshed, x2_meshed, y_values, colors='blue')
plt.show()

num_iters = 2000
(X1, F1) = gradDescent2InputFast(q4f2_func,q4f2_df1_func,q4f2_df2_func,num_iters,x0_start=-1.25,x1_start=0.5,alpha=0.001)
(X2, F2) = gradDescent2InputFast(q4f2_func,q4f2_df1_func,q4f2_df2_func,num_iters,x0_start=-1.25,x1_start=0.5,alpha=0.005)

num_iters = len(X1) + 1
plt.plot(list(range(1, num_iters)), X1, color = 'blue', label='γ = 1')
plt.plot(list(range(1, num_iters)), X2, color = 'red', label='γ = 4')
plt.legend()
plt.show()

num_iters = len(F1) + 1
plt.plot(list(range(1, num_iters)), F1, color = 'blue', label='γ = 1')
plt.plot(list(range(1, num_iters)), F2, color = 'red', label='γ = 4')
plt.legend()
plt.show()

plt.contour(x1_meshed, x2_meshed, y_values)
x1s_1 = []
x2s_1 = []
for x1 in X1:
    x1s_1.append(x1[0])
    x2s_1.append(x1[1])
x1s_2 = []
x2s_2 = []
for x1 in X2:
    x1s_2.append(x1[0])
    x2s_2.append(x1[1])
plt.plot(x1s_1, x2s_1, color = 'green', label='γ = 1')
plt.plot(x1s_2, x2s_2, color = 'red', label='γ = 4')
plt.legend()
plt.show()
