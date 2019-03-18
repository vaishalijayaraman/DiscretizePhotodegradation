
'''
The variables examined are:
c_pd0 - initital photodegradable group concentration
I0 - incident light intensity
L - sample thickness
D - diffusion coefficient of the free network strands
? - the imposed boundary conditions

'''
# HEADERS
import math as m
import numpy as np

import scipy
from sympy import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# PARAMETERS
I0 = 20 #intensity of incident light (mWcm-2)
I_star = 0 #intensity of light (mWcm-2)

lambda_val = 365 #(nm)

alpha_pd = 8061 #absorptivity of undegraded gel (M-1cm-1)
alpha_deg = 6073 #absorptivity of degraded gel (M-1cm-1)

c_pd = 1 #concentration of undegraded gel
c_pd0 = 0.01 #initial concentration of undegraded gel
c_deg = 0 #concentration of degraded gel

k = 3.3 * (10**(-4)) #kinetic degradation constant (cm2mW-1s-1)

e = m.e #constant e

L = 0.015 #thickness of sample 

x0 = 0
xf = 1
x_step = 1/10

t0 = 0
tf = 1
t_step = 1/10

x_num = int((xf-x0) / x_step)
t_num = int((tf-t0) / t_step)

alpha_pd_star = alpha_pd*L*c_pd0 
alpha_deg_star = alpha_deg*L*c_pd0

# FUNCTIONS
# PRETTY PRINT
def prettyprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end=" ")
        print("")

# INTENSITY CALCULATION
def computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num):
	summation = 0
	for k in range (j, x_num):
		s = ((alpha_pd_star * prev[k]) + (alpha_deg_star * (1-prev[k])))/(x_num)
		summation = summation + s

	exponent = (-1 * summation)
	intensity = m.exp(exponent)

	return intensity

# EXPLICIT METHOD
def explicit(grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num):
	for i in range (1, t_num): 
		prev = grid[(i-1)]
		
		for j in range(0, x_num):
			v = computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num)
			# print("the intensity is: ", v)
			grid[i][j] = (-1 * (t_step * v * prev[j])) + prev[j]
		# print(" ")
	return grid

# NEWTON'S METHOD
def newton(f,Df,x0,epsilon,max_iter):
	xn = x0
	for n in range(0,max_iter):
		fxn = f(xn)
		if abs(fxn) < epsilon:
			#print('Found solution after',n,'iterations.')
			return xn
		Dfxn = Df(xn)
		if Dfxn == 0:
			#print('Zero derivative. No solution found.')
			return None
		xn = xn - fxn/Dfxn
	#print('Exceeded maximum iterations. No solution found.')
	return None

# IMPLICIT METHOD
def implicit(grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num):
	for i in range (1, t_num): 
		prev = grid[(i-1)]
		
		for j in range(0, x_num):
			v = computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num)

			f = lambda x: x - prev[j] + (t_step * v * x)

			Df = lambda x: 1 + (t_step * v) + (-1 * t_step * x * ((alpha_deg_star + alpha_pd_star)/x_num) * v)

			value = newton(f,Df,prev[j],1e-10,10)
			grid[i][j] = value

	return grid

# MAIN 

# SET GRID
grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
grid.fill(0.0)
grid[0] = 1.0

print("Initial Grid: ") 
prettyprint(grid)

print("EXPLICIT: ")
g = explicit(grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num)
grid = np.transpose(g)

print("Resulting Grid: ")
prettyprint(grid)

# GRAPH 
plt.imshow(grid, interpolation='none', cmap=cm.Blues)
plt.show()

# RESET GRID
grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
grid.fill(0.0)
grid[0] = 1.0

print("IMPLICIT: ")
g = implicit(grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num)
grid = np.transpose(g)

print("Resulting Grid: ")
prettyprint(grid)

# GRAPH 
plt.imshow(grid, interpolation='none', cmap=cm.Blues)
plt.show()








