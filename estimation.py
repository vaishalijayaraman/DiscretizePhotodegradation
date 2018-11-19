
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
t_step = 1/20

alpha_pd_star = alpha_pd*L*c_pd0 # (divide by 3)
alpha_deg_star = alpha_deg*L*c_pd0

# GRID
x_num = int((xf-x0) / x_step)
t_num = int((tf-t0) / t_step)
grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
grid.fill(0.0)
grid[0] = 1.0

# FUNCTIONS
# PRETTY PRINT
def prettyprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end=" ")
        print("")

# EXPLICIT METHOD
def explicit(grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num):
	for i in range (1, t_num): 
		prev = grid[(i-1)]
		
		for j in range(0, x_num):
			v = computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num)
			grid[i][j] = (-1 * (t_step * v * prev[j])) + prev[j]

	return grid

def computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num):
	summation = 0
	for k in range (j, x_num):
		s = ((alpha_pd_star * prev[k]) - (alpha_deg_star * (1-prev[k])))/(x_num)
		summation = summation + s

	exponent = (-1 * summation)
	intensity = m.exp(exponent)

	return intensity

# MAIN 
print("Initial Grid: ")
prettyprint(grid)

g = explicit(grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num)
grid = np.transpose(g)

print("Resulting Grid: ")
prettyprint(grid)

# GRAPH 
plt.imshow(grid, interpolation='none', cmap=cm.Blues)
plt.show()





