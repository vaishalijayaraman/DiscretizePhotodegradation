
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

import cv2

# PARAMETERS
I0 = 20 #intensity of incident light (mWcm-2)
I_star = 0 #intensity of light (mWcm-2)

lambda_val = 365 #(nm)

alpha_pd = 8061 #absorptivity of undegraded gel (M-1cm-1)
alpha_deg = 6073 #absorptivity of degraded gel (M-1cm-1)

c_pd = 1 #concentration of undegraded gel
c_pd0 = 0.01 #initial concentration of undegraded gel
c_deg = 0 #concentration of degraded gel

ci_0 = 1 #intanct
cd_0 = 0 #dangling
cf_0 = 0 #free

ci = 1 #intanct
cd = 0 #dangling
cf = 0 #free

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

D = 10**-6 #diffusion coefficient
D = 0
D_star = D/(k * I0 * (L**2)) 

f = 20 #cross-link functionality
kB = 1.38064852 * 10**-23 #boltzmann constant in m2 kg s-2 K-1
T = 1 #absolute temperature in K

# FUNCTIONS
# PRETTY PRINT
def prettyprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end=" ")
        print("")

# INTENSITY CALCULATION
def computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num, ci, cd, cf):
	summation = 0
	for k in range (j, x_num):
		s = ((alpha_pd_star * (ci + (cd/2))) + (alpha_deg_star * ((cf + (cd/2)))))/(x_num)
		summation = summation + s

	exponent = (-1 * summation)
	intensity = m.exp(exponent)

	return intensity

# EXPLICIT METHOD
def explicit(grid, ci_grid, cd_grid, cf_grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num, ci, cd, cf):
	for i in range (1, t_num): 
		prev = grid[(i-1)]

		prev_ci = ci_grid[(i-1)]
		prev_cd = cd_grid[(i-1)]
		prev_cf = cf_grid[(i-1)]
		
		for j in range(0, x_num):
			mult = prev_ci[j] + (prev_cd[j]/2)
			mult_0 = ci_0 + (cd_0/2)

			int_star = computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num, prev_ci[j], prev_cd[j], prev_cf[j])

			ci = prev_ci[j] + (-2 * int_star * mult**2) * t_step

			cd = prev_cd[j] + (((-2 * int_star * (mult_0 - 2*(mult))) * mult) * t_step)

			diff_f = get_diff_f(x0, xf, x_step, x_num, cf_grid, i, j)
			cf = prev_cf[j] + ((((-2 * int_star * (mult - mult_0)) * mult) + (D_star * diff_f)) * t_step) #diffusion

			ci_grid[i][j] = ci
			cd_grid[i][j] = cd
			cf_grid[i][j] = cf

			grid[i][j] = prev[j] + (-1 * (t_step) * int_star * mult)

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
			#return None
			return 0
		xn = xn - fxn/Dfxn
	#print('Exceeded maximum iterations. No solution found.')
	#return None
	return 0

# IMPLICIT METHOD
def implicit(grid, ci_grid, cd_grid, cf_grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num, ci, cd, cf):
	for i in range (1, t_num): 
		prev = grid[(i-1)]

		prev_ci = ci_grid[(i-1)]
		prev_cd = cd_grid[(i-1)]
		prev_cf = cf_grid[(i-1)]
		
		for j in range(0, x_num):

			int_star = computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num, prev_ci[j], prev_cd[j], prev_cf[j])

			f_ci = lambda x: x - prev_ci[j] + (2 * t_step * int_star * (x**2)) + ((0.5) * t_step * int_star * (prev_cd[j]**2)) + (2 * t_step * int_star * x * prev_cd[j])
			Df_ci = lambda x: 1 + (4 * t_step * int_star) - (2 * t_step * (x**2) * (alpha_pd_star/x_num) * int_star) + (2 * t_step * int_star * prev_cd[j]) - (2 * t_step * x * prev_cd[j] * (alpha_pd_star/x_num) * int_star)
			value_ci = newton(f_ci,Df_ci,prev_ci[j],1e-10,10)
			ci_grid[i][j] = value_ci

			f_cd = lambda x: x - prev_cd[j] + (2 * t_step * int_star * (((ci_0) + (cd_0/2)) - (2 * prev_ci[j]) - x) * (prev_ci[j] + x/2))
			Df_cd = lambda x: 1 + (2 * t_step * int_star * ((-1 * (prev_ci[j] + (x/2))) + (0.5 * (((ci_0) + (cd_0/2)) - (2 * prev_ci[j]) - x)))) - (2 * t_step * int_star * ((((ci_0) + (cd_0/2)) - (2 * prev_ci[j]) - x) * (prev_ci[j] + x/2)) * ((alpha_pd_star + alpha_deg_star)/(2*x_num))) 
			value_cd = newton(f_cd,Df_cd,prev_cd[j],1e-10,10)
			cd_grid[i][j] = value_cd

			f_cf = lambda x: x - prev_cf[j] + (2 * t_step * int_star * (prev_ci[j] + (prev_cd[j]/2)) * (prev_ci[j] + (prev_cd[j]/2) - ci_0 - (cd_0/2)))
			Df_cf = lambda x: 1 - (2 * t_step * ((prev_ci[j] + (prev_cd[j]/2)) * (prev_ci[j] + (prev_cd[j]/2) - ci_0 - (cd_0/2))) * int_star * (alpha_deg_star/x_num))
			value_cf = newton(f_cf,Df_cf,prev_cf[j],1e-10,10)
			diff_f = get_diff_f(x0, xf, x_step, x_num, cf_grid, i, j)
			cf_grid[i][j] = value_cf + (D_star * diff_f * t_step) #diffusion

			#print("the vals are: ", value_ci, " and ", value_cd, " and ", value_cf)
			value = computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num, value_ci, value_cd, value_cf)
			mult = value_ci + (value_cd/2)
			grid[i][j] = prev[j] + (-1 * (t_step) * value * mult)
			#print("match is: ", grid[i][j])
		#print("next i")

	return grid

# DIFFUSION TERM
def get_diff_f(x0, xf, x_step, x_num, cf_grid, i, j):
	v = 0
	vals = cf_grid[i-1]

	if j==0:
		v = vals[j+1] - vals[j]
	elif j==x_num-1:
		v = vals[j-1] - vals[j]
	else:
		vm= vals[j+1] - 2*vals[j] + vals[j-1]

	v = v/(x_num**2)
	return v

# MAIN 

# SET GRID
grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
grid.fill(0.0)
grid[0] = 1.0

ci_grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
ci_grid.fill(0.0)
ci_grid[0] = 1.0

cd_grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
cd_grid.fill(0.0)

cf_grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
cf_grid.fill(0.0)

print("Initial Grid: ") 
prettyprint(grid)

print("EXPLICIT: ")
g = explicit(grid, ci_grid, cd_grid, cf_grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num, ci, cd, cf)
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
g = implicit(grid, ci_grid, cd_grid, cf_grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num, ci, cd, cf)
grid = np.transpose(g)

print("Resulting Grid: ")
prettyprint(grid)

# GRAPH 
plt.imshow(grid, interpolation='none', cmap=cm.Blues)
plt.show()









