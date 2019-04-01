
'''
The variables examined are:
c_pd0 - initital photodegradable group concentration
I0 - incident light intensity
L - sample thickness
D - diffusion coefficient of the free network strands
? - the imposed boundary conditions

'''
# HEADERS
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
import meshpy.triangle as triangle

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


# LOAD MASK
# Assumption - mask is black
img_path = 'mask1.png'
img = cv2.imread(img_path, 0)
#print("Image is: ", img)

# CREATE GRID
img_grid = img
# print("Image Grid is: ", img_grid)

print("size is: ", len(img_grid), " ", len(img_grid[0]) )
x_step = 1/(len(img_grid))
t_step = 1/(len(img_grid[0]))
# print("x_step is: ", x_step)
# print("t_step is: ", t_step)

x_num = int((xf-x0) / x_step)
t_num = int((tf-t0) / t_step)

#RESHAPE GRID
img_grid = img.reshape((t_num, x_num))

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

#MODULUS
def modulus(grid, modulus_grid, t0, tf, t_step, x0, xf, x_step, f, kB, T):
	A = 1-(2/f)
	for i in range (1, t_num):
		for j in range(0, x_num):
			modulus_grid[i][j] = A * grid[i][j] * kB * T
	return modulus_grid

#SWELLING
def swelling(modulus_grid, swelling_grid, t0, tf, t_step, x0, xf, x_step):
	for i in range (1, t_num):
		for j in range(0, x_num):
			swelling_grid[i][j] = modulus_grid[i][j]**0.4
	return swelling_grid

#TRIANGLE
def area(x1, y1, x2, y2, x3, y3): 
	return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0) 

def isInside(x1, y1, x2, y2, x3, y3, x, y): 
  
    # Calculate area of triangle ABC 
    A = area (x1, y1, x2, y2, x3, y3) 
  
    # Calculate area of triangle PBC  
    A1 = area (x, y, x2, y2, x3, y3) 
      
    # Calculate area of triangle PAC  
    A2 = area (x1, y1, x, y, x3, y3) 
      
    # Calculate area of triangle PAB  
    A3 = area (x1, y1, x2, y2, x, y) 
      
    # Check if sum of A1, A2 and A3 is same as A 
    return True if (A == A1 + A2 + A3) else False 

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

# print("Initial Grid: ") 
# prettyprint(grid)

# print("EXPLICIT: ")
# g = explicit(grid, ci_grid, cd_grid, cf_grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num, ci, cd, cf)
# grid = np.transpose(g)

# # MASK THE COVERED PORTIONS
# for i in range(0, x_num):
# 	for j in range(0, t_num):
# 		if img_grid[i][j] != 255:
# 			grid[i][j] = 0

# print("Resulting Grid: ")
# prettyprint(grid)

# # GRAPH 
# plt.imshow(grid, interpolation='none', cmap=cm.Blues)
# plt.show()

# # RESET GRID
# grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
# grid.fill(0.0)
# grid[0] = 1.0

print("IMPLICIT: ")
g = implicit(grid, ci_grid, cd_grid, cf_grid, t0, tf, t_step, x0, xf, x_step, alpha_pd_star, alpha_deg_star, x_num, ci, cd, cf)
grid = np.transpose(g)

# # MASK THE COVERED PORTIONS
# for i in range(0, x_num):
# 	for j in range(0, t_num):
# 		if img_grid[i][j] != 255:
# 			grid[i][j] = 0

print("Resulting Grid: ")
prettyprint(grid)

# GRAPH 
plt.imshow(grid, interpolation='none', cmap=cm.Blues)
plt.show()

# # MODULUS
# modulus_grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
# modulus_grid.fill(0.0)
# m = modulus(g, modulus_grid, t0, tf, t_step, x0, xf, x_step, f, kB, T)
# print("Modulus: ")
# prettyprint(np.transpose(m))

# # SWELLING
# swelling_grid = np.arange(x_num*t_num*1.0).reshape((t_num, x_num))
# swelling_grid.fill(0.0)
# s = swelling(m, swelling_grid, t0, tf, t_step, x0, xf, x_step)
# print("Swelling: ")
# prettyprint(np.transpose(s))

# np.polyfit(np.transpose(s), np.transpose(m), 2)
# numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)[source]¶

# TRIANGLE
#points = [ (1,1),(-1,1),(-1,-1),(1,-1)]

# n = int((len(img_grid)*len(img_grid[0]))/1000)
# points = [(n,n),(-n,n),(-n,-n),(n,-n)]

# def round_trip_connect(start, end):
#   result = []
#   for i in range(start, end):
#     result.append((i, i+1))
#   result.append((end, start))
#   return result

# info = triangle.MeshInfo()
# info.set_points(points)
# info.set_facets(round_trip_connect(0, len(points)-1))

# mesh = triangle.build(info, max_volume=0.5, min_angle=25)

# triangle.write_gnuplot_mesh("triangles.dat", mesh)

# mesh_points = np.array(mesh.points)
# print("mesh points: ", mesh_points)
# mesh_tris = np.array(mesh.elements)

# import matplotlib.pyplot as pt
# pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
# pt.show()


# temp_list = []
# lst = []
# filepath = "triangles.dat" 
# with open(filepath) as fp:
#     for cnt, line in enumerate(fp):
#         if line != '\n':
#             lst = line.split()
#             temp_list.append(float(lst[0]))
#             temp_list.append(float(lst[1]))
# fp.close()
# # print("Temp List: ", temp_list)
# # print("Length of temp is: ", len(temp_list))

# tri_list = [temp_list[x:x+6] for x in range(0, len(temp_list),8)]
# # print ("Triangles List: ", tri_list)

# # COMPUTE C,D VALUES
# def getK(x,y,z):
# 	return z

# def func(x,y,z):
# 	return 1

# cdVals = []
# for i in range (len(tri_list)):
#     tripts = []
#     #scale the points on surface gel
#     #get the corresponding coordinates x,y
#     #loop through each coordinate 
#     for i in range(0, x_num):
#     	for j in range(0, t_num):
#     		if isInside(tri_list[i][0], tri_list[i][1], tri_list[i][2], tri_list[i][3], tri_list[i][4], tri_list[i][5], i, j):
#     			tripts.append([i,j])
#     x = 1
#     y = 2
#     z = 4

#     k = getK(x,y,z)

#     a = 2*x*y*z
#     b = -4*x*y*(z**2/2)
#     c = 2*k*x*y*z #GENERALIZE 

#     d = -4*x*y*(z**2/2)
#     e = 8*(z**3/3)*x*y
#     f = -4*k*x*y*(z**2/2) #GENERALIZE

#     C = np.array([[a, b], [d, e]])
#     D = np.array([[c], [f]])
#     val = np.linalg.inv(C) @ D
#     #print(val[0], " ", val[1])
#     #cdVals.append[[val[0],val[1]]]

# #print(cdVals)




