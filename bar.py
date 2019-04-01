import numpy as np

def getK(x,y,z):
	return z

def func(x,y,z):
	return 1

x = 1
y = 2
z = 4

k = getK(x,y,z)

a = 2*x*y*z
b = -4*x*y*(z**2/2)
c = 2*k*x*y*z #GENERALIZE 

d = -4*x*y*(z**2/2)
e = 8*(z**3/3)*x*y
f = -4*k*x*y*(z**2/2) #GENERALIZE

# print("a : ", a)
# print("b : ", b)
# print("c : ", c)
# print("d : ", d)
# print("e : ", e)
# print("f : ", f)

C = np.array([[a, b], [d, e]])
D = np.array([[c], [f]])
val = np.linalg.inv(C) @ D
print(val[0], " ", val[1])
