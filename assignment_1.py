import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
from matplotlib.patches import Ellipse
import math
from sympy import Eq, solve, symbols

# Correlated data from the random normally distributed data
def correlation(rand_data, mean):
	rand_data[:,0] = np.sum([rand_data[:,0], mean[0]])
	rand_data[:,1] = np.sum([rand_data[:,1], mean[1]])
	return rand_data, mean 

def ortho_transform(mean, covariance_matrix):
	eig_val, eig_vect = np.linalg.eig(covariance_matrix)

	ang = math.degrees(math.acos(eig_vect[0][0]))
	x = mean[0]
	y = mean[1]

	width = math.sqrt(eig_val[0]) * 2
	height = math.sqrt(eig_val[1]) * 2

	if (eig_vect[0][0] and eig_vect[1][0] < 0):
		eig_vect[0][0] = eig_vect[0][0] * -1.0
		eig_vect[1][0] = eig_vect[1][0] * -1.0
	elif (eig_vect[0][1] and eig_vect[1][1] < 0):
		eig_vect[0][1] = eig_vect[0][1] * -1.0
		eig_vect[1][1] = eig_vect[1][1] * -1.0

	ells = Ellipse((x, y), width, height, angle=ang)


	a = plt.subplot(111, aspect='equal')
   	ells.set_clip_box(a.bbox)
   	ells.set_alpha(0.1)
   	a.add_artist(ells)

	plt.xlim(-20, 20)
	plt.ylim(-20, 20)
	return eig_val, eig_vect

def whitening(eig_val, eig_vect, mean):
	# Transpose eigenvectors
	eig_vect[[[0][0],[1][0]]] = eig_vect[[[1][0],[0][0]]] # double check that this works
	print eig_vect

	print np.array([[1 / math.sqrt(eig_val[0]), 0], [0, 1 / math.sqrt(eig_val[1])]])
	A = np.dot(np.array([[1 / math.sqrt(eig_val[0]), 0], [0, 1 / math.sqrt(eig_val[1])]]), eig_vect)
	


	#print A
	new_mean = np.dot(A, mean)
	#print new_mean
	pass

# Returns the sample means prototypes
def sample_means(data):
	length = len(data)
	average = np.array([[0,0]], np.float64)
	for i in range(0, length):
		average += data[i]
	average = average / length
	
	return average

def minimum_euclidean_distance(z_1, z_2):
	x1, x2 = symbols('x1 x2')

	# Find discriminant functions
	g_1 = -np.dot(z_1, np.array([[x1,x2]]).T)[0][0] + np.dot(z_1, z_1.T) / 2
	g_2 = -np.dot(z_2, np.array([[x1,x2]]).T)[0][0] + np.dot(z_2, z_2.T) / 2
	
	# Get equation of line in x2=m*x1+b
	x2 = solve(g_1 - g_2, x2)
	x1 = np.linspace(0,1,10)
	
	print x2[0].as_coefficients_dict()
	x2 = np.array(x2)

	
	# # # # # # # HOW TO PLOT

#	plt.plot(x1,x2)
#	plt.show()

#	solved = solve([Eq(g_1), Eq(-g_2)], [x1, x2])
#	x_1 = solved[x1]
#	x_2 = solved[x2]








	


'''
# # # Case 1 # # #
# Class A: 
N_A = np.random.randn(200,2)
u_A = np.array([[5,10]]).T
sig_A = np.array([[8,0],[0,4]])

N_A, u_A = correlation(N_A, u_A)
ortho_transform(u_A, sig_A)
#plt.scatter(N_A[:,0], N_A[:,1])



# Class B: 
N_B = rnd.randn(200, 2)
u_B = np.array([[10,15]]).T
sig_B = np.array([[8,0],[0,4]])

N_B, u_B = correlation(N_B, u_B)
ortho_transform(u_B, sig_B)
#plt.scatter(N_B[:,0], N_B[:,1])

plt.show()


# # # Case 2 # # #
# Class C
N_C = rnd.randn(100, 2)
u_C = np.array([[5,10]]).T
sig_C = np.array([[8,4],[4,40]])

N_C, u_C = correlation(N_C, u_C)
#ortho_transform(u_C, sig_C)
plt.scatter(N_C[:,0], N_C[:,1])



# Class D
N_D = rnd.randn(200, 2)
u_D = np.array([[15,10]]).T
sig_D = np.array([[8,0],[0,8]])

N_D, u_D = correlation(N_D, u_D)
ortho_transform(u_D, sig_D)
plt.scatter(N_D[:,0], N_D[:,1])

# Class E
N_E = rnd.randn(2, 2)
u_E = np.array([[10,5]]).T
sig_E = np.array([[10,-5],[-5,20]])

N_E, u_E = correlation(N_E, u_E)
ortho_transform(u_E, sig_E)
plt.scatter(N_E[:,0], N_E[:,1])
'''


# MED Test
N_test1 = np.array([[2, 1],[3,2],[2,7],[5,2]])
N_test2 = np.array([[3, 3],[4,4],[3,9],[6,4]])

minimum_euclidean_distance(sample_means(N_test1), sample_means(N_test2))



# GED Test
'''
u_GED = np.array([[0, 10]]).T
S_GED = np.array([[16, -12],[-12, 34]])

eig_val_GED, eig_vect_GED = ortho_transform(u_GED, S_GED)
print eig_val_GED, eig_vect_GED

whitening(eig_val_GED, eig_vect_GED, u_GED)
'''

# plt.show()