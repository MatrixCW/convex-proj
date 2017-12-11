import numpy as np
import math
import random
import matplotlib.pyplot as plt

n = 10000
dim = 20
noise_scale = 1

A = [4.33022124, 7.74464044, 5.24166521, 1.35518629, 3.43548901, 5.74349493,
     4.7775614, 4.60552821, 4.998043, 2.73809266, 5.27669699, 5.40630708,
     0.94251011, 6.00761224, 5.53649911, 4.40277961, 3.56491275, 3.81042547,
     9.24116867, 7.90604646, 1.80325553]
A = np.array(A)

data_X = []
data_y = []

for i in range(n):
	tmp_x = np.random.rand(dim + 1) * 10
	tmp_x[-1] = 1.0
	tmp_y = A.dot(tmp_x)
	noise = np.random.normal(scale=noise_scale)
	tmp_y += noise
	data_X.append(tmp_x)
	data_y.append(tmp_y)

np.savetxt("X.csv", data_X, delimiter=",")
np.savetxt("y.csv", data_y, delimiter=",")


