import numpy as np
import math
import random
import matplotlib.pyplot as plt

n = 10000
dim = 5
noise_scale = 5
rate = 0.0005

A = np.random.rand(dim + 1) * 10

print(A)

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

X = np.zeros((n, dim + 1))

for i in range(n):
	for j in range(dim + 1):
		X[i][j] = data_X[i][j]

y = np.zeros(n)
for i in range(n):
	y[i] = data_y[i]

# X_inv = np.linalg.inv(a)
X_trans = X.transpose()
tmp = np.linalg.inv(X_trans.dot(X))
tmp = tmp.dot(X_trans)
tmp = tmp.dot(y)
print(tmp)


def loss_func(my_A):
	global X
	global y
	global n
	loss = 0
	for i in range(n):
		loss += (y[i] - X[i].dot(my_A)) ** 2
	loss /= n
	return loss

# print("beginning gradient descent")

# current_A = np.zeros(dim + 1)
# for it in range(10000):
# 	gradient = np.zeros(dim + 1)
# 	for i in range(n):
# 		gradient = gradient + (-y[i] + X[i].dot(current_A)) * X[i]
# 	gradient = (2.0 / n) * gradient
# 	current_A = current_A - rate * gradient
# 	if it % 10 == 0:
# 		loss_func(current_A)

# print("beginning stochatic gradient descent")

# current_A = np.random.rand(dim + 1) * 10
# for it in range(10000):
# 	for i in range(n):
# 		gradient = 2.0 * (-y[i] + X[i].dot(current_A)) * X[i]
# 		current_A = current_A - rate * gradient
# 	loss_func(current_A)

loss_serial = []
loss_stale = []

start_A = np.random.rand(dim + 1) * 10
iter_cnt = 1000
print("begining mini-batch gradient descent")
batch_size = 1000
batch_cnt = n // batch_size
X_batches = []
y_batches = []
for i in range(batch_cnt):
	X_batches.append(X[i * batch_size : (i + 1) * batch_size])
	y_batches.append(y[i * batch_size : (i + 1) * batch_size])

# current_A = start_A
# for it in range(iter_cnt):
# 	X_batch_to_cal = X_batches[i % batch_cnt]
# 	y_batch_to_cal = y_batches[i % batch_cnt]
# 	gradient = np.zeros(dim + 1)
# 	for i in range(batch_size):
# 		gradient = gradient + (-y_batch_to_cal[i] + X_batch_to_cal[i].dot(current_A)) * X_batch_to_cal[i]
# 	gradient = (2.0 / batch_size) * gradient
# 	current_A = current_A - rate * gradient
# 	loss_serial.append(math.log(loss_func(current_A)))
# 	print(it)

# plt.plot([i for i in range(iter_cnt)], loss_serial, color='red', label='serial')



# staleness = [5, 7, 9, 11]
staleness = [2, 4, 6, 8, 10]
colors = ['yellow', 'blue', 'black', 'red', 'grey']
loss_stales = [[], [], [], [], []]
print("begining mini-batch stale gradient descent")

for iii in range(len(staleness)):
	all_prev_As = [start_A]
	loss_stale = loss_stales[iii]

	prev_version = 0

	for it in range(iter_cnt):
		X_batch_to_cal = X_batches[it % batch_cnt]
		y_batch_to_cal = y_batches[it % batch_cnt]

		rand_prev = int(np.random.normal(loc=staleness[iii], scale=1))
		rand_prev = max(rand_prev, 0)

		version_to_use = len(all_prev_As) - 1 - rand_prev
		version_to_use = max(version_to_use, prev_version)
		prev_version = version_to_use
		param_to_use = all_prev_As[version_to_use]

		gradient = np.zeros(dim + 1)
		for i in range(batch_size):
			gradient = gradient + (-y_batch_to_cal[i] + X_batch_to_cal[i].dot(param_to_use)) * X_batch_to_cal[i]
		gradient = (2.0 / batch_size) * gradient
		param_to_use = all_prev_As[-1] - rate * gradient
		all_prev_As.append(param_to_use)
		loss_stale.append(math.log(loss_func(param_to_use)))
		# print(it)
	plt.plot([i for i in range(iter_cnt)], loss_stale, color=colors[iii])

plt.show()

# print("beginning accelerated gradient descent")

# current_A = np.zeros(dim + 1)
# all_As = [current_A]
# for it in range(10000):
# 	gradient = np.zeros(dim + 1)
# 	for i in range(n):
# 		gradient = gradient + (-y[i] + X[i].dot(current_A)) * X[i]
# 	gradient = (2.0 / n) * gradient

# 	if len(all_As) == 1:
# 		current_A = current_A - rate * gradient
# 		all_As.append(current_A)
# 	else:
# 		momentum = ((it - 2) / (it + 1)) * (all_As[-1] - all_As[-2])
# 		current_A = current_A - rate * gradient + momentum
# 		all_As.append(current_A)
# 	if it % 10 == 0:
# 		loss_func(current_A)
