import numpy as np

n = 10000
dim = 5
noise_scale = 5
rate = 0.005

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

print("beginning gradient descent")

current_A = np.zeros(dim + 1)
for it in range(10000):
	gradient = np.zeros(dim + 1)
	for i in range(n):
		gradient = gradient + (-y[i] + X[i].dot(current_A)) * X[i]
	gradient = (2.0 / n) * gradient
	current_A = current_A - rate * gradient
	if it % 10 == 0:
		print(current_A)

print(current_A)
